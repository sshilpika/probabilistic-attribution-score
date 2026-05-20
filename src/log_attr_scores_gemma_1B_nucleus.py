from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import time
import torch
from logit_processor import ForcedSequenceProcessor
import os
import random
import numpy as np

random.seed(0)  # Sets the seed for Python's built-in random module
np.random.seed(0)  # Sets the seed for NumPy's random number generator

torch.manual_seed(0)  # Sets the seed for PyTorch's CPU random number generator
torch.cuda.manual_seed(0)  # Sets the seed for the current GPU device
torch.cuda.manual_seed_all(0)  # Sets the seed for all available GPU devices
# Determinism/precision knobs
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)  # may raise for some ops
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # for CUDA matmul determinism
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False  # Disables TF32 on cuDNN
torch.set_float32_matmul_precision("high")  # avoids TF32 on some setups
#disable cuDNN entirely
# https://www.ingonyama.com/oldblogs/solving-reproducibility-challenges-in-deep-learning-and-llms-our-journey
torch.backends.cudnn.enabled = False 


class ProbabilisticAttributionScores():
    
    def __init__(self, model, tokenizer, 
                 prompt : str = "", 
                 resp_length : int = 20, 
                 padding : bool = True, 
                 device : str = None, 
                 do_sample : bool = False,
                eos_token_id : int = None, set_bos=True,
                top_p: int = None, temperature:int = None, top_k: int = None, repetition_penalty: float = 1.0):


        
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.resp_length = resp_length
        self.padding = padding
        self.top_p = top_p
        self.temperature = temperature
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.return_tensors="pt"
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.do_sample = do_sample
        self.num_beans = 1
        self.eos_token_id = eos_token_id
        self.pad_token_id = self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.eos_token_id
        self.output_scores = True
        self.return_dict_in_generate = True
        self.inputs = self.tokenizer(self.prompt, 
                                     return_tensors = self.return_tensors, 
                                     padding = self.padding).to(self.device)
        self.vocab_size = len(tokenizer)
        self.input_ids_ori = self.inputs["input_ids"]
        self.input_ids = self.add_bos(self.input_ids_ori)
        self.ori_response = ""
        self.ori_response_ids = None
        self.ori_response_all = None
        self.ori_response_ids_all = None
        self.max_length = self.inputs['input_ids'].shape[-1] + self.resp_length

    def add_bos(self, input_ids):
        bos = torch.full((input_ids.size(0), 1),
                         self.bos_token_id, dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([bos, input_ids], dim=1)

    def generate_response_string_all(self, logits):
        
        logits_max = torch.max(logits, dim=-1) #getting max logits for responses
        self.ori_response_ids_all = logits_max.indices#[0]
        return self.tokenizer.batch_decode(self.ori_response_ids_all)

    def generate_response_string(self, logits):
        logits_max = torch.max(logits, dim=-1) #getting max logits for responses
        self.ori_response_ids = logits_max.indices[0]
        return self.tokenizer.decode(self.ori_response_ids)

    def get_response_logits(self, inputs = None, get_response_sequences = False, generate_response = False, coerce_response = False, softmax_resp = True):

        if inputs is None:
            inputs = self.input_ids #has bos token at 0
        else:
            inputs = self.add_bos(inputs['input_ids'])
        try:
            with torch.inference_mode():
                output_generate = self.model.generate( input_ids = inputs, 
                                               min_new_tokens = self.resp_length, 
                                               max_new_tokens = self.resp_length, 
                                               do_sample = self.do_sample,
                                                      # synced_gpus=True,
                                                top_p = self.top_p,
                                                temperature = self.temperature,
                                                top_k = self.top_k,
                                               num_beams= self.num_beans,
                                               repetition_penalty = None,#self.repetition_penalty,
                                                encoder_repetition_penalty=None,#1.0,
                                                # no_repeat_ngram_size=0,
                                                # presence_penalty=0.0,       # neutral (if your HF version supports it)
                                                # frequency_penalty=0.0,      # neutral (if supported)
                                                # typical_p=1.0,              # disable typical sampling
                                                # diversity_penalty=0.0,
                                                penalty_alpha=None,         # disable contrastive search
                                                renormalize_logits=False,   # avoid extra normalization processor
                                               eos_token_id = self.eos_token_id,
                                               pad_token_id = self.pad_token_id,
                                               logits_processor = None if not coerce_response else self.get_coerced_response(inputs.shape[1]), #inputs is input_ids
                                               output_scores = self.output_scores, return_dict_in_generate = self.return_dict_in_generate )

            
            
            response_logits = torch.stack(output_generate.scores, dim=1)
            
            if not coerce_response and softmax_resp:
                assert( torch.sum(response_logits, dim=-1).sum() != response_logits.shape[1] ) 
                
                response_logits = torch.nn.functional.softmax(response_logits, dim=-1)
                
                assert( torch.sum(response_logits, dim=-1).sum().round_() == torch.tensor(response_logits.shape[1], dtype=float) ) 

            if generate_response:
                self.ori_response = self.generate_response_string(response_logits)
                self.ori_response_all = self.generate_response_string_all(response_logits)

    
            if get_response_sequences:
                return response_logits, output_generate["sequences"]

            return response_logits

        except Exception as e:
            print(f"My An unexpected error occurred: {e}")

        

    def get_input_logits(self, inputs = None):

        if inputs is None:
            inputs = self.input_ids
        else:
            inputs = self.add_bos(inputs['input_ids'])
        
        with torch.inference_mode():
            output = self.model(inputs)
            input_logits = output.logits
            input_logits = torch.nn.functional.softmax(input_logits, dim=-1)
            
        return input_logits[:,:-1,:] 
        
        
    def get_responses(self, inputs = None, get_response_sequences = False, generate_response = False, coerce_response = False, softmax_resp = True):
        
        response_logits, response_sequences = self.get_response_logits(inputs = inputs, 
                                                                       get_response_sequences = get_response_sequences, 
                                                                       generate_response=generate_response,
                                                                       coerce_response = coerce_response,
                                                                       softmax_resp = softmax_resp) # Getting Response Logits
        input_logits = self.get_input_logits(inputs = inputs) # Getting input logits

        if inputs is None:
            return input_logits, response_logits,  response_sequences
        
        elif inputs is not None and coerce_response: # we only return required logits
            #no need to shift below because the BOS token is added during logit generation
            shifted_labels = inputs['input_ids'][:, :] #input_ids are for all but input_logits are for the next
            
            assert(shifted_labels.shape[1] == input_logits.shape[1])#"Inputs logits and Input ids should have same shape"
            token_input_logits = torch.gather(input_logits, 
                            2, 
                            shifted_labels.unsqueeze(-1)  # shape (batchsize, seq_len-1, 1)
                        ).squeeze(-1)
            token_response_logits = torch.max(response_logits, dim=-1).values
            return token_input_logits, token_response_logits
                
        else:
            return input_logits, response_logits,  response_sequences

    def prefix_allowed_tokens_fn(self, batch_id, generated_ids): #outdated - does not account for BOS token addition
        """
        generated_ids: the full sequence so far (prompt + generated).
        We compute our position within the forced sequence and only allow that next id.
        """
        
        prompt_len = self.input_ids.shape[1]
        pos = generated_ids.shape[0] - prompt_len  # position in forced sequence (0-based)
        print(generated_ids.shape, self.ori_response_ids.shape, pos)
        if pos < len(self.ori_response_ids):
            
            return [self.ori_response_ids[pos].item()] # return only the desired token in our target sequence
        else:
            
            return [self.tokenizer.eos_token_id] if tok.eos_token_id is not None else list(range(self.tokenizer.vocab_size))

    def get_coerced_response(self, prompt_len):
        
        proc = ForcedSequenceProcessor(
                forced_ids = self.ori_response_ids,
                prompt_len = prompt_len,
                vocab_size = self.vocab_size,
                eos_id = self.eos_token_id,
                continue_after = False # setting this to False now
            )
        return [proc]
        
    