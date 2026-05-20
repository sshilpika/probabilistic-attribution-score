
from log_attr_scores_nucleus import ProbabilisticAttributionScores
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Model, GPT2Tokenizer, GPTNeoForCausalLM
import time
from accelerate import PartialState
from accelerate.utils import gather_object, gather, pad_across_processes
import torch.distributed as dist
import os
import json
from collections import Counter
from scipy.stats import entropy
import matplotlib.pyplot as plt
import random
import numpy as np

random.seed(0)  # Sets the seed for Python's built-in random module
np.random.seed(0)  # Sets the seed for NumPy's random number generator

torch.manual_seed(0)  # Sets the seed for PyTorch's CPU random number generator
torch.cuda.manual_seed(0)  # Sets the seed for the current GPU device
torch.cuda.manual_seed_all(0)  # Sets the seed for all available GPU devices

torch.backends.cuda.matmul.allow_tf32 = False  # Disables TensorFloat32 (TF32) on matmul ops
torch.backends.cudnn.allow_tf32 = False  # Disables TF32 on cuDNN
torch.backends.cudnn.benchmark = False  # Disables the cuDNN auto-tuner
torch.backends.cudnn.deterministic = True  # Forces cuDNN to use deterministic algorithms
torch.backends.cudnn.enabled = False 

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# Determinism/precision knobs
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # for CUDA matmul determinism
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("high")  # avoids TF32 on some setups

if __name__ == "__main__":

    distributed_state = PartialState()
    pad_to_multiple_of = 8

    prompts = ["""Context: "Tim was nervous about his talk. He had been preparing and practicing for weeks, but he still wasn't sure he was fully ready." Sentence: "As Tim stood in front of the audience, he took deep breaths and tried to remain _____.""",
               """The surfer woke up early to go surfing. He checked the weather forecast to see if it would be windy, since the previous days the weather had been calm. He quickly grabbed his equipment and headed to the beach, hoping the weather would""",
               """Dr. Chang was excited about her hike to the mountains. She packed her tent, hiking boots, a warm jacket, and food for the journey. When she reached the mountain top, the view was breathtaking, and Chang felt a sense of""",
               """The police officer walked into the room, carefully examining every clue. The information was sparse, but he believed the clues would lead him to the truth. He looked at the doorknob, noticing something odd about it. He made a note to check it more""",
               """Passage: "In 1997, NASA's Cassini space probe was launched to explore Saturn. It was active for twenty years, spending it's last 13 years orbiting Saturn." Question: "When was the Cassini launched?""",
               """Sentence: "The water bottle didn't fit into the bag because it was too big." Question: "What is 'it' referring to?""",
               """What year did Albert Einstein visit Mars?"""
              ]
    try:
        for s_id, prompt in enumerate(prompts):

            
            model_name = "Qwen/Qwen3-1.7B"
            #"Qwen/Qwen2.5-1.5B-Instruct" 
            #"EleutherAI/gpt-neo-1.3B"
            #"openai-community/gpt2"
            #"allenai/OLMo-2-0425-1B-Instruct"
            #"meta-llama/Llama-3.2-1B-Instruct"
            #"allenai/OLMo-2-0425-1B"
            device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu") 
            
            
            if "gpt-neo" in model_name:
                tokenizer = GPT2Tokenizer.from_pretrained(model_name)
                model = GPTNeoForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.float32, attn_implementation="eager")
                model.to(device)
            
            elif "allenai" in model_name:
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                model     = AutoModelForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.float32, attn_implementation="eager")
                model.to(device)
            elif "gemma" in model_name:
                tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ['HUGGINGFACEHUB_API_TOKEN'])
                model = AutoModelForCausalLM.from_pretrained(model_name, token=os.environ['HUGGINGFACEHUB_API_TOKEN'],
                                                 torch_dtype=torch.float32, attn_implementation="eager").eval()
                model.to(device)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ['HUGGINGFACEHUB_API_TOKEN'])
                model = AutoModelForCausalLM.from_pretrained(model_name, token=os.environ['HUGGINGFACEHUB_API_TOKEN'],
                                                 torch_dtype=torch.float32, attn_implementation="eager").eval()
                model.to(device)
            
            model_save = model_name.split("/")[-1]
            
            
            tokenizer.pad_token = tokenizer.eos_token  # Use <eos> token as padding
            tokenizer.padding_side = "left"  # Ensure left-padding
        
            #--------------------------------
            # Get Scores for Original Input
            #---------------------------------

            Anum_path = "../results/one_shot_nucleus/TOPP_A_NUM/"
            resp_length = 20

            with open(f"{Anum_path}A_NUM_{model_save}_{s_id}.csv", 'r') as f:
                lines_anum_pre = f.readlines()

                l_ans_fin = []
                l_ans = []
                for lan_id, lan in enumerate(lines_anum_pre):
                    if model_save in lan:
                        l_ans_fin.append("".join(l_ans))
                        l_ans = []
                    l_ans.append(lan)
                    if lan_id == len(lines_anum_pre)-1:
                        l_ans_fin.append("".join(l_ans))
                
                
                lines_anum_ = [[j for j in i.split("|@|")] for i in l_ans_fin]
                lines_anum = lines_anum_[3]
            assert(lines_anum[2] == "0.4")
            
            pa = ProbabilisticAttributionScores(model, tokenizer, prompt, device=device, resp_length= resp_length if not "gpt" in model_name and not "Qwen" in model_name else len(tokenizer.encode(lines_anum[3])))
            
            il, _, _ = pa.get_responses(generate_response=True, get_response_sequences=True)


            if "allenai" in model_name:
                lines_anum[3] = tokenizer.decode(tokenizer.encode(lines_anum[3])[:resp_length])

            
            pa.ori_response = lines_anum[3]
            pa.ori_response_all = [lines_anum[3]]

            if "llama" in model_name or "facebook" in model_name or "allenai_JJ" in model_name:
                pa.ori_response_ids_all = [tokenizer.encode(lines_anum[3])[1:]]
                pa.ori_response_ids = tokenizer.encode(lines_anum[3])[1:]
            else:
                pa.ori_response_ids_all = [tokenizer.encode(lines_anum[3])]
                pa.ori_response_ids = tokenizer.encode(lines_anum[3])
          
            
            response = lines_anum[3]
            assert( response == tokenizer.decode(pa.ori_response_ids))
            A_num = torch.tensor(float(lines_anum[-1].split("\n")[0]))
                    
            #--------------------------------
            # Get Scores for P' Input
            #---------------------------------

            file_path = f"vocab_{model_save}.json"
            vocab_dict = {}
            if os.path.exists(file_path):
                with open(file_path, 'r') as fvcb:
                    print("reading vocab from saved file")
                    vocab_dict = json.load(fvcb)
            else:
                vocab_dict = tokenizer.get_vocab()
    
                try:
                    with open(file_path, "w") as json_file:
                        json.dump(vocab_dict, json_file, indent=4)
                    print(f"Dictionary successfully saved to {file_path}")
                except IOError as e:
                    print(f"Error writing to file: {e}")
                
            replacement_token_ids = list(vocab_dict.values())
            
            first_mu = 0
            step = 2
            if "allenai" in model_name:
                step = 200
            elif "google/gemma-3-270m" in model_name:
                step = 185
            elif "gpt2" in model_name:
                step = 150
            elif "gpt-neo" in model_name:
                step = 150
            elif "Qwen" in model_name:
                step = 150
            elif "meta-llama/Llama-3.2-1B-Instruct" in model_name:
                step = 200
            elif "facebook/MobileLLM-R1-950M" in model_name:
                step = 150
                
            tensor_path = "../results/one_shot_nucleus/torch_tensors/"
            input_token_len = il.shape[1]
        
            #file save 
            if distributed_state.is_main_process:
                    
                st = time.time()
                with open(f"../results/one_shot_nucleus/ALLENAI_{model_save}_{s_id}.csv", 'a') as f:
                    f.write("model_name|prompt|response|muid|mu_val|D|D_num|D_denom|A_num|LogA|time\n")
                
                try:
                    os.makedirs(os.path.join(tensor_path,model_save), exist_ok=True)
                    print(f"Directory '{tensor_path}' created or already exists.")
                except OSError as e:
                    print(f"Error creating directory '{tensor_path}': {e}")
                        
            for mu in range(first_mu, input_token_len):#input_token_len
                
                if not (dist.is_available() and dist.is_initialized()):
                    distributed_state = PartialState()
                    pad_to_multiple_of = 8
                
                curr_inp_tkn_id = pa.input_ids[:, mu+1] # this contains BOS as first token so we skip it
            
                #-------------------
                # Generate Batches
                #-------------------
                sentences = [prompt] * len(replacement_token_ids)
                padding_side_default = tokenizer.padding_side
                tokenizer.padding_side = "left"
                inputs_batch = [
                    tokenizer(sentences[i:i+step], padding=True, return_tensors="pt").to(device)
                    for i in range(0, len(sentences), step)
                ] 
                tokenizer.padding_side = padding_side_default
                mu_val = tokenizer.decode(curr_inp_tkn_id, skip_special_tokens=False)
                for ii, r_id in enumerate(range(0, len(replacement_token_ids), step)):
                    inputs_batch[ii]["input_ids"][:, mu] = torch.tensor(replacement_token_ids[r_id:r_id+step])
            
                #--------------------
                # Generate Responses
                #--------------------
                from_mu_logits_all = []
                from_mu_input_logits_all = []
                replacement_tokens_entropy_list = []
                count_ori_response = []
                
            
                with distributed_state.split_between_processes(inputs_batch, apply_padding=False) as batched_prompts:
        
                    for batch in batched_prompts:
                        # Move the batch to the device
                        batch = batch.to(distributed_state.device) 
        
                        il_dash, rl_dash = pa.get_responses(batch, generate_response=True, get_response_sequences=True, coerce_response = True)
                        
                       
                        assert( response == pa.ori_response )
                        
                        from_mu_logits = torch.cat((il_dash, rl_dash), dim=1)[:, mu:] 
                        from_mu_input_logits = il_dash[:, mu:]

    
                        from_mu_logits_logSUM = torch.log( from_mu_logits + 1e-40 ).sum(dim=1)
                        from_mu_input_logits_logSUM = torch.log( from_mu_input_logits + 1e-40 ).sum(dim=1)
    
                        from_mu_logits_all.append(from_mu_logits_logSUM)
                        from_mu_input_logits_all.append(from_mu_input_logits_logSUM)
                        
                        repl_ids_inp = batch['input_ids']
                        
                        
                        if torch.is_tensor(repl_ids_inp):
                            repl_ids_inp = repl_ids_inp[:, mu]
                        else:
                            repl_ids_inp = torch.Tensor(repl_ids_inp[:, mu])
                       
                        replacement_tokens_entropy_list.append( repl_ids_inp )
                
                
    
                from_mu_logits_all_local = torch.cat(from_mu_logits_all, dim=0)
                from_mu_input_logits_all_local = torch.cat(from_mu_input_logits_all, dim=0)
                replacement_tokens_entropy_list_local = torch.cat(replacement_tokens_entropy_list, dim=0 )
                
                # If shapes could differ, pad then gather; otherwise skip padding.
                from_mu_logits_all_local = pad_across_processes(from_mu_logits_all_local, dim = 0, pad_index = -1000 )
                from_mu_input_logits_all_local = pad_across_processes(from_mu_input_logits_all_local, dim = 0, pad_index = -1000 )
                replacement_tokens_entropy_list_local = pad_across_processes(replacement_tokens_entropy_list_local, dim=0, pad_index = -1000)
        
                # gather
                from_mu_logits_all_global = gather(from_mu_logits_all_local)
                from_mu_input_logits_all_global = gather(from_mu_input_logits_all_local)
                replacement_tokens_entropy_list_global = gather(replacement_tokens_entropy_list_local)
                        
                if distributed_state.is_main_process:
    
                    muLF = from_mu_logits_all_global.cpu()
                    muILF = from_mu_input_logits_all_global.cpu()
                    replacement_tokens_entropy = replacement_tokens_entropy_list_global.cpu()
    
                    muLF = torch.masked_select(muLF, muLF != -1000)
                    muILF = torch.masked_select(muILF, muILF != -1000)

                    replacement_tokens_entropy = torch.masked_select(replacement_tokens_entropy, replacement_tokens_entropy != -1000 )
                    
                    D_num = torch.logsumexp(muLF, dim = 0 )
                    D_denom = torch.logsumexp(muILF, dim = 0 )
    
                    D = D_num - D_denom
                    assert( D <= 1 )
    
                    LogA = A_num - ( D )
    
                    torch.save(muLF, os.path.join(tensor_path, model_save, f"ALLENAI_MULF_{model_save}_MU{mu}_SID{s_id}.pt"))
                    torch.save(muILF, os.path.join(tensor_path, model_save, f"ALLENAI_MUILF_{model_save}_MU{mu}_SID{s_id}.pt"))
                    torch.save(replacement_tokens_entropy, os.path.join(tensor_path, model_save, f"ALLENAI_REPL_TOKEN_ENT_{model_save}_MU{mu}_SID{s_id}.pt"))

                    ed = time.time()-st
                    print(f"Per process took {ed} seconds")
                    
                    
                    print(f"from_mu_logits_all_ ---> {muLF.shape} and from_mu_input_logits_all_ ---> {muILF.shape} -->> LogA={LogA} A_num={A_num} == {torch.log(A_num)} D_num={D_num} D_denom{D_denom} D={D}")
                    with open(f"../results/one_shot_nucleus/ALLENAI_{model_save}_{s_id}.csv", 'a') as f:
                        resp_clean = response.replace("\n", "NEWLINE")
                        f.write(f"{model_name}|{prompt}|{resp_clean}|{mu}|{mu_val}|{D.cpu().float().numpy()}|{D_num.cpu().float().numpy()}|{D_denom.cpu().float().numpy()}|{A_num.cpu().float().numpy()}|{LogA.cpu().float().numpy()}|{ed}\n")
                    
                    st = time.time()
                        
                
                if dist.is_available() and dist.is_initialized():
                    distributed_state.wait_for_everyone()
                    dist.barrier(device_ids=[distributed_state.local_process_index] if dist.get_backend() == "nccl" else None)
            
                # (optional) sync CUDA before teardown
                if torch.cuda.is_available():
                    torch.cuda.synchronize(distributed_state.device)
                                            
    finally:
        # destroy the process group so NCCL can shut down cleanly
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
            torch.cuda.empty_cache()
                        
                
