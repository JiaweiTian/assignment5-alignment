from vllm import LLM, SamplingParams
from datasets import load_dataset
import os
import json
import re

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

#########################################################
# Model and file path
#########################################################
model_name = "Qwen/Qwen2.5-Math-1.5B"

#########################################################
# function
#########################################################
def parse_gsm8k(answer_text):
    match = re.search(r"####\s*(.+?)(?:\s*$)", answer_text, re.MULTILINE)
    
    if match:
        final_answer = match.group(1).strip()
        reasoning = answer_text[:match.start()].strip()
    else:
        lines = answer_text.splitlines()
        final_answer = lines[-1].strip()
        reasoning = "\n".join(lines[:-1]).strip()
    
    return final_answer, reasoning
    
def load_cot_prompt(valid_data):
    prompts, responses, answers = [], [], []

    """读取思维链提示模板"""
    prompt_path = '/home/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt'
    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    for element in valid_data:
        prompts.append(prompt_template.format(question=element['question']))
 
        final_answer, reasoning = parse_gsm8k(element['answer'])
        responses.append(f"<think>{reasoning}</think> <answer>{final_answer}</answer>")
        answers.append(final_answer)

    return prompts, responses, answers

def load_dataset_from_local(file_path):
    dataset = []
    with open(file_path, "r") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

def look_up_dataset():
    # dataset = load_dataset("gsm8k", "main")

    # train_data = dataset['train']
    # validation_data = dataset['test']
    train_data = load_dataset_from_local('/home/assignment5-alignment/data/gsm8k/train.jsonl')
    validation_data = load_dataset_from_local('/home/assignment5-alignment/data/gsm8k/test.jsonl')

    print(f'len of train_data and validation_data are {len(train_data)} and {len(validation_data)}')
    print(f'Traing data sample: {train_data[0]}')
    print(f'Valid data sample: {validation_data[0]}')
    print('------------------------------------------')

    return train_data, validation_data

def calculate_average(num_prompts, response_lens, categories):
    sum_lens = 0
    average_lens = {}
    for key in categories.keys():
        if categories[key] > 0:
            sum_lens += response_lens[key]
            average_lens[key] = response_lens[key] / categories[key]
        else:
            average_lens[key] = 0

    if num_prompts > 0:
        average_lens['total'] = sum_lens / num_prompts
    else:
        average_lens['total'] = 0
        
    return average_lens

def evaluate_vllm(vllm_model, reward_fn, prompts, responses, answers, eval_sampling_params, serialize_path, save_results=False):
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    results = {}
    response_token_lens = {'format_1_answer_1': 0, 'format_1_answer_0': 0, 'format_0_answer_1': 0, 'format_0_answer_0': 0, 'total': 0}
    response_char_lens = {'format_1_answer_1': 0, 'format_1_answer_0': 0, 'format_0_answer_1': 0, 'format_0_answer_0': 0, 'total': 0}
    categories = {'format_1_answer_1': 0, 'format_1_answer_0': 0, 'format_0_answer_1': 0, 'format_0_answer_0': 0}
    categories_ratios = {'format_1_answer_1': 0, 'format_1_answer_0': 0, 'format_0_answer_1': 0, 'format_0_answer_0': 0}
    accuracy = {'total_accuracy': 0, 'format_accuracy': 0, 'answer_accuracy': 0}

    num_prompts = len(prompts)
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    idx = 0
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        reward = reward_fn(generated_text, answers[idx])
        temp = {}
        temp['example'] = prompt
        temp['generated_text'] = generated_text
        temp['gt_response'] = responses[idx]
        temp['answer'] = answers[idx]
        temp['reward'] = reward
        results[idx] = temp

        format_reward = int(reward['format_reward'])
        answer_reward = int(reward['answer_reward'])
        if format_reward == 1 and answer_reward == 1:
            categories['format_1_answer_1'] += 1
            response_token_lens['format_1_answer_1'] += len(output.outputs[0].token_ids)
            response_char_lens['format_1_answer_1'] += len(generated_text)
        elif format_reward == 0 and answer_reward == 1:
            categories['format_0_answer_1'] += 1
            response_token_lens['format_0_answer_1'] += len(output.outputs[0].token_ids)
            response_char_lens['format_0_answer_1'] += len(generated_text)
        elif format_reward == 1 and answer_reward == 0:
            categories['format_1_answer_0'] += 1
            response_token_lens['format_1_answer_0'] += len(output.outputs[0].token_ids)
            response_char_lens['format_1_answer_0'] += len(generated_text)
        elif format_reward == 0 and answer_reward == 0:
            categories['format_0_answer_0'] += 1
            response_token_lens['format_0_answer_0'] += len(output.outputs[0].token_ids)
            response_char_lens['format_0_answer_0'] += len(generated_text)

        accuracy['total_accuracy'] += reward['reward'] / num_prompts
        accuracy['format_accuracy'] += reward['format_reward'] / num_prompts
        accuracy['answer_accuracy'] += reward['answer_reward'] / num_prompts

        idx += 1

    categories_ratios['format_1_answer_1'] = categories['format_1_answer_1'] / num_prompts
    categories_ratios['format_0_answer_0'] = categories['format_0_answer_0'] / num_prompts
    categories_ratios['format_1_answer_0'] = categories['format_1_answer_0'] / num_prompts
    categories_ratios['format_0_answer_1'] = categories['format_0_answer_1'] / num_prompts

    results['eval_metrics_nums'] = categories
    results['eval_metrics_ratios'] = categories_ratios
    results['accuracy'] = accuracy
    results['response_token_lens'] = calculate_average(num_prompts, response_token_lens, categories)
    results['response_char_lens'] = calculate_average(num_prompts, response_char_lens, categories)

    if save_results:
        with open(serialize_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Save evaluate results to {serialize_path}")

    print(f'categories is {categories}')
    print(f'categories_ratios is {categories_ratios}')
    print(f'accuracy is {accuracy}')
    print(f'response_token_lens is {results["response_token_lens"]}')
    print(f'response_char_lens is {results["response_char_lens"]}')

    return results


def math_baseline():
    serialize_path = '/home/assignment5-alignment/cs336_alignment/evaluate_results_temperature_1203.json'

    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024)
    sampling_params.stop = ["</answer>"]
    sampling_params.include_stop_str_in_output = True

    llm = LLM(model=model_name)

    train_data, valid_data = look_up_dataset()

    prompts, responses, answers = load_cot_prompt(valid_data)

    _ = evaluate_vllm(llm, r1_zero_reward_fn, prompts, responses, answers, sampling_params, serialize_path, False)


if __name__ == "__main__":
    math_baseline()



# import torch
# import wandb
# from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
# from vllm import LLM, SamplingParams
# from vllm.model_executor import set_random_seed as vllm_set_random_seed
# from torch.nn.utils import clip_grad_norm_
# import json
# from pathlib import Path
# from torch.utils.data import DataLoader, Subset
# # 导入你实现的 helper 函数：tokenize_prompt_and_output, get_response_log_probs, masked_normalize, sft_microbatch_train_step, log_generations, evaluate_vllm 等

# # Starter code from document
# def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
#     vllm_set_random_seed(seed)
#     # Monkeypatch (from document)
#     from unittest.mock import patch
#     world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
#     profiling_patch = patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None)
#     with world_size_patch, profiling_patch:
#         return LLM(model=model_id, device=device, dtype=torch.bfloat16, enable_prefix_caching=True, gpu_memory_utilization=gpu_memory_utilization)

# def load_policy_into_vllm_instance(policy, llm):
#     state_dict = policy.state_dict()
#     llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
#     llm_model.load_weights(state_dict.items())

# # 数据集类
# class SFTDataset(torch.utils.data.Dataset):
#     def __init__(self, data_path):
#         self.data = []
#         with open(data_path, 'r') as f:
#             for line in f:
#                 self.data.append(json.loads(line))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]['prompt'], self.data[idx]['response']

# # 过滤正确样本 (for part 2)
# def filter_correct_examples(dataset, reward_fn):
#     filtered = []
#     for prompt, response in dataset:
#         eval_result = reward_fn(prompt + response, ground_truth)  # 假设有 ground_truth，或从数据提取
#         if eval_result['answer_reward'] == 1:
#             filtered.append((prompt, response))
#     return filtered

# # 主训练函数
# def run_sft(dataset_size, filtered=False, lr=1e-5, batch_size=8, epochs=3, gradient_accumulation_steps=4, seed=42):
#     wandb.init(project="cs336_sft")
#     wandb.define_metric("train_step")
#     wandb.define_metric("eval_step")
#     wandb.define_metric("train/*", step_metric="train_step")
#     wandb.define_metric("eval/*", step_metric="eval_step")

#     # 加载模型和 tokenizer
#     model_path = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
#     policy = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to('cuda:0')
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     optimizer = AdamW(policy.parameters(), lr=lr)

#     # 初始化 vLLM (on cuda:1)
#     vllm = init_vllm(model_path, device='cuda:1', seed=seed)

#     # 加载数据
#     full_dataset = SFTDataset("/data/a5-alignment/MATH/sft.jsonl")
#     if filtered:
#         dataset = filter_correct_examples(full_dataset, reward_fn=cs336_alignment.drgrpo_grader.r1_zero_reward_fn)  # 使用文档中的 reward_fn
#         print(f"Filtered dataset size: {len(dataset)}")
#     else:
#         indices = list(range(min(dataset_size, len(full_dataset))))
#         dataset = Subset(full_dataset, indices)
    
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     train_step = 0
#     for epoch in range(epochs):
#         policy.train()
#         for batch_idx, (prompts, responses) in enumerate(dataloader):
#             # Tokenize
#             tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
#             input_ids = tokenized['input_ids'].to('cuda:0')
#             labels = tokenized['labels'].to('cuda:0')
#             response_mask = tokenized['response_mask'].to('cuda:0')

#             # 前向 + log probs
#             log_probs_dict = get_response_log_probs(policy, input_ids, labels, return_token_entropy=True)
#             policy_log_probs = log_probs_dict['log_probs']
#             token_entropy = log_probs_dict['token_entropy']

#             # Microbatch train step
#             loss, metadata = sft_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps)
#             wandb.log({"train/loss": loss.item(), "train/entropy": token_entropy.mean().item()}, step=train_step)
            
#             if (batch_idx + 1) % gradient_accumulation_steps == 0:
#                 clip_grad_norm_(policy.parameters(), 1.0)  # 梯度裁剪
#                 optimizer.step()
#                 optimizer.zero_grad()
#                 train_step += 1

#         # 定期评估
#         if epoch % 1 == 0:  # 每 epoch 评估一次
#             load_policy_into_vllm_instance(policy, vllm)
#             eval_metrics = evaluate_vllm(vllm, reward_fn=cs336_alignment.drgrpo_grader.r1_zero_reward_fn, prompts=val_prompts, eval_sampling_params=SamplingParams(...))
#             wandb.log({"eval/accuracy": eval_metrics['accuracy']}, step=eval_step)
#             log_generations(...)  # 日志生成样本
#             eval_step += 1

#     # 保存模型
#     policy.save_pretrained(output_dir)
#     tokenizer.save_pretrained(output_dir)

# # 主入口
# if __name__ == "__main__":
#     # 对于部分 1：循环不同 dataset_size
#     for size in [128, 256, 512, 1024, 'full']:
#         run_sft(dataset_size=size if size != 'full' else None, filtered=False)
    
#     # 对于部分 2：过滤模式
#     run_sft(dataset_size=None, filtered=True)
