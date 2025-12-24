import os
import numpy as np
import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup, PreTrainedModel
from datetime import datetime
from typing import Literal
from unittest.mock import patch
from pathlib import Path
import random
import argparse
import json
from tqdm import tqdm

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.zero_shot_baseline import (load_cot_prompt,
                                                look_up_dataset,
                                                evaluate_vllm
                                                )
from cs336_alignment.expert_iteration import tokenize_prompt_and_output, get_response_log_probs, masked_normalize

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def compute_group_normalized_rewards(reward_fn, rollout_responses, repeated_ground_truths, group_size, advantage_eps, normalize_by_std):
    '''
        Input:
            rollout_responses: List[str], len = N
            repeated_ground_truths: List[str], len = N
            group_size: int
            normalize_by_std: bool
        Output:
            advantages: List[float], len = N
            raw_rewards: List[float], len = N
            metadata: dict
    '''
    
    reward_list = []
    for i in range(len(rollout_responses)):

        reward = reward_fn(rollout_responses[i], repeated_ground_truths[i])

        reward_list.append(reward['reward'])

    reward_array = np.array(reward_list, dtype=np.float32).reshape(-1, group_size)
    reward_mean = np.mean(reward_array, axis=1, keepdims=True)
    reward_std = np.std(reward_array, axis=1, ddof=1, keepdims=True)

    if normalize_by_std:
        advantages = (reward_array - reward_mean) / (reward_std + advantage_eps)
    else:
        advantages = reward_array - reward_mean

    metadata = {}

    return advantages.flatten().tolist(), reward_array.flatten().tolist(), metadata


def compute_naive_policy_gradient_loss(raw_rewards_or_advantages: torch.Tensor, policy_log_probs: torch.Tensor) -> torch.Tensor:
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(advantages: torch.Tensor, policy_log_probs: torch.Tensor, old_log_probs: torch.Tensor, cliprange: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    log_probs_ratios = policy_log_probs - old_log_probs
    probs_ratios = torch.exp(log_probs_ratios)

    num_prompts, seq_length = policy_log_probs.shape

    clip_log_probs = torch.empty(num_prompts, seq_length)
    for i in range(num_prompts):
        advantage = advantages[i, 0]
        for j in range(seq_length):
            if advantage >= 0:
                clip_log_probs[i][j] = min(probs_ratios[i][j], 1+cliprange)
            else:
                clip_log_probs[i][j] = max(probs_ratios[i][j], 1-cliprange)

    grpo_clip_loss = compute_naive_policy_gradient_loss(advantages, clip_log_probs)

    metadata = {}
    clip_or_not = (torch.abs(probs_ratios - 1.0) > cliprange).float()
    clip_fraction = clip_or_not.mean(dim=-1)
    metadata['clip_or_not'] = clip_or_not
    metadata['clip_fraction'] = clip_fraction

    return grpo_clip_loss, metadata

    ##################################################
    # torch.clamp写法
    ##################################################
    '''
    num_prompts, seq_length = policy_log_probs.shape
    
    # 计算概率比例（在log space中做减法）
    log_prob_ratio = policy_log_probs - old_log_probs
    
    # 转换为概率比例
    ratios = torch.exp(log_prob_ratio)
    
    # 扩展优势值到每个时间步
    advantages_expanded = advantages.expand(-1, seq_length)
    
    # PPO clip损失的核心计算
    # 公式: L = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
    surr1 = ratios * advantages_expanded
    surr2 = torch.clamp(ratios, 1.0 - cliprange, 1.0 + cliprange) * advantages_expanded
    
    # 取最小值（更保守的估计）
    loss = -torch.min(surr1, surr2)
    
    # 计算平均损失
    grpo_clip_loss = loss.mean()
    
    # 收集统计信息
    metadata = {
        'clip_fraction': (torch.abs(ratios - 1.0) > cliprange).float().mean(),
        'avg_ratio': ratios.mean(),
        'min_ratio': ratios.min(),
        'max_ratio': ratios.max(),
        'avg_advantage': advantages.mean(),
        'advantages_expanded': advantages_expanded,
        'ratios': ratios,
    }
    
    return loss, metadata
    '''
    
def compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange):
    metadata = {}

    if loss_type == 'no_baseline':
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
    elif loss_type == 'reinforce_with_baseline':
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
    elif loss_type == 'grpo_clip':
        loss, metadata = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)

    return loss, metadata

    
def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    masked = tensor * mask
    return masked.sum(dim=dim) / mask.sum(dim=dim) # 只除以mask=1的个数， 而不是整个seq_len


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size, seq_len = policy_log_probs.shape

    loss, metadata = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    loss = masked_mean(loss, response_mask, None)
    loss /= gradient_accumulation_steps

    loss.backward()

    return loss, metadata


####################################################
# Training codes
###################################################
def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.5):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    13
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )

    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
            enforce_eager=True
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def load_model_and_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


class SFTDataset(Dataset):
    def __init__(self, inputs_data, tokenizer):
        self.num_samples = len(inputs_data)

        prompts = [inputs_data[idx]['prompt'] for idx in range(self.num_samples)]
        responses = [inputs_data[idx]['response'] for idx in range(self.num_samples)]

        self.dictionary = tokenize_prompt_and_output(prompts, responses, tokenizer)
        # 新增：保存原始索引
        self.indices = list(range(self.num_samples))

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        input_ids = self.dictionary['input_ids'][idx, :]
        labels = self.dictionary['labels'][idx, :]
        response_mask = self.dictionary['response_mask'][idx, :]

        return input_ids, labels, response_mask, self.indices[idx]

def generate_and_filter_samples(vllm_model, sampling_params, sampling_data, group_size, advantage_eps, normalize_by_std):
    prompts, _, answers = load_cot_prompt(sampling_data)

    generation_outputs = vllm_model.generate(prompts, sampling_params)

    samples, rollout_responses, repeated_ground_truths = [], [], []
    for i, output in enumerate(generation_outputs):
        for num in range(len(output.outputs)):
            rollout_responses.append(output.outputs[num].text)
            repeated_ground_truths.append(answers[i])
            samples.append({'prompt': output.prompt, 'response': output.outputs[num].text})

    advantages, raw_rewards, metadata = compute_group_normalized_rewards(r1_zero_reward_fn, rollout_responses, repeated_ground_truths,
                                                                            group_size, advantage_eps, normalize_by_std)

    return samples, advantages, raw_rewards, metadata

def main(args):
    #############################################
    # assertions and derived parameters
    #############################################
    assert args.train_batch_size % args.gradient_accumulation_steps == 0, ("train_batch_size must be divisible by gradient_accumulation_steps")
    assert args.rollout_batch_size % args.group_size == 0, ("rollout_batch_size must be divisible by group_size")
    assert args.train_batch_size >= args.group_size, ("train_batch_size must be greater than or equal to group_size")
    micro_train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    n_prompts_per_rollout_batch = args.rollout_batch_size // args.group_size
    n_microbatches_per_rollout_batch = args.rollout_batch_size // micro_train_batch_size

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    vllm_device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-Math-1.5B"

    wandb.init(project="gsm8k-grpo-experiment", name=args.run_name)
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    ##################################################
    # get the model and tokenizer
    ##################################################
    model, tokenizer = load_model_and_tokenizer(model_name)
    model.gradient_checkpointing_enable()
    model.to(device)

    #################################################
    # initilize vllm model for inference
    #################################################
    SEED = 42
    sampling_params = SamplingParams(temperature=args.sampling_temperature, top_p=1.0, max_tokens=args.sampling_max_tokens, min_tokens=args.sampling_min_tokens, n=args.group_size, stop=["</answer>"], include_stop_str_in_output=True)
    valid_sampling_params = SamplingParams(temperature=args.sampling_temperature, top_p=1.0, max_tokens=args.sampling_max_tokens, stop=["</answer>"], include_stop_str_in_output=True)
    vllm_model = init_vllm(model_name, vllm_device, SEED, gpu_memory_utilization=args.gpu_memory_utilization)

    #################################################
    # optimizer
    #################################################
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95)
    )

    ##################################################
    # get the training and validation data
    ##################################################
    training_data, valid_data = look_up_dataset()
    valid_prompts, valid_responses, valid_answers = load_cot_prompt(valid_data)

    tokens_seen = 0
    train_step = 0

    model.train()
    for step in range(args.n_grpo_steps):
        sampling_data = random.sample(training_data, k = n_prompts_per_rollout_batch)

        with torch.no_grad():
            load_policy_into_vllm_instance(model, vllm_model)
            reasoning_traces, advantages, raw_rewards, metadata = generate_and_filter_samples(vllm_model, sampling_params, sampling_data,
                                                                            args.group_size, args.advantage_eps, args.use_std_normalization)

        train_dataset = SFTDataset(reasoning_traces, tokenizer)

        dataloader = DataLoader(
            train_dataset,
            batch_size=micro_train_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        #################################################
        # training loop
        #################################################
        for epoch in range(args.epochs_per_rollout_batch):
            # record information
            total_loss = 0
            total_token_entropy = 0.0

            # Add tqdm progress bar here
            for idx, (input_ids, labels, response_mask, indices) in enumerate(
                tqdm(dataloader, desc=f"Step: {step}, Epoch: {epoch} Training", leave=False)
            ):
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                response_mask = response_mask.to(device)

                # feed_forward + loss calculation
                log_probs_dict = get_response_log_probs(model, input_ids, labels, True)
                policy_log_probs = log_probs_dict['log_probs']
                token_entropy = log_probs_dict['token_entropy']
                mean_token_entropy = torch.mean(token_entropy).item()

                # minibatch train step
                indices = indices.tolist()
                if args.loss_type == 'no_baseline':
                    raw_rewards_batch = torch.tensor([raw_rewards[i] for i in indices], dtype=torch.float32, device=device).unsqueeze(-1)
                    loss, _ = grpo_microbatch_train_step(policy_log_probs, response_mask, args.gradient_accumulation_steps, args.loss_type, raw_rewards=raw_rewards_batch)
                elif args.loss_type == 'reinforce_with_baseline':
                    advantages_batch = torch.tensor([advantages[i] for i in indices], dtype=torch.float32, device=device).unsqueeze(-1)
                    loss, _ = grpo_microbatch_train_step(policy_log_probs, response_mask, args.gradient_accumulation_steps, args.loss_type, advantages=advantages_batch)    
                elif args.loss_type == 'grpo_clip':
                    pass

                total_loss += loss.item() * args.gradient_accumulation_steps
                total_token_entropy += mean_token_entropy

                train_step += 1
                tokens_seen += torch.sum(response_mask).item()

                if (idx + 1) % (args.gradient_accumulation_steps) == 0:
                    # Gradient clipping
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    # Update weights every `gradient_accumulation_steps` batches.
                    optimizer.step()
                    # Zero gradients every `gradient_accumulation_steps` batches.
                    optimizer.zero_grad()

            avg_loss = total_loss / len(dataloader)
            avg_token_entropy = total_token_entropy / len(dataloader)
            wandb.log({"train/training_avg_loss": avg_loss, "train/training_avg_token_entropy": avg_token_entropy, "train/grad_norm": total_grad_norm, "train_step": train_step})
            print(f"grpo training step: {step}, epoch: {epoch}, total micro-batch steps: {train_step}, training loss: {avg_loss:.3f}")

        #################################################
        # evaluation process
        #################################################
        print("evaluation:")
        if step % args.eval_every_n_steps == 0 or step == args.n_grpo_steps - 1:
            with torch.no_grad():
                load_policy_into_vllm_instance(model, vllm_model)

                path = Path(args.output_dir)
                path.mkdir(parents=True, exist_ok=True)
                serialize_path = Path(args.output_dir) / f"evaluate_results_grpo_{step}.json"
                results = evaluate_vllm(vllm_model, r1_zero_reward_fn, valid_prompts, valid_responses, valid_answers, valid_sampling_params, serialize_path, True)

                print('###########################################################')
                print(f'grpo evaluation step: {step}, total micro-batch steps: {train_step}, Validation results')
                print(f"Number of different format and answer rewards is {results['eval_metrics_nums']}")
                print(f"Ratio of different format and answer rewards is {results['eval_metrics_ratios']}")
                print(f"Accuracy metrics is {results['accuracy']}")

                wandb.log({
                    'eval/total_seen_tokens': tokens_seen,
                    'eval/rewards': results['eval_metrics_nums'],
                    'eval/reward_ratios': results['eval_metrics_ratios'],
                    'eval/accuracy': results['accuracy'],
                    'eval_step': step
                })

                if step == args.n_grpo_steps - 1:
                    save_dir = Path(args.output_dir) / f"checkpoint_last"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
                    print(f"Saved checkpoint to {save_dir}")

    print('Finished')
    wandb.finish()

    # Cleanup vLLM resources
    try:
        del vllm_model
        torch.cuda.empty_cache()
        print("Cleaned up vLLM resources")
    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")


def get_training_args():
    parser = argparse.ArgumentParser(description='training codes for group relative policy optimization')

    parser.add_argument("--n_grpo_steps", type=int, default=200, help="Number of GRPO training steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--advantage_eps", type=float, default=1e-6, help="Epsilon added to std when normalizing advantages")
    parser.add_argument("--rollout_batch_size", type=int, default=256, help="Number of rollouts collected per batch")
    parser.add_argument("--group_size", type=int, default=8, help="Size of each group for group-relative normalization")
    parser.add_argument("--sampling_temperature", type=float, default=1.0, help="Sampling temperature used during generation")
    parser.add_argument("--sampling_min_tokens", type=int, default=4, help="Minimum number of tokens to sample (prevents empty responses)")
    parser.add_argument("--sampling_max_tokens", type=int, default=1024, help="Maximum number of tokens to sample")
    parser.add_argument("--epochs_per_rollout_batch", type=int, default=1, help="Number of training epochs per collected rollout batch (on-policy = 1)")
    parser.add_argument("--train_batch_size", type=int, default=256, help="Training batch size (on-policy)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=128, help="Number of gradient accumulation steps (microbatching)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6, help="Target GPU memory utilization for vLLM")
    parser.add_argument("--loss_type",
            type=str,
            choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"],
            default="reinforce_with_baseline",
            help="Type of policy gradient loss to use"
    )
    # Requires Python 3.9+ for BooleanOptionalAction; fallback to store_true/store_false can be used otherwise.
    parser.add_argument("--use_std_normalization", action=argparse.BooleanOptionalAction, default=True, help="Whether to normalize advantages by their standard deviation")
    parser.add_argument("--cliprange", type=float, default=0.2, help="Clipping range for GRPO-clip loss")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/outputs/2025_1224_grpo_no_baseline_lr_1e-4")
    parser.add_argument("--run_name", type=str, default=(lambda: datetime.now().strftime("%Y-%m%d-%H%M%S"))(), help="Experiment name, default is current timestamp")
    parser.add_argument("--eval_every_n_steps", type=int, default=10, help="Evaluate every N steps")

    return parser.parse_args()


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    
    # 确保确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    args = get_training_args()

    set_seed(42)

    main(args)