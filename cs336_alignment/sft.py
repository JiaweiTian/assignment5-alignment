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
from unittest.mock import patch
from pathlib import Path
import random
import argparse

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.zero_shot_baseline import (load_cot_prompt,
                                                look_up_dataset,
                                                evaluate_vllm
                                                )

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    # Ensure tokenizer has a pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    all_sequences = []
    mask_sequences = []
    
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)['input_ids']
        output_tokens = tokenizer(output, add_special_tokens=False)['input_ids']
        
        full_sequence = prompt_tokens + output_tokens
        all_sequences.append(full_sequence)
        
        mask = [0] * len(prompt_tokens) + [1] * len(output_tokens)
        mask_sequences.append(mask)

    max_len = max(len(seq) for seq in all_sequences)

    padded_sequences = []
    padded_masks = []

    for seq, mask in zip(all_sequences, mask_sequences):
        padded_seq = seq + [tokenizer.pad_token_id] * (max_len - len(seq))
        padded_sequences.append(padded_seq)
        
        padded_mask = mask + [0] * (max_len - len(mask))
        padded_masks.append(padded_mask)
    
    input_ids = torch.tensor(padded_sequences, dtype=torch.long)
    response_mask = torch.tensor(padded_masks, dtype=torch.bool)
        
    return {
        'input_ids': input_ids[:, :-1],
        'labels': input_ids[:, 1:], 
        'response_mask': response_mask[:, 1:]
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    x = logits
    dim = -1
    """数值稳定的log_softmax实现"""
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_stable = x - x_max  # 减去最大值，避免指数溢出
    torch_sum = torch.sum(torch.exp(x_stable), dim=dim, keepdim=True)
    log_sum_exp = torch.log(torch_sum)
    log_P_of_x = x_stable - log_sum_exp
    # 方法1: 直接计算
    # P_of_x = (torch.exp(x_stable) / torch_sum)
    # 方法2: 因为已经知道logP(x)的值，所以只要做一次指数运算，就可以得到P(x)的值
    P_of_x = torch.exp(log_P_of_x)

    result = P_of_x * log_P_of_x
    result = -torch.sum(result, dim=dim)

    return result


def get_response_log_probs(model, input_ids, labels, return_token_entropy=False):
    logits = model(input_ids).logits

    batch_size, seq_len, vocab_size = logits.shape

    logits_reshaped = logits.view(-1, vocab_size)
    labels_reshaped = labels.view(-1)

    log_probs = -F.cross_entropy(logits_reshaped, labels_reshaped, reduction='none')
    log_probs = log_probs.view(batch_size, seq_len)

    results = {}
    results['log_probs'] = log_probs

    if return_token_entropy:
        entropy = compute_entropy(logits)
        results['token_entropy'] = entropy

    return results


def masked_normalize(tensor, mask, normalize_constant, dim):

    masked_sum = torch.sum(tensor * mask, dim=dim)
 
    return masked_sum / normalize_constant


def sft_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps, normalize_constant):
    batch_size, seq_len = policy_log_probs.shape

    loss = masked_normalize(policy_log_probs, response_mask, normalize_constant, None)
    loss /= batch_size
    loss = -loss / gradient_accumulation_steps
    print(f'shape of loss is {loss.shape}')

    loss.backward()

    return loss, None


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
            # device=device,
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
        attn_implementation="flash_attention_2",
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


class SFTDataset(Dataset):
    def __init__(self, inputs_data, num_samples, tokenizer):
        self.num_samples = num_samples if num_samples else len(inputs_data)

        random_idx = random.sample(range(len(inputs_data)), self.num_samples)

        subset_data = [inputs_data[i] for i in random_idx]
        prompts, responses, _ = load_cot_prompt(subset_data)

        self.dictionary = tokenize_prompt_and_output(prompts, responses, tokenizer)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        input_ids = self.dictionary['input_ids'][idx, :]
        labels = self.dictionary['labels'][idx, :]
        response_mask = self.dictionary['response_mask'][idx, :]
        
        return input_ids, labels, response_mask


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Qwen/Qwen2.5-Math-1.5B"

    wandb.init(project="gsm8k-sft-experiment", name=args.run_name)
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")


    ##################################################
    # get the model and tokenizer
    ##################################################
    model, tokenizer = load_model_and_tokenizer(model_name)
    model.to('cuda:0')

    ##################################################
    # get the training and validation data
    ##################################################
    training_data, valid_data = look_up_dataset()

    train_dataset = SFTDataset(training_data, args.num_samples, tokenizer)

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    valid_prompts, valid_responses, valid_answers = load_cot_prompt(valid_data)
    
    #################################################
    # optimizer
    #################################################
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.999), 
        eps=1e-8
    )

    num_training_steps = len(dataloader) * args.num_epochs // args.gradient_accumulation_steps
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    #################################################
    # initilize vllm model for inference
    #################################################
    SEED = 42
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True)
    vllm_model = init_vllm(model_name, device, SEED, gpu_memory_utilization=0.3)

    #################################################
    # training loop
    #################################################
    model.train()
    tokens_seen = 0

    train_step = 0
    for epoch in range(args.num_epochs):
        # record information
        total_loss = 0

        for idx, (input_ids, labels, response_mask) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            response_mask = response_mask.to(device)

            # feed_forward + loss calculation
            log_probs_dict = get_response_log_probs(model, input_ids, labels, True)
            log_probs = log_probs_dict['log_probs']
            # token_entropy = log_probs_dict['token_entropy']

            # minibatch train step
            loss, _ = sft_microbatch_train_step(log_probs, response_mask, args.gradient_accumulation_steps, 1.0)
            total_loss += loss.item() * args.gradient_accumulation_steps

            wandb.log({"train/loss": loss.item() * args.gradient_accumulation_steps, "train_step": train_step})
            train_step += 1

            # # update counters and remember the most recent training loss
            tokens_seen += torch.sum(response_mask).item()

            if (idx + 1) % (args.gradient_accumulation_steps) == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # Update weights every `gradient_accumulation_steps` batches.
                optimizer.step()
                scheduler.step()
                # Zero gradients every `gradient_accumulation_steps` batches.
                optimizer.zero_grad()

        avg_loss = total_loss / len(dataloader)
        print(f"epoch: {epoch}, training loss: {avg_loss:.3f}")
        
        print("evaluation:")
        with torch.no_grad():
            load_policy_into_vllm_instance(model, vllm_model)

            serialize_path = Path(args.output_dir) / f"evaluate_results_epoch{epoch+1}.json"
            results = evaluate_vllm(vllm_model, r1_zero_reward_fn, valid_prompts, valid_responses, valid_answers, sampling_params, serialize_path, True)
            
            # filted_samples = {}
            # for key, value in results.items():
            #     if isinstance(key, int):
            #         if int(values['rewards']['answer_reward']) == 1

            print('###########################################################')
            print(f'Epoch {epoch}, Validation results')
            print(f"Number of different format and answer rewards is {results['eval_metrics_nums']}")
            print(f"Ratio of different format and answer rewards is {results['eval_metrics_ratios']}")
            print(f"Accuracy metrics is {results['accuracy']}")

            wandb.log({
                'eval/total_seen_tokens': tokens_seen,
                'eval/training_avg_loss': avg_loss,
                'eval/rewards': results['eval_metrics_nums'],
                'eval/reward_ratios': results['eval_metrics_ratios'],
                'eval/accuracy': results['accuracy'],
                'eval_step': epoch
                })

            if epoch == args.num_epochs - 1:
                save_dir = Path(args.output_dir) / f"checkpoint-epoch{epoch+1}"
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
    parser = argparse.ArgumentParser(description='training codes for supervised finetuning')

    parser.add_argument('--num_epochs', type=int, default=10, help='total number of training epoch')
    parser.add_argument('--num_samples', type=int, default=7472, help='number of unique examples for SFT')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='ratios for warmup steps')
    parser.add_argument('--batch_size', type=int, default=2, help='batch_size during training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='gradient_accumulation_steps')
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/outputs/2025_1202_sft/7472samples")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--run_name", type=str, default=(lambda: datetime.now().strftime("%Y-%m%d-%H%M%S"))(), help="Experiment name, default is current timestamp")
    
    return parser.parse_args()


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 确保确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    args = get_training_args()

    set_seed(42)

    main(args)