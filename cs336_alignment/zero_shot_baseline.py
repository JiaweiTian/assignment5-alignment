from vllm import LLM, SamplingParams
from drgrpo_grader import r1_zero_reward_fn
from datasets import load_dataset
import os
import json

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

#########################################################
# Model and file path
#########################################################
model_name = "Qwen/Qwen2.5-Math-1.5B"
serialize_path = '/Users/jiawei/Desktop/cs336/assignment5-alignment/cs336_alignment/evaluate_results.json'

#########################################################
# function
#########################################################
def load_cot_prompt(question):
    """读取思维链提示模板"""
    prompt_template = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {question}. Assistant: <think>"""

    return prompt_template.format(question=question)


def look_up_dataset():
    dataset = load_dataset("openai/gsm8k", "main")

    train_data = dataset['train']
    validation_data = dataset['test']

    print(f'len of train_data and validation_data are {len(train_data)} and {len(validation_data)}')
    print(f'Traing data sample: {train_data[0]}')
    print(f'Valid data sample: {validation_data[0]}')
    print('------------------------------------------')

    return train_data, validation_data


def evaluate_vllm(vllm_model, reward_fn, prompts, answers, eval_sampling_params, save_results=False):
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    results = {}
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
        temp['answer'] = answers[idx]
        temp['reward'] = reward
        results[idx] = temp

        format_reward = int(reward['format_reward'])
        answer_reward = int(reward['answer_reward'])
        if format_reward == 1 and answer_reward == 1:
            categories['format_1_answer_1'] += 1
        elif format_reward == 0 and answer_reward == 1:
            categories['format_0_answer_1'] += 1
        elif format_reward == 1 and answer_reward == 0:
            categories['format_1_answer_0'] += 1
        elif format_reward == 0 and answer_reward == 0:
            categories['format_0_answer_0'] += 1

        accuracy['total_accuracy'] += reward['reward'] / num_prompts
        accuracy['format_accuracy'] += reward['format_reward'] / num_prompts
        accuracy['answer_accuracy'] += reward['answer_reward'] / num_prompts

    categories_ratios['format_1_answer_1'] = categories['format_1_answer_1'] / num_prompts
    categories_ratios['format_0_answer_0'] = categories['format_0_answer_0'] / num_prompts
    categories_ratios['format_1_answer_0'] = categories['format_1_answer_0'] / num_prompts
    categories_ratios['format_0_answer_1'] = categories['format_0_answer_1'] / num_prompts

    results['eval_metrics_nums'] = categories
    results['eval_metrics_ratios'] = categories_ratios
    results['accuracy'] = accuracy

    if save_results:
        with open(serialize_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Save evaluate results to {serialize_path}")

    print('categories is {categories}')
    print('categories_ratios is {categories_ratios}')
    print('accuracy is {accuracy}')

    return results


def math_baseline():
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024)
    sampling_params.stop = ["</answer>"]
    sampling_params.include_stop_str_in_output = True

    llm = LLM(model=model_name)

    train_data, valid_data = look_up_dataset()

    prompts = [load_cot_prompt(valid_data[i]['question']) for i in range(len(valid_data))]
    answers = [valid_data[i]['answer'] for i in range(len(valid_data))]


    _ = evaluate_vllm(llm, r1_zero_reward_fn, prompts, answers, sampling_params, True)


if __name__ == "__main__":
    math_baseline()