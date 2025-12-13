# 1. math.baseline

(b)

**eval_metrics_nums**

```json
{
    "format_1_answer_1": 35,
    "format_1_answer_0": 230,
    "format_0_answer_1": 0,
    "format_0_answer_0": 1054
}
```

说明：

- 格式得分为 0 的 prompt response 主要与 Qwen / Qwen2.5-Math-1.5B 本身有关。模型在最初训练时并没有遵循 `r1_zero_prompt` 的格式，导致它不能很好地遵循该指令格式。
- 同时，答案正确性为 0 的 prompt response 表明在没有针对相关问题进行训练之前，模型的泛化能力不够好。



# 5. expert iteration
我选去了G=2, batch_size=7472, epochs=2, n_ei_steps=5; 与此对比的是不做expert iteration的sft, 也是选去7472个samples，一共训练10个epochs; 从结果来看，采取了expert iteration的经过训练后，达到了更高的accuracy; 对于format_correct的比例，相比下从97.8%降低到了89.9%， 这可能是跟训练样本数变小了有关，因为expert iteration需要做sampling的缘故，每一个n_ei_step，真正用来训练的样本数分别是345, 3096, 6029, 7834, 8508; 而不使用expert iteration的sft每一轮的样本数都是7472， 所以更有利于模型学到format相关的要求；实验结果可以证明，使用了expert iteration的确可以提升模型的推理能力，生成更准确的答案

sft_without_expert_iteration
```json
{
    'format_1_answer_1': 0.45034116755117515, 
    'format_1_answer_0': 0.5276724791508719, 
    'format_0_answer_1': 0.0, 
    'format_0_answer_0': 0.021986353297952996
}

{
    'total_accuracy': 0.45034116755118225, 
    'format_accuracy': 0.9780136467020382, 
    'answer_accuracy': 0.45034116755118225
}
```

sft_with_expert_iteration
```json
{
    'format_1_answer_1': 0.4920394238059136, 
    'format_1_answer_0': 0.4071266110689917, 
    'format_0_answer_1': 0.0, 
    'format_0_answer_0': 0.10083396512509477
}

{
    'total_accuracy': 0.4920394238059222, 
    'format_accuracy': 0.8991660348748993, 
    'answer_accuracy': 0.4920394238059222
}
```