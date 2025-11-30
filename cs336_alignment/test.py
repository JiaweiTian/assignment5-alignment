# from vllm import LLM, SamplingParams
import torch
import torch.nn.functional as F

if __name__ == "__main__":
	# Sample prompts.
	# prompts = [
	# "Hello, my name is",
	# "The president of the United States is",
	# "The capital of France is",
	# "The future of AI is",
	# ]

	# # Create a sampling params object, stopping generation on newline.
	# sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"])

	# # Create an LLM.
	# llm = LLM(model="Qwen/Qwen2.5-Math-1.5B")

	# # Generate texts from the prompts. The output is a list of RequestOutput objects
	# # that contain the prompt, generated text, and other information.
	# outputs = llm.generate(prompts, sampling_params)

	# # Print the outputs.
	# for output in outputs:
	# 	prompt = output.prompt
	# 	generated_text = output.outputs[0].text
	# 	print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

	# prompts = [
	#     "Hello, my name is",
	#     "The president of the United States is",
	#     "The capital of France is",
	#     "The future of AI is",
	# ]

	# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

	# llm = LLM(model="facebook/opt-125m")

	# outputs = llm.generate(prompts, sampling_params)

	# for output in outputs:
	#     prompt = output.prompt
	#     generated_text = output.outputs[0].text
	#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

	# 验证代码

    logits = torch.randn(2, 5, 1000)  # batch=2, seq_len=5, vocab=1000
    labels = torch.randint(0, 1000, (2, 5))
    
    print("logits shape:", logits.shape)
    print("labels shape:", labels.shape)
    
    # 正确方法 - 手动提取对数概率
    log_probs = F.log_softmax(logits, dim=-1)
    batch_idx = torch.arange(2).unsqueeze(1).expand(-1, 5)
    seq_idx = torch.arange(5).unsqueeze(0).expand(2, -1)
    correct_log_probs = log_probs[batch_idx, seq_idx, labels]
    
    print("correct_log_probs shape:", correct_log_probs.shape)
    print("Correct log_probs sample:", correct_log_probs[0, :3])
    
    # 错误方法 - 使用F.cross_entropy的方式
    # 需要重新塑造维度
    logits_reshaped = logits.reshape(-1, 1000)  # [10, 1000]
    labels_reshaped = labels.reshape(-1)        # [10]
    
    wrong_log_probs = F.cross_entropy(logits_reshaped, labels_reshaped, reduction='none')
    wrong_log_probs = wrong_log_probs.reshape(2, 5)  # 恢复原始形状
    
    print("wrong_log_probs shape:", wrong_log_probs.shape)
    print("Wrong log_probs sample:", wrong_log_probs[0, :3])
    
    # 验证关系
    print("All values negative?", (wrong_log_probs <= 0).all())  # False - 应该是正的
    print("All values reasonable?", (correct_log_probs <= 0).all())  # True
    print("Relationship:", torch.allclose(wrong_log_probs, -correct_log_probs))  # True