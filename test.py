from transformers import LlamaForCausalLM, LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained('pretrained_models/Llama2-7b-hf')
model = LlamaForCausalLM.from_pretrained('pretrained_models/Llama2-7b-hf')

input_text = "Hello, how are you?"
inputs = tokenizer.encode(input_text, return_tensors="pt")
outputs = model.generate(inputs, max_length=50, num_return_sequences=5, temperature=0.7)
print('Generated text:')
for i, output in enumerate(outputs):
    print("{}: {}".format(i, tokenizer.decode(output, skip_special_tokens=True)))