from transformers import AutoModelForCausalLM, AutoTokenizer

def load_mistral_model():
    print("🔄 Loading Mistral-7B...")
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # This will use CUDA if available, otherwise CPU
        torch_dtype="auto"  # Or: torch.float16 if GPU is available, otherwise leave as auto
    )

    def generate(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=256)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    return generate
