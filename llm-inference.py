import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# Corrected (RIGHT)
from requests.exceptions import HTTPError

# Load the Hugging Face token from the environment
hf_token = os.getenv("HUGGINGFACE_TOKEN")

if not hf_token:
    raise EnvironmentError("HUGGINGFACE_TOKEN not found. Please export it to your environment.")

try:
    # Load LLaMA Model with the token
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        use_auth_token=hf_token,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        use_auth_token=hf_token,
        trust_remote_code=True
    )
except HTTPError as e:
    print("\n[ERROR] Authentication failed. Please check if your token has access permissions.")
    print(f"Details: {e}")
    exit(1)

# Move model to GPU if available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

# Example prompt
prompt = "How does semantic communication improve satellite efficiency?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate text
output = model.generate(**inputs, max_length=100, temperature=0.7)
response = tokenizer.decode(output[0], skip_special_tokens=True)

# Display the result
print("\n=== Inference Result ===")
print(response)