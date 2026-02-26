import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# 1. Setup Device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Running on: {device}")

# 2. Model Selection
model_id = "microsoft/Phi-3-mini-4k-instruct"
print(f"Loading {model_id}...")

# 3. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 4. Load Config and Fix rope_scaling
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

# Fix for KeyError: 'type' - ensure rope_scaling has proper structure
if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
    if 'type' not in config.rope_scaling:
        config.rope_scaling['type'] = 'linear'

# 5. Load Model with Fixed Config
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    attn_implementation='eager',
).to(device)

# 6. Chat Loop
print("\n--- Phi-3 Mini Chat Started (Type 'quit' to exit) ---")
conversation = []

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "quit":
        break

    conversation.append({"role": "user", "content": user_input})

    input_text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    print(f"AI: {response}")
    conversation.append({"role": "assistant", "content": response})