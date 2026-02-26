import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# 1. Setup Device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Running on: {device}")

# 2. Model Selection (Use the one that works for you)
# Change this to "HuggingFaceTB/SmolLM2-1.7B-Instruct" if you prefer SmolLM
model_id = "microsoft/Phi-3-mini-4k-instruct"
print(f"Loading {model_id}... (please wait)")

# 3. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 4. Load Config and Fix rope_scaling (The fix that made it work)
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
    if 'type' not in config.rope_scaling:
        config.rope_scaling['type'] = 'linear'

# 5. Load Model (Global scope so it loads only ONCE)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    attn_implementation='eager',
).to(device)

# 6. Define the Chat Function
def chat(message, history):
    # Convert Gradio history to format the model understands
    conversation = []
    for user_msg, assistant_msg in history:
        conversation.append({"role": "user", "content": user_msg})
        conversation.append({"role": "assistant", "content": assistant_msg})
    conversation.append({"role": "user", "content": message})

    # Format with chat template
    input_text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode response
    response = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    
    return response

# 7. Launch the Interface
print("Launching Chat Interface...")
demo = gr.ChatInterface(
    fn=chat, 
    title="My Mac Mini M4 AI", 
    description="Running locally on Apple Silicon"
)
demo.launch()