from transformers import AutoConfig, AutoModelForCausalLM

# Define your model names
model_bases = [
               "moon4sake/Llama-3.1-8B_s0.50_block",
               "moon4sake/Llama-3.1-8B_s0.75_block",
               "moon4sake/DeepSeek-R1-Distill-Llama-8B_s0.25_block",
               "moon4sake/DeepSeek-R1-Distill-Llama-8B_s0.50_block",
               "moon4sake/DeepSeek-R1-Distill-Llama-8B_s0.75_block",
               "moon4sake/DeepSeek-R1-Distill-Qwen-1.5B_s0.50_block",
               "moon4sake/DeepSeek-R1-Distill-Qwen-1.5B_s0.75_block",
               ]
AUTH_TOKEN = "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"

# Loop over each model, update the configuration, and save/push
for model_base in model_bases:
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_base, ignore_mismatched_sizes=True)
    
    # Calculate updated number of heads and intermediate sizes
    for layer in model.model.layers:
        layer.self_attn.num_heads = layer.self_attn.q_proj.weight.size(0) // layer.self_attn.head_dim
        layer.self_attn.num_key_value_heads = layer.self_attn.k_proj.weight.size(0) // layer.self_attn.head_dim
        layer.mlp.intermediate_size = layer.mlp.down_proj.weight.size(0)

    model.config.num_attention_heads = model.model.layers[0].self_attn.num_heads
    model.config.num_key_value_heads = model.model.layers[0].self_attn.num_key_value_heads
    model.config.intermediate_size = model.model.layers[0].mlp.intermediate_size

    # Save updated model configuration back to model directory
    model.save_pretrained(model_base)
    print(f"Updated config for {model_base}")

    # Verification step: Load the model again to ensure it's correct
    try:
        test_model = AutoModelForCausalLM.from_pretrained(model_base)
        print(f"Model {model_base} reloaded successfully with no mismatch errors.")
    except Exception as e:
        print(f"Error reloading model {model_base}: {e}")

    
    # Push updated model to Hugging Face hub
    REPO_NAME = model_base.split('/')[-1]
    model.push_to_hub(REPO_NAME, use_temp_dir=True, token=AUTH_TOKEN)

    print(f"Pushed updated config for {model_base} to Hugging Face hub.")
