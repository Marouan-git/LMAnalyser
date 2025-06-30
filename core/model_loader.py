import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_path: str, use_auth_token: bool = True):
    """
    Loads a Hugging Face model and tokenizer.

    Args:
        model_path (str): The path or name of the model on Hugging Face Hub.
        use_auth_token (bool): Whether to use an auth token for gated models.
                               Note: Llama 2 requires an auth token.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    print(f"Loading model and tokenizer from '{model_path}'...")
    
    # Define the data type for loading the model. bfloat16 is preferred for recent GPUs.
    # float16 is a safe fallback.
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            token=use_auth_token,
            trust_remote_code=True
        )
        
        # Use device_map='auto' to automatically distribute the model across available GPUs.
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            token=use_auth_token,
            trust_remote_code=True
        )
        
        # Set pad token if it's not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        print("Model and tokenizer loaded successfully.")
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nNOTE: To access Llama 2 models, you must:")
        print("1. Visit https://ai.meta.com/resources/models-and-libraries/llama-downloads/ and accept the license.")
        print("2. Be granted access by Meta (this can take a few days).")
        print("3. Log in to your Hugging Face account using `huggingface-cli login` in your terminal.")
        return None, None
