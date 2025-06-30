from datasets import load_dataset

def get_wikitext2_test_data(tokenizer, seq_len: int = 2048):
    """
    Loads and prepares the WikiText-2 test dataset.

    Args:
        tokenizer: The tokenizer to use for encoding the text.
        seq_len (int): The sequence length for the model.

    Returns:
        torch.Tensor: A tensor of the tokenized test data.
    """
    print("Loading and preparing WikiText-2 test data...")
    try:
        test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        test_encodings = tokenizer("\n\n".join(test_data["text"]), return_tensors="pt")
        print(f"WikiText-2 test data loaded. Total tokens: {test_encodings.input_ids.size(1)}")
        return test_encodings
        
    except Exception as e:
        print(f"Failed to load or process wikitext-2 dataset: {e}")
        return None

def get_c4_calibration_data(tokenizer, n_samples: int = 128, seq_len: int = 512):
    """
    Loads and prepares a subset of the C4 dataset for calibration.

    Args:
        tokenizer: The tokenizer to use for encoding the text.
        n_samples (int): The number of samples to use for calibration.
        seq_len (int): The sequence length for each sample.

    Returns:
        list: A list of tokenized samples (as dicts of tensors).
    """
    print("Loading C4 calibration data...")
    try:
        # Use streaming to avoid downloading the entire dataset
        calib_dataset = load_dataset("c4", "en", split="train", streaming=True)
        
        tokenized_samples = []
        for sample in calib_dataset.take(n_samples):
            print(f"Processing sample: {len(sample['text'])}...")
            tokenized = tokenizer(
                sample['text'], 
                return_tensors="pt", 
                max_length=seq_len, 
                padding="max_length", # This ensures shorter sequences are padded to seq_len
                truncation=True
            )
            tokenized_samples.append(tokenized)

        # Return a list of tensors for easy iteration
        return tokenized_samples
        
    except Exception as e:
        print(f"Failed to load or process C4 dataset: {e}")
        return None