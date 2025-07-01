import torch
from datasets import load_dataset

def get_wikitext2_test_data(tokenizer, seq_len: int = 2048):
    """
    Loads and prepares the WikiText-2 test dataset for perplexity evaluation.

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

def get_calibration_data(dataset_name: str, tokenizer, n_samples: int = 128, seq_len: int = 512):
    """
    Loads and prepares a subset of a specified dataset for calibration.

    Args:
        dataset_name (str): The name of the dataset ('c4' or 'wikitext2').
        tokenizer: The tokenizer to use for encoding the text.
        n_samples (int): The number of samples to use for calibration.
        seq_len (int): The sequence length for each sample.

    Returns:
        list: A list of tokenized samples (as dicts of tensors).
    """
    print(f"Loading '{dataset_name}' for calibration...")
    
    if dataset_name == 'c4':
        try:
            # Use streaming to avoid downloading the entire dataset
            calib_dataset = load_dataset("c4", "en", split="train", streaming=True)
            
            tokenized_samples = []
            for sample in calib_dataset.take(n_samples):
                tokenized = tokenizer(
                    sample['text'], 
                    return_tensors="pt", 
                    max_length=seq_len, 
                    padding="max_length",
                    truncation=True
                )
                tokenized_samples.append(tokenized)
            return tokenized_samples
            
        except Exception as e:
            print(f"Failed to load or process C4 dataset: {e}")
            return None
            
    elif dataset_name == 'wikitext2':
        try:
            train_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            # Filter out empty or very short lines
            train_data = train_data.filter(lambda example: len(example['text']) > 10)
            
            # Concatenate all text and then chunk into samples
            all_text = "\n\n".join(train_data["text"])
            tokenized_text = tokenizer(all_text, return_tensors="pt")
            
            input_ids = tokenized_text.input_ids[0]
            tokenized_samples = []
            
            for i in range(n_samples):
                start_index = i * seq_len
                end_index = start_index + seq_len
                if end_index > input_ids.size(0):
                    break # Stop if we run out of tokens
                
                sample_ids = input_ids[start_index:end_index].unsqueeze(0)
                # Create a dictionary similar to what the tokenizer returns
                sample = {
                    'input_ids': sample_ids,
                    'attention_mask': torch.ones_like(sample_ids)
                }
                tokenized_samples.append(sample)
            
            if not tokenized_samples:
                print("Warning: Could not generate any calibration samples from wikitext2.")
                return None
                
            return tokenized_samples

        except Exception as e:
            print(f"Failed to load or process wikitext-2 dataset: {e}")
            return None
    else:
        raise ValueError(f"Unsupported calibration dataset: {dataset_name}")

