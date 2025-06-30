import torch
from tqdm import tqdm
from data.data_loader import get_wikitext2_test_data

@torch.no_grad()
def evaluate_perplexity(model, tokenizer, dataset_name: str = "wikitext2", seq_len: int = 2048):
    """
    Calculates the perplexity of a model on a given dataset using a non-overlapping chunking method,
    matching the methodology used in many research papers like PrefixQuant.

    Args:
        model: The language model to evaluate.
        tokenizer: The tokenizer associated with the model.
        dataset_name (str): The name of the dataset to use. Currently only 'wikitext2' is supported.
        seq_len (int): The sequence length to use for evaluation chunks.
    """
    if dataset_name.lower() != "wikitext2":
        print(f"Dataset '{dataset_name}' is not supported for perplexity evaluation.")
        return

    print(f"Starting perplexity evaluation on {dataset_name} using non-overlapping chunks...")
    model.eval()
    
    # Get the tokenized test data
    test_data = get_wikitext2_test_data(tokenizer, seq_len)
    if test_data is None:
        return
        
    test_encodings = test_data.input_ids
    
    # Calculate the number of non-overlapping samples
    num_samples = test_encodings.size(1) // seq_len
    
    nlls = []
    
    # Use tqdm for a progress bar
    for i in tqdm(range(num_samples)):
        start_index = i * seq_len
        end_index = (i + 1) * seq_len
        
        input_ids = test_encodings[:, start_index:end_index].to(model.device)
        
        # The labels are the same as the input_ids for language modeling.
        target_ids = input_ids.clone()

        # Calculate loss
        outputs = model(input_ids, labels=target_ids)
        
        # The returned loss is the average negative log-likelihood per token.
        # We multiply by the sequence length to get the NLL for the entire chunk.
        neg_log_likelihood = outputs.loss * seq_len
        
        nlls.append(neg_log_likelihood)
            
    # Calculate perplexity over the entire dataset
    # PPL = exp(total_nll / total_tokens)
    total_nll = torch.stack(nlls).sum()
    total_tokens = num_samples * seq_len
    ppl = torch.exp(total_nll / total_tokens)
    
    print(f"\nPerplexity on {dataset_name}: {ppl.item():.4f}")


# import torch
# from tqdm import tqdm
# from data.data_loader import get_wikitext2_test_data

# @torch.no_grad()
# def evaluate_perplexity(model, tokenizer, dataset_name: str = "wikitext2", seq_len: int = 2048):
#     """
#     Calculates the perplexity of a model on a given dataset.

#     Args:
#         model: The language model to evaluate.
#         tokenizer: The tokenizer associated with the model.
#         dataset_name (str): The name of the dataset to use. Currently only 'wikitext2' is supported.
#         seq_len (int): The sequence length to use for evaluation.
#     """
#     if dataset_name.lower() != "wikitext2":
#         print(f"Dataset '{dataset_name}' is not supported for perplexity evaluation.")
#         return

#     print(f"Starting perplexity evaluation on {dataset_name}...")
    
#     # Get the tokenized test data
#     test_data = get_wikitext2_test_data(tokenizer, seq_len)
#     if test_data is None:
#         return
        
#     test_encodings = test_data.input_ids
    
#     # Define sequence length and stride
#     max_length = model.config.max_position_embeddings
#     stride = 512  # A common stride value to balance speed and coverage
    
#     seq_len = test_encodings.size(1)
    
#     nlls = []
#     prev_end_loc = 0
    
#     # Use tqdm for a progress bar
#     for begin_loc in tqdm(range(0, seq_len, stride)):
#         end_loc = min(begin_loc + max_length, seq_len)
#         trg_len = end_loc - prev_end_loc
#         input_ids = test_encodings[:, begin_loc:end_loc].to(model.device)
#         target_ids = input_ids.clone()
#         target_ids[:, :-trg_len] = -100  # -100 is the ignore_index for CrossEntropyLoss

#         # Calculate loss
#         outputs = model(input_ids, labels=target_ids)
#         neg_log_likelihood = outputs.loss
        
#         nlls.append(neg_log_likelihood)
        
#         prev_end_loc = end_loc
#         if end_loc == seq_len:
#             break
            
#     # Calculate perplexity
#     ppl = torch.exp(torch.stack(nlls).mean())
#     print(f"\nPerplexity on {dataset_name}: {ppl.item():.4f}")

