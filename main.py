import argparse
from core.model_loader import load_model_and_tokenizer
from evaluation.perplexity import evaluate_perplexity
from analysis.analyser import LLMAnalyser

def main():
    """
    Main function to run the LLM Analysis Tool.
    """
    parser = argparse.ArgumentParser(description="LLM Analysis Tool", formatter_class=argparse.RawTextHelpFormatter)
    
    # --- Model Arguments ---
    parser.add_argument(
        "--model_path",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Path to the Hugging Face model or model identifier."
    )
    
    # --- Functionality Arguments ---
    func_group = parser.add_argument_group('Functionalities')
    func_group.add_argument(
        "--eval_perplexity",
        action="store_true",
        help="If set, evaluates the model's perplexity on the WikiText-2 dataset."
    )
    func_group.add_argument(
        "--eval_quant_error",
        action="store_true",
        help="If set, evaluates the quantization error (MSE) for model activations."
    )
    func_group.add_argument(
        "--eval_top_token_error",
        action="store_true",
        help="If set, finds the top 10 tokens with the highest per-token quantization error."
    )
    func_group.add_argument(
        "--eval_act_magnitude",
        action="store_true",
        help="Analyzes and plots activation magnitudes."
    )
    func_group.add_argument(
        "--plot_results",
        action="store_true",
        help="If set, generates and saves plots for the analysis results."
    )

    # --- Quantization Arguments ---
    quant_group = parser.add_argument_group('Quantization Options')
    quant_group.add_argument("--quant_bits", type=int, default=4, help="Number of bits for quantization (e.g., 8, 4).")
    quant_group.add_argument(
        "--quant_granularity", 
        type=str, 
        default="per-token", 
        choices=["per-tensor", "per-token"],
        help="Granularity for activation quantization (used for --eval_quant_error)."
    )
    quant_group.add_argument("--quant_calib_samples", type=int, default=128, help="Number of C4 samples to use for calibration.")

    
    args = parser.parse_args()
    
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    if model is None or tokenizer is None:
        return # Exit if model loading failed
    
    # Instantiate the analyzer
    analyzer = LLMAnalyser(model, tokenizer)
    
    # --- Conditionally launch functionalities ---
    
    if args.eval_perplexity:
        evaluate_perplexity(model, tokenizer)

    if args.eval_quant_error:
        analyzer.run_quantization_error_analysis(
            bits=args.quant_bits,
            granularity=args.quant_granularity,
            num_samples=args.quant_calib_samples,
            plot=args.plot_results
        )

    if args.eval_top_token_error:
        analyzer.run_per_token_error_analysis(
            bits=args.quant_bits,
            granularity=args.quant_granularity,
            num_samples=args.quant_calib_samples,
            plot=args.plot_results
        )
    if args.eval_act_magnitude:
        analyzer.run_activation_magnitude_analysis(
            num_samples=args.quant_calib_samples, plot=args.plot_results
        )

        
    # --- Other functionalities will be added here in the future ---
    
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()