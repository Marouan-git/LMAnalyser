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
        "--eval_act_kurtosis",
        action="store_true",
        help="Analyzes and plots the kurtosis of activation distributions."
    )
    func_group.add_argument(
        "--eval_per_token_kurtosis",
        action="store_true",
        help="Finds top tokens with the highest kurtosis by module and by layer."
    )
    func_group.add_argument(
        "--eval_down_proj_spikes",
        action="store_true",
        help="Visualizes top token activation spikes for each down_proj module."
    )
    func_group.add_argument(
        "--eval_token_occurrence",
        action="store_true",
        help="Analyzes activation magnitude of a specific token vs. its occurrence number."
    )
    func_group.add_argument(
        "--analyze_prompt",
        type=str,
        help="Analyzes and plots activation magnitudes for a given text prompt."
    )
    func_group.add_argument(
        "--eval_bops",
        action="store_true",
        help="Calculates and plots the theoretical Bit-Operations (BOPs) for the model."
    )
    func_group.add_argument(
        "--eval_fisher_info",
        action="store_true",
        help="Calculates and plots the Fisher Information for activations."
    )
    func_group.add_argument(
        "--eval_max_median_ratio",
        action="store_true",
        help="Calculates and plots the max-to-median ratio of activation scales."
    )
    func_group.add_argument(
        "--eval_fgmp_sensitivity",
        action="store_true",
        help="Calculates the FGMP sensitivity metric for activations."
    )
    func_group.add_argument(
        "--plot_results",
        action="store_true",
        help="If set, generates and saves plots for the analysis results."
    )

    # --- Analysis / Quantization Arguments ---
    analysis_group = parser.add_argument_group('Analysis and Quantization Options')
    analysis_group.add_argument("--quant_bits", type=int, default=4, help="Number of bits for quantization (e.g., 8, 4).")
    analysis_group.add_argument(
        "--quant_granularity", 
        type=str, 
        default="per-token", 
        choices=["per-tensor", "per-token"],
        help="Granularity for activation quantization (used for --eval_quant_error)."
    )
    analysis_group.add_argument(
        "--calib_dataset", 
        type=str, 
        default="wikitext2", 
        choices=["c4", "wikitext2"],
        help="Dataset to use for calibration."
    )
    analysis_group.add_argument("--calib_samples", type=int, default=128, help="Number of samples to use for calibration.")
    analysis_group.add_argument(
        "--target_token",
        type=str,
        default="\n",
        help="The specific token string to analyze for the occurrence analysis."
    )
    analysis_group.add_argument(
        "--layers_to_plot",
        type=int,
        nargs='+',
        default=None,
        help="A list of layer indices to plot for prompt analysis (e.g., 0 15 31)."
    )
    analysis_group.add_argument(
        "--use_hadamard_transform",
        action="store_true",
        help="Apply Hadamard transform to activations before analysis."
    )

    analysis_group.add_argument(
        "--exclude_tokens",
        type=str,
        nargs='+',
        default=None,
        help="A list of token strings to exclude from magnitude analysis (e.g., --exclude_tokens '.' ',' '\n')."
    )
    analysis_group.add_argument(
        "--block_size",
        type=int,
        default=128,
        help="Block size for block-wise analysis (e.g., FGMP)."
    )
    analysis_group.add_argument("--high_prec_bits", type=int, default=16, help="High precision bit-width for FGMP analysis.")
    analysis_group.add_argument("--low_prec_bits", type=int, default=8, help="Low precision bit-width for FGMP analysis.")

    args = parser.parse_args()
    
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    if model is None or tokenizer is None:
        return # Exit if model loading failed
    
    # Instantiate the analyzer
    analyzer = LLMAnalyser(model, tokenizer, use_hadamard_transform=args.use_hadamard_transform)
    
    # --- Conditionally launch functionalities ---
    
    if args.eval_perplexity:
        evaluate_perplexity(model, tokenizer)

    if args.eval_quant_error:
        analyzer.run_quantization_error_analysis(
            calib_dataset=args.calib_dataset,
            bits=args.quant_bits, granularity=args.quant_granularity,
            num_samples=args.calib_samples, plot=args.plot_results, exclude_tokens=args.exclude_tokens
        )

    if args.eval_top_token_error:
        analyzer.run_per_token_error_analysis(
            calib_dataset=args.calib_dataset,
            bits=args.quant_bits, granularity=args.quant_granularity,
            num_samples=args.calib_samples, plot=args.plot_results, exclude_tokens=args.exclude_tokens
        )

    if args.eval_act_magnitude:
        analyzer.run_activation_magnitude_analysis(
            calib_dataset=args.calib_dataset,
            num_samples=args.calib_samples, plot=args.plot_results,
            exclude_tokens=args.exclude_tokens
        )
    
    if args.eval_act_kurtosis:
        analyzer.run_activation_kurtosis_analysis(
            calib_dataset=args.calib_dataset,
            num_samples=args.calib_samples, plot=args.plot_results
        )
    
    if args.eval_per_token_kurtosis:
        analyzer.run_per_token_kurtosis_analysis(
            calib_dataset=args.calib_dataset,
            num_samples=args.calib_samples, plot=args.plot_results
        )

    if args.eval_down_proj_spikes:
        analyzer.run_down_proj_spike_analysis(
            calib_dataset=args.calib_dataset,
            num_samples=args.calib_samples, plot=args.plot_results
        )
    
    if args.eval_token_occurrence:
        analyzer.run_token_occurrence_analysis(
            calib_dataset=args.calib_dataset,
            target_token_str=args.target_token,
            num_samples=args.calib_samples,
            plot=args.plot_results
        )
    
    if args.analyze_prompt:
        analyzer.run_prompt_spike_analysis(
            prompt_text=args.analyze_prompt,
            plot=args.plot_results,
            layers_to_plot=args.layers_to_plot
        )

    if args.eval_bops:
        analyzer.run_bops_analysis(
            bits=args.quant_bits,
            plot=args.plot_results
        )
    
    if args.eval_fisher_info:
        analyzer.run_fisher_information_analysis(
            calib_dataset=args.calib_dataset,
            num_samples=args.calib_samples,
            plot=args.plot_results
        )

    if args.eval_max_median_ratio:
        analyzer.run_max_median_ratio_analysis(
            calib_dataset=args.calib_dataset,
            num_samples=args.calib_samples,
            plot=args.plot_results
        )

    if args.eval_fgmp_sensitivity:
        analyzer.run_fgmp_sensitivity_analysis(
            calib_dataset=args.calib_dataset,
            num_samples=args.calib_samples,
            plot=args.plot_results,
            high_prec_bits=args.high_prec_bits,
            low_prec_bits=args.low_prec_bits,
            block_size=args.block_size
        )

    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()