# LMAnalyser

This tool is designed to analyze the internal states of Large Language Models (LLMs), particularly Llama-like architectures, to better understand their behavior and identify opportunities for optimization, such as quantization. It provides several analysis functionalities that can be run from the command line.

## 1. Setup

### Prerequisites
- Python 3.9+
- An NVIDIA GPU with CUDA support is highly recommended.

### Installation
1.  Clone the repository:
    ```bash
    git clone [https://github.com/Marouan-git/LMAnalyser.git](https://github.com/Marouan-git/LMAnalyser.git)
    cd LMAnalyser
    ```

2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Hugging Face Authentication**:
    The default model (`meta-llama/Llama-2-7b-hf`) is a gated model. To use it, you must first be granted access by Meta and then log in to your Hugging Face account via the terminal:
    ```bash
    huggingface-cli login
    ```
    Enter your Hugging Face access token when prompted.

## 2. Functionalities

All functionalities are run via the `main.py` script. You can specify which analysis to perform using command-line arguments.

### a. Perplexity Evaluation

**Purpose**: Measures the model's language modeling performance on a standard dataset. A lower perplexity score indicates a better model. This is useful for establishing a baseline performance metric before and after applying optimizations. Perplexity is evaluated on the Wikitext2 dataset.

**Command**:
```bash
python main.py --model_path <model-id> --eval_perplexity
```
*Example:*
```bash
python main.py --model_path meta-llama/Llama-2-7b-hf --eval_perplexity
```

---

### b. Activation Quantization Error

**Purpose**: Simulates the effect of quantization on activations and measures the resulting error (Mean Squared Error). This helps identify which layers and modules are most sensitive to precision loss.

-   `--eval_quant_error`: Enables this analysis.
-   `--quant_bits`: Set the number of bits (e.g., 4 or 8).
-   `--quant_granularity`: Choose between `per-token` and `per-tensor`.
-   `--calib_dataset`: Choose the calibration dataset (`c4` or `wikitext2`).
-   `--plot_results`: Generates and saves plots of the results.

**Command**:
```bash
python main.py --eval_quant_error --quant_bits <bits> --quant_granularity <granularity> --calib_dataset <datastet> --plot_results
```
*Example (4-bit, per-token quantization):*
```bash
python main.py --eval_quant_error --quant_bits 4 --quant_granularity per-token --calib_dataset c4 --plot_results
```

---

### c. Top Token Quantization Error

**Purpose**: Pinpoints the specific tokens whose activation vectors are most distorted by quantization. This is key to finding outliers that might require special handling (e.g., via techniques like PrefixQuant).

-   `--eval_top_token_error`: Enables this analysis.
-   Other arguments are the same as for quantization error.

**Command**:
```bash
python main.py --eval_top_token_error --quant_bits <bits> --quant_granularity <granularity> --plot_results
```
*Example (4-bit, per-tensor quantization):*
```bash
python main.py --eval_top_token_error --quant_bits 4 --quant_granularity per-tensor --plot_results
```

---

### d. Activation Magnitude Analysis

**Purpose**: Measures the raw scale (magnitude) of activations. High magnitudes often indicate the presence of outliers, which are a primary cause of quantization difficulties. This unified analysis provides three perspectives:
1.  **Per-Layer**: Shows how the average maximum magnitude changes as you go deeper into the model.
2.  **Per-Module**: Compares the overall magnitude across different module types (e.g., `q_proj` vs. `down_proj`).
3.  **Per-Token**: Identifies which specific tokens consistently produce the highest magnitude activations.

**Command**:
```bash
python main.py --eval_act_magnitude --plot_results --calib_samples <num_samples> --calib_dataset <dataset>
```
*Example:*
```bash
python main.py --eval_act_magnitude --plot_results --calib_samples 64 --calib_dataset wikitext2
```

---

### e. Activation Kurtosis Analysis

**Purpose**: Kurtosis is a statistical measure of how "outlier-prone" a distribution is. A high kurtosis value indicates that the activation distribution has "heavy tails," meaning extreme values are more likely. This analysis directly identifies which layers and modules produce the most statistically significant outlier distributions.

**Command**:
```bash
python main.py --eval_act_kurtosis --plot_results --calib_samples <num_samples>
```
*Example:*
```bash
python main.py --eval_act_kurtosis --plot_results --calib_samples 128
```

---

### f. Per-Token Kurtosis Analysis

**Purpose**: Drills down to find the specific tokens whose activation vectors form the most outlier-prone distributions. This is a powerful way to find tokens that create quantization challenges, not just because of their scale, but because of the shape of their activation distribution.

**Command**:
```bash
python main.py --eval_per_token_kurtosis --plot_results --calib_samples <num_samples>
```
*Example:*
```bash
python main.py --eval_per_token_kurtosis --plot_results --calib_samples 64
```

---

### g. Down-Projection Spike Analysis

**Purpose**: Visualizes the top token activation spikes for each `down_proj` module. This can help identify which tokens are causing outlier issues in the MLP layers.

**Command**:
```bash
python main.py --eval_down_proj_spikes --plot_results
```

---

### h. Token Occurrence Analysis

**Purpose**: Analyzes the activation magnitude of a specific token versus its occurrence number in a sequence. This can reveal if the model's handling of a token changes depending on its position.

**Command**:
```bash
python main.py --eval_token_occurrence --target_token <token> --plot_results
```
*Example:*
```bash
python main.py --eval_token_occurrence --target_token "\n" --plot_results
```

---

### i. Prompt Analysis

**Purpose**: Analyzes and plots activation magnitudes for a given text prompt. This allows for a detailed inspection of how the model processes a specific input.

**Command**:
```bash
python main.py --analyze_prompt "<your prompt>" --plot_results
```
*Example:*
```bash
python main.py --analyze_prompt "The quick brown fox jumps over the lazy dog." --plot_results
```

---

### j. BOPs Calculation

**Purpose**: Calculates and plots the theoretical Bit-Operations (BOPs) for the model. This is useful for understanding the computational cost of the model and the potential savings from quantization.

**Command**:
```bash
python main.py --eval_bops --quant_bits <bits> --plot_results
```
*Example:*
```bash
python main.py --eval_bops --quant_bits 4 --plot_results
```

---

### k. Fisher Information Analysis

**Purpose**: Calculates and plots the Fisher Information for activations. This is a measure of how sensitive the model's output is to changes in the activations, providing another perspective on identifying critical parts of the model.

**Command**:
```bash
python main.py --eval_fisher_info --plot_results
```

---

### l. Max-to-Median Ratio Analysis

**Purpose**: Calculates and plots the max-to-median ratio of activation scales (from QFEP paper: https://arxiv.org/abs/2405.14428). This is a direct measure of outlier presence and can be a strong indicator of quantization difficulty.

**Command**:
```bash
python main.py --eval_max_median_ratio --plot_results
```

---

### m. FGMP Sensitivity Analysis

**Purpose**: Calculates the FGMP (Fine-Grained  Mixed Precision) sensitivity metric for activations (from FGMP paper: https://arxiv.org/abs/2504.14152). This metric helps in determining the optimal precision for different parts of the model in a mixed-precision quantization setup.

**Command**:
```bash
python main.py --eval_fgmp_sensitivity --plot_results
```

## 3. General Options

-   `--model_path <id>`: Specify which model to analyze (e.g., `meta-llama/Llama-3-8B-hf`).
-   `--calib_dataset <name>`: Choose the calibration dataset (`c4` or `wikitext2`). Default is `wikitext2`.
-   `--calib_samples <num>`: Set the number of samples to use for calibration-based analyses. Default is 128.
-   `--plot_results`: Add this flag to any analysis to generate and save corresponding plots as `.png` files.