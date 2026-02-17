# Java2CS: Transformer-based Code Transpiler

A sequence-to-sequence Deep Learning model built entirely from scratch in PyTorch to translate Java source code into C#. This project implements the original "Attention Is All You Need" architecture without relying on high-level abstraction libraries like `HuggingFace.AutoModel`.

## Key Features

* **From-Scratch Architecture:** Manual implementation of MultiHeadAttention, PositionalEncoding, LayerNorm, and FeedForward networks.
* **Custom Tokenizer:** Trained a Byte-Pair Encoding (BPE) tokenizer specifically on the CodeXGlue dataset.
* **Robust Inference:** Implemented custom decoding logic with repetition penalties to prevent loop generation in code.
* **Interpretability:** Includes an Attention Heatmap visualizer to see how the model maps Java tokens to C# tokens during translation.
* **Metric Evaluation:** Integrated BLEU score tracking using `torchmetrics`.

## Model Architecture

The model follows the standard Transformer Encoder-Decoder structure:

1.  **Input Embeddings:** Learned vector representations of code tokens scaled by the square root of `d_model`.
2.  **Positional Encoding:** Sinusoidal injections to retain the order of code tokens (crucial for syntax).
3.  **Encoder:** A stack of layers containing Self-Attention and Feed-Forward networks to understand the context of the Java code.
4.  **Decoder:** A stack of layers utilizing Cross-Attention to map the Encoder's understanding to C# syntax, utilizing Causal Masking to prevent "peeking" at future tokens.

### Hyperparameters (Configurable)

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `d_model` | 256 | Dimension of the embedding vector |
| `N` (Layers) | 2 | Number of Encoder/Decoder blocks |
| `h` (Heads) | 4 | Number of Attention Heads |
| `Context` | 256 | Max sequence length |
| `Vocab` | 30k | Tokenizer vocabulary size |

## Dataset

The model is trained on the **CodeXGlue (Coarse-Grained)** dataset.

* **Source:** `google/code_x_glue_cc_code_to_code_trans`
* **Input:** Java Methods
* **Target:** C# Methods

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/java2cs-transformer.git](https://github.com/yourusername/java2cs-transformer.git)
    cd java2cs-transformer
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch datasets torchmetrics tokenizers matplotlib seaborn
    ```

## Results

After training for 30 epochs on the subset:

* **Training Loss:** ~0.08
* **BLEU Score (4-gram):** 0.4441

The model successfully handles:
* Variable declaration types (`int` -> `int`).
* Method signatures (`public static void main` -> `static void Main`).
* Basic arithmetic operations and scoping.

## Future Improvements

* **Scale Up:** Increase `N` (layers) to 6 and `d_model` to 512 for production-grade performance.
* **Beam Search:** Implement Beam Search decoding for higher quality translations compared to Greedy decoding.
* **AST Parsing:** Integrate Abstract Syntax Tree (AST) information into embeddings for better syntactic correctness.
* **Model Hosting:** Deploy the model using ONNX Runtime or TorchServe to expose the transpiler via a REST API for real-time usage.

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

**Author:** Ansh Tyagi
Built using PyTorch
