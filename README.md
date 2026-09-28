# CS5720 – Homework 2: RNNs, LSTMs, and CNNs in PyTorch

**Author:** Baraka Mandi

This repository contains my solutions to Homework 2 for CS5720 (Neural Networks and Deep Learning). Each question is a self-contained Jupyter notebook implemented in **PyTorch**, covering recurrent networks for text (character-level generation and sentiment classification) and convolutional networks (convolution mechanics, feature extraction, pooling, and classic architectures).

---

## Repository Structure

| File | Description |
|------|-------------|
| `Qn1_char_lstm_text_generation.ipynb` | Q1 – Character-level LSTM trained on Tiny Shakespeare to generate text, with a study of temperature scaling |
| `Qn2_imdb_lstm_sentiment.ipynb` | Q2 – Bidirectional LSTM sentiment classifier on the IMDB movie review dataset |
| `Qn3_Convolution_Stride_Padding.ipynb` | Q3 – 2D convolution with different stride and padding settings, implemented from scratch in NumPy and verified against PyTorch |
| `Qn4_CNN_Feature_Extraction.ipynb` | Q4 – Sobel edge detection (NumPy + OpenCV) and max/average pooling (PyTorch) |
| `Qn5_CNN_Architectures_PyTorch.ipynb` | Q5 – Simplified AlexNet, a residual block, and a small ResNet-like model |
| `shakespeare.txt` | Tiny Shakespeare dataset (~1.1 MB) used in Q1 |
| `char_lstm_best.pt` | Saved best checkpoint of the Q1 character LSTM (~14 MB) |
| `bently.jpg` | Sample image used for edge detection in Q4 |

---

## Setup

Python 3.9+ is recommended.

```bash
git clone https://github.com/baraka-mandi/CS5720_HW2.git
cd CS5720_HW2

pip install torch numpy matplotlib seaborn scikit-learn opencv-python torchinfo jupyter
jupyter notebook
```

Every notebook automatically picks the best available device: CUDA GPU, Apple Silicon (MPS, in Q1), or CPU. Datasets are downloaded automatically the first time a notebook runs if they aren't already present.

> **Note:** Q2 trains on 25,000 reviews and runs much faster on a GPU. I trained it in Google Colab on a **T4 GPU** (Runtime → Change runtime type → T4 GPU).

---

## Q1 – Character-Level Text Generation with an LSTM

Trains a character-level language model on the Tiny Shakespeare corpus and uses it to generate new "Shakespearean" text.

- **Data:** 1,115,394 characters, 65-character vocabulary, 90/10 train/validation split. Each sample is a 100-character sequence whose target is the same sequence shifted by one character.
- **Model:** Embedding (128) → 2-layer LSTM (512 hidden units, dropout 0.3) → Linear to vocabulary. ~3.46M trainable parameters. A `USE_ONEHOT` flag switches to one-hot inputs instead of learned embeddings.
- **Training:** Adam (lr 2e-3), cross-entropy loss, gradient clipping at 5.0, `ReduceLROnPlateau` scheduler, 20 epochs. The best model by validation loss is saved to `char_lstm_best.pt`.
- **Generation:** The hidden state is warmed up on a prompt (e.g. `"ROMEO:"`), then characters are sampled one at a time from a temperature-scaled softmax (optional top-k).

**Result:** Best validation loss of **1.474** (epoch 11). Samples progress from near-gibberish in epoch 1 to text with realistic character names, line breaks, and play-like structure.

**Temperature analysis:** The notebook compares samples at T = 0.2, 0.5, 0.8, 1.0, and 1.5 and shows how temperature changes the entropy of the output distribution. Low temperatures give safe but repetitive text; high temperatures give more varied text with more spelling errors and broken structure.

---

## Q2 – Sentiment Classification on IMDB with an LSTM

Classifies movie reviews as positive or negative using the raw Stanford IMDB dataset (50,000 labeled reviews).

- **Preprocessing:** HTML tags removed, lowercased, regex tokenization (keeps contractions such as *don't*). A 20,000-word vocabulary is built from training data only to avoid test leakage. Reviews are padded/truncated to 300 tokens, keeping the *last* 300 tokens because reviews often end with the verdict. 10% of the training set is held out for validation.
- **Model:** Embedding (128) → 2-layer **bidirectional** LSTM (128 hidden units, dropout 0.4) → Linear → single logit. Packed sequences let the LSTM skip padding tokens. ~3.22M trainable parameters.
- **Training:** Adam (lr 1e-3), `BCEWithLogitsLoss`, gradient clipping, `ReduceLROnPlateau`, and early stopping (patience 2), up to 8 epochs.

**Test results (25,000 reviews):**

| Metric | Negative | Positive |
|--------|----------|----------|
| Precision | 0.880 | 0.887 |
| Recall | 0.888 | 0.879 |
| F1-score | 0.884 | 0.883 |

**Overall test accuracy: 88.34%**

The notebook also includes a confusion matrix, a precision–recall curve, a table showing how precision, recall, and F1 shift as the decision threshold moves from 0.2 to 0.8, a helper to predict sentiment on custom reviews, and a written discussion of why the precision–recall tradeoff matters in real sentiment applications.

---

## Q3 – Convolution with Different Stride and Padding

Convolves a 5×5 input matrix (values 1–25) with a 3×3 Laplacian kernel under four settings:

| Stride | Padding | Output shape |
|--------|---------|--------------|
| 1 | VALID | 3 × 3 |
| 1 | SAME | 5 × 5 |
| 2 | VALID | 2 × 2 |
| 2 | SAME | 3 × 3 |

Convolution is implemented twice: once **from scratch in NumPy** and once with `torch.nn.functional.conv2d`. A shared helper computes TensorFlow-style VALID/SAME padding, with explicit padding in PyTorch because its built-in `padding='same'` does not support stride > 1. Both implementations produce identical feature maps in every case.

---

## Q4 – CNN Feature Extraction: Edge Detection and Pooling

**Task 1 – Sobel edge detection:** Loads a grayscale image (`bently.jpg`) and applies Sobel-X and Sobel-Y kernels with `cv2.filter2D`. Sobel-X highlights vertical edges and Sobel-Y highlights horizontal edges, displayed side by side with the original. If the local image is missing, the notebook downloads a sample image, and if that fails, it generates a synthetic image of geometric shapes so it always runs.

**Task 2 – Pooling:** Creates a random 4×4 matrix and applies 2×2 max pooling and 2×2 average pooling (stride 2) using `nn.MaxPool2d` and `nn.AvgPool2d`, printing the original and pooled matrices.

---

## Q5 – Implementing CNN Architectures

The assignment specified TensorFlow/Keras, but both architectures here are implemented in **PyTorch**, with model summaries from `torchinfo`.

**Task 1 – Simplified AlexNet** (input 3×227×227, 10 classes)
- 5 convolutional layers (96 → 256 → 384 → 384 → 256 filters) with ReLU and three 3×3/stride-2 max-pooling layers
- Classifier: Flatten → Dense 4096 → Dropout 0.5 → Dense 4096 → Dropout 0.5 → Dense 10 → Softmax
- **58,322,314 parameters**

**Task 2 – Residual Block and ResNet-like Model** (input 3×32×32, 10 classes)
- `ResidualBlock`: two 3×3 convolutions with an identity skip connection added before the final ReLU (a 1×1 projection shortcut is used if the channel counts differ). A functional `residual_block(input_tensor, filters)` helper matching the assignment's signature is also provided.
- `ResNetLike`: 7×7/stride-2 stem convolution (64 filters) → 2 residual blocks → Flatten → Dense 128 → Dense 10 → Softmax
- **2,255,754 parameters**

Sanity checks confirm the output shapes and that softmax rows sum to 1.

---

## Notes

- Random seeds are fixed (`42`) for reproducibility, though GPU nondeterminism can cause small differences between runs.
- Q2 saves its best model to `best_lstm_imdb.pt` at runtime; this file is not committed to the repo. The IMDB dataset (~84 MB) is also downloaded at runtime rather than stored here.
- To reuse the Q1 checkpoint without retraining, load `char_lstm_best.pt`. It stores the model weights, the character mappings (`stoi`/`itos`), and the model config.
