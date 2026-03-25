# Problem 2: Character-Level Name Generation

This project implements and compares three different RNN architectures for generating plausible Indian names.

## 🧠 Model Architectures

All models are built from scratch using PyTorch tensor operations (no `nn.RNN` or `nn.LSTM` helpers used for core cells).

1. **Vanilla RNN**: Simple recursive structure using `tanh` activation. Best at memorizing existing names.
2. **BiLSTM**: Bidirectional Long Short-Term Memory. Captures future context but tends toward random generation in autoregressive mode.
3. **RNN with Attention**: Incorporates an attention mechanism to weigh previous characters, allowing for better "memory" of phonetic patterns.

## 📂 Project Structure

- `models.py`: Contains the character vocabulary (`CVocab`) and the three model class implementations.
- `train.py`: Handles the training loop, gradient clipping, and checkpoint saving.
- `generate.py`: Sample names from trained models at different **Temperatures** (0.5, 0.8, 1.0).
- `evaluate.py`: Computes quantitative metrics:
  - **Novelty Rate**: % of names not found in the training set.
  - **Diversity**: Ratio of unique names to total generated names.

## 🚀 Usage

1. **Train Models**:
   ```bash
   python train.py
   ```

2. **Generate Samples**:
   ```bash
   python generate.py
   ```

3. **Evaluate Results**:
   ```bash
   python evaluate.py
   ```

## 📈 Summary of Findings
- **High Realism**: Vanilla RNN (T=0.5) generates very safe, existing names.
- **High Creativity**: RNN + Attention (T=0.8) generates novel, plausible names.
- **Metrics**: Detailed comparisons are stored in `outputs/evaluation_results.json` and visualized in `outputs/model_comparison.png`.
