# Assignment 2 Report

## Problem 1: Learning Word Embeddings from IIT Jodhpur Data

### Task 1: Dataset Preparation
Data sources were collected from IIT Jodhpur web pages using a tree-based crawler and preprocessed with boilerplate removal, tokenization, lowercasing, and punctuation/non-text filtering.

- Total documents: 80
- Total sentences: 787
- Total tokens: 26,409
- Vocabulary size: 2,021
- Average sentence length: 33.56

Top frequent words include: of, the, and, for, to, jodhpur, engineering, institute, research.

Generated artifacts:
- Cleaned corpus: Problem1/corpus.txt
- Corpus statistics: Problem1/outputs/corpus_statistics.json
- Word cloud: Problem1/outputs/wordcloud.png

### Task 2: Model Training
Two model families were trained from scratch and with gensim:
- CBOW
- Skip-gram with Negative Sampling

Hyperparameters explored:
- Embedding dimensions: 50, 100, 200
- Window sizes: 2, 3, 5
- Negative samples (Skip-gram): 5, 10, 15

Summary from scratch experiments:
- Best CBOW final loss observed at dim=200, window=3 (final loss ~0.3883)
- Best Skip-gram final loss observed at dim=100, window=2, neg=5 (final loss ~1.1379)

Artifacts:
- Scratch experiments: Problem1/outputs/experiment_results.json
- Model curves: Problem1/outputs/training_curves.png
- Saved embeddings/models: Problem1/outputs/models/
- Gensim comparison: Problem1/outputs/gensim_analysis.json

### Task 3: Semantic Analysis
Nearest-neighbor and analogy evaluation was run for both scratch and gensim models.

Queried words:
- research
- student
- phd
- exam

Notes:
- research, student, and phd had meaningful nearest-neighbor outputs across models.
- exam was not found in the final vocabulary due corpus filtering/token distribution.
- Analogy results are partially meaningful; several analogy prompts had missing tokens (for example btech, exam, learn).

Artifact:
- Semantic results: Problem1/outputs/semantic_analysis.json

### Task 4: Visualization
PCA and t-SNE visualizations were generated for semantic groups (academic programs, research, people, departments, assessment), including CBOW vs Skip-gram comparisons.

Artifacts:
- PCA CBOW: Problem1/outputs/pca_cbow_dim100_win3.png
- PCA Skip-gram: Problem1/outputs/pca_sg_dim100_win3_neg5.png
- t-SNE CBOW: Problem1/outputs/tsne_cbow_dim100_win3.png
- t-SNE Skip-gram: Problem1/outputs/tsne_sg_dim100_win3_neg5.png

Interpretation:
- Skip-gram shows slightly tighter separation for some category words in PCA.
- t-SNE reveals non-linear neighborhood structure but with overlap across broad academic categories, expected for mixed institutional text.

---

## Problem 2: Character-Level Name Generation Using RNN Variants

### Task 0: Dataset
Training names were loaded from TrainingNames.txt.

### Task 1: Model Implementation and Training
Implemented models:
- Vanilla RNN
- Bidirectional LSTM (BLSTM)
- RNN with basic attention mechanism

Hyperparameters:
- Embed dimension: 32
- Hidden size: 128
- Layers: 1
- Learning rate: 0.003
- Epochs: 100
- Batch size: 64

Trainable parameters:
- Vanilla RNN: 25,244
- BLSTM: 172,956
- RNN+Attention: 74,524

Training artifacts:
- Checkpoints: Problem2/outputs/vanilla_rnn_best.pth, Problem2/outputs/blstm_best.pth, Problem2/outputs/rnn_attention_best.pth
- Training summary: Problem2/outputs/training_info.json
- Loss plot: Problem2/outputs/training_loss.png

### Task 2: Quantitative Evaluation
Metrics used:
- Novelty Rate = generated names not in training set / total generated
- Diversity = unique generated names / total generated

Main comparison at temperature 0.8:
- Vanilla RNN: Novelty 12.00%, Diversity 0.8600
- BLSTM: Novelty 100.00%, Diversity 1.0000
- RNN+Attention: Novelty 91.50%, Diversity 0.9150

Artifact:
- Evaluation metrics: Problem2/outputs/evaluation_results.json
- Comparison plots: Problem2/outputs/model_comparison.png and Problem2/outputs/temperature_effect.png

### Task 3: Qualitative Analysis
Representative behavior:
- Vanilla RNN generates realistic names more frequently, especially at low/moderate temperature.
- BLSTM produces highly novel and diverse outputs but many are less realistic (over-fragmented character combinations).
- RNN+Attention balances reuse and novelty, but often repeats long morpheme-like chunks.

Common failure modes:
- Over-generation of long strings
- Repetitive syllables
- Drift to non-name character patterns at higher temperature

Generated sample files:
- Problem2/outputs/generated_vanilla_rnn_temp0.5.txt
- Problem2/outputs/generated_vanilla_rnn_temp0.8.txt
- Problem2/outputs/generated_vanilla_rnn_temp1.0.txt
- Problem2/outputs/generated_blstm_temp0.5.txt
- Problem2/outputs/generated_blstm_temp0.8.txt
- Problem2/outputs/generated_blstm_temp1.0.txt
- Problem2/outputs/generated_rnn_attention_temp0.5.txt
- Problem2/outputs/generated_rnn_attention_temp0.8.txt
- Problem2/outputs/generated_rnn_attention_temp1.0.txt

---

## Deliverables Checklist
- Source code: completed in Problem1 and Problem2 folders
- Cleaned corpus: Problem1/corpus.txt
- Visualizations: completed in Problem1/outputs and Problem2/outputs
- Evaluation scripts and outputs: completed in Problem2/outputs
- Report: this file
