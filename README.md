# Hindi-English Code-Mixed Sentiment Analysis System

This repository provides an end-to-end deep learning pipeline for sentiment classification (Positive, Negative, Neutral) on **Hinglish (Hindi-English)** code-mixed text using the academic **IIIT-Hyderabad Code-Mixed Benchmark Dataset (`IIITH_Codemixed.txt`)**. 

Standard monolingual NLP pipelines struggle with code-mixed social media datasets due to unpredictable phonetic variations (e.g., *zindagi* vs *jindagi*) and alternating syntactical grammar constraints. This project addresses those challenges by training **Byte-Pair Encoding (BPE) subword tokenizers** and **FastText character n-gram embeddings** completely from scratch, followed by a comparative performance evaluation of bidirectional **LSTM** and **GRU** recurrent architectures.

---

## 🚀 System Pipeline Architecture

The implementation is modularized into five progressive pipelines:

[Raw Dataset Ingestion]│▼[Phonetic Normalization & Cleaning] ──► Lowers elongation ("goood" -> "good") & unifies phonetics│▼[Subword Tokenizer & FastText] ──────► Compiles custom BPE lexicon & trains character n-grams│▼[Sequential Models Training] ────────► Train / Val / Test (70/15/15) split on Bi-LSTM and Bi-GRU│▼[Stratified Error Analysis] ─────────► Evaluates metrics across Code-Mixing Index (CMI) bands


---

## 🛠️ Requirements & Installation

Run the following command to set up the necessary dependencies:

```bash
pip install torch gensim pandas numpy tokenizers scikit-learn
```

---

## 📂 Key Source Code Implementations

### 1. Data Cleaning & Language Identification (LID)
Handles typical social media structural elements (URLs, mentions, and emoticons) while minimizing phonetic variations. It tags individual tokens to map native matrices:

```python
# Feature highlights from the core preprocessor
# Converts elongated strings and maps chaotic transliterations:
PHONETIC_REPLACEMENTS = { r'z': 'j', r'oo+': 'u', r'ee+': 'i', r'aa+': 'a' }
```

### 2. Custom Embeddings Training
Because pretrained English embeddings (like GloVe) or standard Hindi language models suffer from high out-of-vocabulary (OOV) error rates on code-mixed data, we train a customized subword embedding layer from scratch:

```python
# Custom Subword Setup using FastText N-Grams
ft_model = FastText(
    sentences=tokenized_sentences,
    vector_size=100, 
    window=5, 
    min_count=2, 
    sg=1,              # Skip-gram prioritizes context proximity
    min_n=3, max_n=6   # Captures structural transliterated roots
)
```

### 3. Model Architecture Specs
Both the **Bidirectional LSTM** and **Bidirectional GRU** models utilize:
* An initial `nn.Embedding` layer loaded with the custom FastText weights (`freeze=False` for fine-tuning).
* 2 structural sequential layers with a hidden dimension size of `128`.
* Global Average Pooling (`torch.mean`) to retain balanced representations across short social posts.
* A final Fully Connected layer mapping outputs to the 3 target classes (`0: Negative`, `1: Neutral`, `2: Positive`).

---

## 📊 Evaluation & Stratified Error Analysis

To evaluate how language mixing impacts performance, the final test step runs a stratified analysis using the **Code-Mixing Index (CMI)**.

\[\text{CMI Approximation} = 100 \times \left(1 - \frac{\max(\text{Lang}_{\text{Tokens}})}{\text{Total}_{\text{Lang Token Count}}}\right)\]

Sentences are dynamically evaluated across three categories:
1. **Mostly Monolingual (Low Mix):** CMI ≤ 10
2. **Moderately Mixed:** 10 < CMI ≤ 30
3. **Heavily Mixed (High Mix):** CMI > 30

### Execution Commands

To run the complete pipeline, execute your notebook scripts sequentially:
1. Load dataset partitions and apply formatting fixes.
2. Build vocabulary and initialize embedding weights.
3. Train `SentimentLSTM` and `SentimentGRU` networks.
4. Run the final validation block:
```python
# Executes the stratified error matrix logic
analysis_results = perform_error_analysis(model, gru_model, test_loader, dataset)
```

---

## 📈 Expected Repository Findings

* **Linguistic Robustness:** Both architectures typically reach their highest classification scores within the **Mostly Monolingual** band.
* **LSTM vs GRU Gating:** In the **Heavily Mixed** category, performance generally decreases. GRUs often achieve faster training times per epoch and show a slight edge in handling abrupt, mid-sentence language switches, as they don't maintain a separate cell state.
* **Common Mistakes:** Mutual failures often stem from sarcasm or sentiment inversion, where a sentence transitions from an English positive frame to a Romanized Hindi negative punchline (e.g., *"The display quality is quite brilliant but frame drops ne sara maza kharab kar diya"*).

---

## 🤝 Contributing
Contributions, issue reports, and alternative model suggestions (such as fine-tuned multilingual BERT or XLM-RoBERTa variants) are welcome! Feel free to open a pull request.
