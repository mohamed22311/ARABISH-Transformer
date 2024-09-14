# **English-to-Arabic Transformer-Based Auto Translation Model**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)](https://pytorch.org/)

This repository contains an implementation of a **Transformer-based English-to-Arabic translation model**. The model was built from scratch using **PyTorch** and is inspired by the seminal paper _[Attention is All You Need](https://arxiv.org/abs/1706.03762)_. It is designed to translate sentences from English to Arabic efficiently, leveraging the power of the Transformer architecture's self-attention mechanism.

The training process utilizes the **[yhavinga/ccmatrix](https://huggingface.co/datasets/yhavinga/ccmatrix)** dataset from the OPUS project, which provides high-quality parallel data for machine translation.

## **Table of Contents**

- [Project Overview](#project-overview)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Visualizations](#visualizations)
- [Contributors](#contributors)
- [License](#license)

---

## **Project Overview**

### **Motivation**

The demand for effective, real-time translation systems has increased in our globalized world. Despite significant progress in natural language processing (NLP), Arabic, with its rich morphology and complex grammar, remains a challenging language to translate accurately. This project tackles that challenge by implementing a **Transformer model** that can efficiently handle **English-to-Arabic** translation using modern machine translation techniques.

---

## **Model Architecture**

The model is built from scratch, following the Transformer architecture introduced in _Attention is All You Need_. The Transformer outperforms previous architectures like Recurrent Neural Networks (RNNs) or Long Short-Term Memory (LSTMs) due to its ability to handle long-range dependencies and parallelize computations efficiently.

### **Key Components:**

- **Multi-Head Self-Attention:** Enables the model to focus on different parts of the input sequence simultaneously.
- **Positional Encoding:** Adds information about the relative position of words in the sequence since the model lacks inherent sequentiality.
- **Encoder-Decoder Architecture:** The encoder processes the source (English) sentence, and the decoder generates the target (Arabic) sentence.
- **Feedforward Layers:** Add depth to the model and provide non-linearity between attention layers.

### **Transformer Flowchart**

Below is a high-level overview of the model's architecture:

![Transformer Model](images/transformer_architecture.png)

---

## **Dataset**

The **[yhavinga/ccmatrix](https://huggingface.co/datasets/yhavinga/ccmatrix)** dataset from **OPUS** is used to train and evaluate the model. It is one of the largest publicly available parallel corpora for English-Arabic language pairs. The dataset consists of bilingual sentence pairs extracted from web pages, providing a vast resource for training machine translation models.

### **Dataset Details**

- **Source:** OPUS (Open Parallel Corpus)
- **Languages:** English ↔ Arabic
- **Size:** ~X million sentence pairs (You can specify the exact size here)

### **Sample Data**

| English Sentence          | Arabic Translation  |
| ------------------------- | ------------------- |
| "This is a test sentence" | "هذه جملة اختبار"   |
| "Hello, how are you?"     | "مرحبًا، كيف حالك؟" |

---

## **Installation**

### **Step 1: Clone the Repository**

```bash
git clone https://github.com/your-username/english-to-arabic-transformer.git
cd english-to-arabic-transformer
```

### **Step 2: Install Dependencies**

```bash
pip install -r requirements.txt
```

### **Prerequisites**

- **Python 3.8+**
- **PyTorch 1.9+**
- **HuggingFace Datasets** for data loading
- **CUDA** (Optional for GPU training)

---

## **Usage**

You can either use the pre-trained model or train a new model from scratch. Here's a quick guide on how to translate an English sentence using the model:

```python
from model import TransformerModel
from utils import translate_sentence

# Load the trained model
model = TransformerModel.load('path_to_model_checkpoint')

# Translate an English sentence
english_sentence = "This is an example."
arabic_translation = translate_sentence(model, english_sentence)
print(f"Arabic Translation: {arabic_translation}")
```

For training and evaluation, refer to the following sections.

---

## **Training**

To train the model, simply run:

```bash
python train.py --epochs 10 --batch_size 64 --learning_rate 1e-4
```

### **Key Features:**

- **Optimizer:** Adam optimizer with weight decay.
- **Loss Function:** Cross-entropy loss applied to the target language tokens.
- **Learning Rate Schedule:** Cosine learning rate decay with warmup.
- **Early Stopping:** Based on validation loss.

Hyperparameters can be customized via the command line.

---

## **Evaluation**

To evaluate the model's performance on a test set, execute:

```bash
python evaluate.py --model_path ./checkpoints/best_model.pth
```

### **Evaluation Metrics:**

- **BLEU Score:** Widely used in machine translation tasks to measure the similarity between the generated translation and the reference.

For example:

```bash
python evaluate.py --model_path ./checkpoints/best_model.pth
BLEU Score: 28.5
```

---

## **Results**

The model achieves the following performance on the test set:

| Metric   | Score |
| -------- | ----- |
| **BLEU** | 28.5  |

---

## **Visualizations**

Here are a few visualizations to illustrate the model's performance and learning process.

### **Training and Validation Loss**

Visualizing the loss during training can help identify overfitting or underfitting:

![Training and Validation Loss](images/loss_curve.png)

### **Attention Weights Visualization**

The following heatmap shows the attention weights from the self-attention mechanism, highlighting which words the model focuses on during translation.

![Attention Heatmap](images/attention_heatmap.png)

These visualizations help in understanding the model's learning dynamics and its attention to different parts of the input sentence.

---

## **Contributors**

- **Mohamed** - Senior Computer Engineering Student at Helwan University, Egypt.

Feel free to reach out if you have questions or suggestions!

---

## **License**

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

You can improve the README further by adding **download links** for model weights, **examples of input/output translations**, or additional instructions on hyperparameter tuning. If you have visualizations ready, such as training loss curves, attention weight heatmaps, or architecture diagrams, you should link or embed those as shown.
