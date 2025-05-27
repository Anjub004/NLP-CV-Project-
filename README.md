# NLP-CV-Project
A professional deep learning project solving exam-based NLP and Computer Vision tasks. Includes tweet sentiment classification using SVM and Word2Vec, transfer learning for house image classification, OpenCV image processing, and visual insights.

- **DL-AI-Exam-Project/**
  - **NLP/**
    - **notebooks/**
      - Sentiment_Analysis_Tweets.ipynb
    - **datasets/**
      - tweets_train.csv  
      - tweets_test.csv  
    - **models/**
  - **CV/**
    - **notebooks/**
      - House_Image_Classification.ipynb
    - **datasets/**
      - **House Dataset/**
        - Training/  
        - Testing/  
    - **random_images/** (5+ images for OpenCV tasks)  
    - **outputs/**
  - README.md  
  - requirements.txt  


# DL + AI Exam Project 🔍📊🧠

This project contains solutions to a comprehensive exam-based assignment focused on Deep Learning and Artificial Intelligence using NLP and Computer Vision. It demonstrates professional-grade implementations with clean visualizations, pretrained models, and structured pipelines.

---

## 📌 Project Sections

### 🧠 1. Natural Language Processing (NLP)

**Task:** Perform sentiment classification on a dataset of tweets related to *The Social Dilemma* documentary.

#### ✅ Implemented Steps:
- Data exploration and value distribution
- Preprocessing (contractions, tokenization, normalization)
- Bag-of-Words (BoW) vectorization + Linear SVM
- Word2Vec embeddings + SVM (with and without TF-IDF weighting)
- Evaluation using classification reports
- Polarity analysis using TextBlob
- Exploratory insights:
  - Temporal sentiment trends
  - Word clouds and top hashtag visualizations
  - Sentiment by location, verification status, and source
  - Co-occurring hashtag network using NetworkX

📁 Files:
- `NLP/notebooks/Sentiment_Analysis_Tweets.ipynb`
- `NLP/datasets/tweets_train.csv`, `tweets_test.csv`

---

### 🏠 2. Computer Vision (CV)

**Task:** Build a classifier using pretrained models to distinguish between "House" and "Not House" images. Additional OpenCV-based processing was also performed.

#### ✅ Implemented Steps:
- Data cleaning (removal of corrupted images using PIL)
- Transfer learning with `MobileNetV2` using TensorFlow/Keras
- Train/test accuracy plotting
- Image histogram visualizations (RGB channels)
- OpenCV transformations:
  - Rotate, flip, blur, resize, shift
- RGB channel separation for video frame processing

📁 Files:
- `CV/notebooks/House_Image_Classification.ipynb`
- `CV/datasets/House Dataset/` (with `Training/` and `Testing/`)
- `CV/datasets/Random_Images/` (for OpenCV tasks)

---

## 📂 Folder Guide

| Folder         | Description                                  |
|----------------|----------------------------------------------|
| `NLP/`         | Contains tweet sentiment analysis pipeline   |
| `CV/`          | Contains image classification + OpenCV tasks |
| `datasets/`    | Original and processed datasets              |
| `notebooks/`   | All executed Jupyter notebooks               |
| `outputs/`     | Optional folder for exported images/models   |

---

## 🔧 Requirements

To install dependencies:

```bash
pip install -r requirements.txt
```
---
Recommended libraries:
- numpy, pandas, matplotlib, seaborn
- scikit-learn, nltk, gensim, textblob, tensorflow
- opencv-python, networkx, wordcloud

---
## 📈 Result Highlights
- BoW + SVM Accuracy: 88%
- Word2Vec + SVM (TF-IDF weighted): ~73%
- CV Classifier Accuracy (MobileNetV2): ~90%

All transformations implemented using OpenCV in a clean, side-by-side visual format
---

## 📬 Contact
For questions or contributions, feel free to connect.
---

Project by: [Anju Barai]

---

Let me know if you’d like:
- A custom **`requirements.txt`** file  
- Git commit messages for upload  
- A professional **project thumbnail/banner** for GitHub

Just say the word!
