# Sentiment Analysis with TF-IDF and Machine Learning

## 📌 Project Overview

This project implements an **end-to-end sentiment analysis pipeline** using classical Natural Language Processing (NLP) and Machine Learning techniques. The goal is to automatically classify text data (tweets or customer reviews) into sentiment categories (positive / negative).

The workflow covers the full data science lifecycle: **data cleaning, exploratory data analysis (EDA), feature extraction with TF-IDF, model training, evaluation, performance improvement, and model persistence**.

---

## 🎯 Objectives

* Clean and preprocess raw text data
* Explore sentiment distribution and textual characteristics
* Convert text into numerical features using TF-IDF
* Train and evaluate machine learning models for sentiment classification
* Improve model performance using pipelines and hyperparameter tuning
* Save and reuse the trained model

---

## 🧠 Dataset

The project is designed to work with **Kaggle sentiment analysis datasets**, such as:

[

import kagglehub

path = kagglehub.dataset_download("vishakhdapat/imdb-movie-reviews")

print("Path to dataset files:", path)

]

Each dataset typically contains:

* A text column (e.g. `text`)
* A sentiment label column (e.g. `y` or `sentiment`)

---

## 🧹 Data Preprocessing

Text preprocessing includes:

* Lowercasing
* Removing URLs, mentions, and hashtags
* Removing or converting emojis
* Removing punctuation and numbers
* Stopword removal
* Lemmatization

The cleaning logic is integrated into a **scikit-learn Pipeline** to ensure consistency and prevent data leakage.

---

## 📊 Exploratory Data Analysis (EDA)

EDA is performed to better understand the dataset:

* Sentiment class distribution
* Text length analysis
* Frequent words by sentiment

Key insight:

> Text length is similar across sentiment classes, indicating that sentiment prediction depends mainly on lexical content rather than text size.

---

## 🔢 Feature Engineering

Text is transformed into numerical vectors using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

TF-IDF configuration includes:

* Unigrams and bigrams
* Frequency thresholds to remove noise
* Sublinear term frequency scaling

This representation highlights sentiment-bearing words while reducing the impact of common terms.

---

## 🤖 Modeling

Several machine learning models can be applied. The main baseline models include:

* Logistic Regression
* Linear Support Vector Machine (LinearSVC)

A **Pipeline** is used to chain:

1. Text cleaning
2. TF-IDF vectorization
3. Classification model

This ensures a clean, reproducible, and production-ready workflow.

---

## ⚙️ Model Improvement

Model performance is improved using:

* Hyperparameter tuning with `GridSearchCV`
* Cross-validation
* Class weighting for imbalanced datasets

The evaluation focuses on:

* Precision
* Recall
* F1-score (preferred over accuracy for imbalanced data)

---

## 💾 Model Saving

The trained pipeline is saved using `joblib`, which stores:

* Preprocessing steps
* TF-IDF vectorizer
* Trained classifier

This allows easy reuse of the model for inference or deployment without retraining.

---

## 🛠️ Technologies Used

* Python
* Pandas, NumPy
* NLTK
* scikit-learn
* Matplotlib / Seaborn

---

## 🚀 Future Improvements

* Use character n-grams for noisy text
* Integrate emoji sentiment instead of removing emojis
* Experiment with word embeddings (Word2Vec, GloVe)
* Apply transformer-based models (BERT, RoBERTa)
* Deploy the model using FastAPI or Streamlit

---

## 📂 Project Structure

```
├── Sentiment_Analyse.ipynb
├── sentiment_analysis_pipeline.joblib
├── README.md
```

---

## ✅ Conclusion

This project demonstrates a complete and professional approach to sentiment analysis using traditional NLP and machine learning techniques. The use of pipelines ensures robustness, reproducibility, and readiness for real-world applications.
