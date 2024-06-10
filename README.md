# Machine-Learning-Projects

This repository contains a collection of Jupyter notebooks focused on various financial data analysis and machine learning projects. Each project is aimed at solving specific problems using advanced data science techniques.

## Projects Overview

### 1. Named Entity Recognition (NER)
**Objective:** Extract named entities from financial agreements.

**Methodology:** 
- **Data Preprocessing:** Loaded and preprocessed a dataset of financial agreements, labeling tokens with one of four NE types: location (LOC), miscellaneous (MISC), organization (ORG), and person (PER).
- **Model Training:** Utilized a Conditional Random Fields (CRF) tagger with an extensive feature set, including word-level features and POS tags. The dataset was split into training and testing sets using an 80-20 split.
- **Feature Selection:** Features included capitalization, presence of numbers, punctuation, suffixes, current word, previous and next words, and POS tags.
- **Evaluation:** Assessed performance using standard NER metrics.

**Strengths & Limitations:**
- CRFs are excellent for handling sequential data and incorporating various features but may overfit with high-dimensional feature spaces and struggle with long-range dependencies.

**Notebook:** [named_entity_recognition.ipynb](notebooks/named_entity_recognition.ipynb)

### 2. Bankruptcy Prediction
**Objective:** Predict the likelihood of bankruptcy using machine learning models.

**Methodology:**
- **Data Preparation:** Cleaned and preprocessed financial data, selecting relevant features for bankruptcy prediction.
- **Model Selection:** Evaluated multiple models, including Logistic Regression, Decision Trees, Random Forest, and SVM.
- **Feature Engineering:** Incorporated financial ratios, historical trends, and industry benchmarks.
- **Model Evaluation:** Used metrics like accuracy, precision, recall, and AUC to evaluate model performance.

**Results & Discussion:**
- Logistic Regression and Random Forest showed the best performance, with significant insights derived from feature importance analysis.

**Notebook:** [bankruptcy_prediction.ipynb](notebooks/bankruptcy_prediction.ipynb)

### 3. Financial News Sentiment Classification
**Objective:** Classify the sentiment of financial news articles to gauge market sentiment.

**Methodology:**
- **Data Collection:** Gathered a dataset of financial news articles with labeled sentiments.
- **Text Processing:** Tokenized text, removed stop words, and applied TF-IDF vectorization.
- **Model Training:** Implemented Naive Bayes and Logistic Regression models.
- **Evaluation:** Used metrics such as accuracy, precision, recall, and F1 score.

**Results & Discussion:**
- Logistic Regression outperformed Naive Bayes, particularly in handling imbalanced classes. Future improvements could include using transformer-based models like BERT, enhanced feature engineering, and cross-validation.

**Notebook:** [financial_news_sentiment_classification.ipynb](notebooks/financial_news_sentiment_classification.ipynb)
