# FakeNews-Detector

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Objective](#-objective)
3. [Dataset](#-dataset)
4. [Setup and Environment](#-setup-and-environment)
5. [Jupyter Notebook Structure](#-jupyter-notebook-structure)
6. [Methodology](#-methodology)
   - [Data Preprocessing](#data-preprocessing)
   - [Feature Extraction](#feature-extraction)
   - [Model Selection](#model-selection)
7. [Experiments and Results](#-experiments-and-results)
   - [Models without Preprocessing](#models-without-preprocessing)
   - [Models with Preprocessing](#models-with-preprocessing)
8. [Best Model](#-best-model)
9. [Analysis](#-analysis)
10. [Conclusion](#-conclusion)
11. [Future Work](#-future-work)
12. [References](#-references)

## 📰 Project Overview

FakeNews-Detector is a Jupyter notebook-based machine learning project designed to tackle the pervasive issue of misinformation in digital media. By leveraging state-of-the-art natural language processing (NLP) techniques and various classification algorithms, this project aims to automatically distinguish between genuine and fabricated news articles with high accuracy.

## 🎯 Objective

The primary objectives of this project are:
1. To develop a robust model capable of accurately classifying news articles as either real or fake.
2. To compare the effectiveness of various machine learning algorithms in the context of fake news detection.
3. To evaluate the impact of text preprocessing on model performance.
4. To contribute to the ongoing efforts in combating misinformation and promoting media literacy.

## 📊 Dataset

The project utilizes the "Fake and Real News Dataset" from Kaggle:

- **Source**: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Structure**: 
  - Two columns: 'Text' (news content) and 'label' (0 for Fake, 1 for Real)
- **Task Type**: Binary Classification

## 🛠 Setup and Environment

To run the FakeNews-Detector Jupyter notebook:

1. Ensure you have Jupyter Notebook or JupyterLab installed.
2. Install required libraries:
   ```
   pip install pandas scikit-learn spacy numpy
   ```
3. Download the spaCy English language model:
   ```
   python -m spacy download en_core_web_sm
   ```
4. Download the dataset from Kaggle and place it in the same directory as the notebook.

## 📓 Jupyter Notebook Structure

The project is contained in a single Jupyter notebook, structured as follows:

1. **Introduction and Setup**
   - Project overview
   - Library imports
   - Data loading

2. **Data Preprocessing**
   - Text cleaning function
   - Application of preprocessing

3. **Feature Extraction**
   - Bag-of-N-grams implementation

4. **Model Training and Evaluation**
   - KNN (Euclidean and Cosine)
   - Random Forest
   - Multinomial Naive Bayes

5. **Results and Analysis**
   - Performance comparison
   - Best model identification

6. **Conclusion and Future Work**

## 🛠 Methodology

### Data Preprocessing

Two approaches were implemented:

1. **Without Preprocessing**: Raw text data used directly.
2. **With Preprocessing**: 
   - Removal of stop words
   - Elimination of punctuation
   - Lemmatization of words

Preprocessing function:

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)
    return " ".join(filtered_tokens)
```

### Feature Extraction

The Bag-of-N-Grams approach was employed using scikit-learn's CountVectorizer.

### Model Selection

Several classification algorithms were implemented and compared:

1. K-Nearest Neighbors (KNN)
   - Euclidean distance metric
   - Cosine similarity metric
2. Random Forest
3. Multinomial Naive Bayes

## 📈 Experiments and Results

### Models without Preprocessing

#### 1. KNN (Euclidean Distance)
```
              precision    recall  f1-score   support

           0       0.96      0.49      0.65      1000
           1       0.65      0.98      0.78       980

    accuracy                           0.73      1980
   macro avg       0.81      0.74      0.72      1980
weighted avg       0.81      0.73      0.72      1980
```

#### 2. KNN (Cosine Similarity)
```
              precision    recall  f1-score   support

           0       0.99      0.55      0.71      1000
           1       0.69      1.00      0.81       980

    accuracy                           0.77      1980
   macro avg       0.84      0.77      0.76      1980
weighted avg       0.84      0.77      0.76      1980
```

#### 3. Random Forest
```
              precision    recall  f1-score   support

           0       1.00      0.99      0.99      1000
           1       0.99      1.00      0.99       980

    accuracy                           0.99      1980
   macro avg       0.99      0.99      0.99      1980
weighted avg       0.99      0.99      0.99      1980
```

#### 4. Multinomial Naive Bayes
```
              precision    recall  f1-score   support

           0       0.99      0.99      0.99      1000
           1       0.99      0.98      0.99       980

    accuracy                           0.99      1980
   macro avg       0.99      0.99      0.99      1980
weighted avg       0.99      0.99      0.99      1980
```

### Models with Preprocessing

#### 1. Random Forest (Trigrams)
```
              precision    recall  f1-score   support

           0       0.93      0.98      0.96      1000
           1       0.98      0.93      0.95       980

    accuracy                           0.96      1980
   macro avg       0.96      0.95      0.95      1980
weighted avg       0.96      0.96      0.96      1980
```

#### 2. Random Forest (1-3 grams)
```
              precision    recall  f1-score   support

           0       0.99      1.00      1.00      1000
           1       1.00      0.99      1.00       980

    accuracy                           1.00      1980
   macro avg       1.00      1.00      1.00      1980
weighted avg       1.00      1.00      1.00      1980
```

## 🏆 Best Model

The best-performing model in our experiments was the **Random Forest classifier with 1-3 grams and preprocessing**. This model achieved perfect or near-perfect scores across all metrics:

- **Accuracy**: 1.00 (100%)
- **Precision**: 0.99 (Fake), 1.00 (Real)
- **Recall**: 1.00 (Fake), 0.99 (Real)
- **F1-score**: 1.00 (Fake), 1.00 (Real)

Implementation details:
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ('vectorizer_n_grams', CountVectorizer(ngram_range=(1, 3))),
    ('random_forest', RandomForestClassifier())
])

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

This model's exceptional performance can be attributed to:
1. The use of a wide range of n-grams (1-3), capturing both individual words and short phrases
2. The Random Forest algorithm's ability to handle high-dimensional data and capture complex relationships
3. Effective preprocessing, which removed noise and standardized the text data

## 🔍 Analysis

1. **KNN Performance**: 
   - Cosine similarity metric outperformed Euclidean distance
   - Overall performance was lower compared to other algorithms, possibly due to the high-dimensional nature of text data

2. **Random Forest**:
   - Demonstrated excellent performance across all configurations
   - Showed robustness to different preprocessing approaches
   - The combination of multiple decision trees likely contributed to its ability to capture complex patterns in the text data

3. **Multinomial Naive Bayes**:
   - Performed very well without preprocessing
   - Its strong performance aligns with its reputation as an effective algorithm for text classification tasks

4. **Effect of Preprocessing**:
   - Generally maintained or slightly improved model performance
   - The slight performance reduction in some cases (e.g., Random Forest with trigrams) suggests that some informative features might have been lost during preprocessing

5. **N-gram Impact**:
   - The use of 1-3 grams consistently outperformed models using only trigrams, indicating the importance of capturing both individual words and short phrases

## 🚀 Conclusion

The FakeNews-Detector project successfully demonstrates the effectiveness of machine learning techniques in distinguishing between real and fake news articles. Key findings include:

1. Random Forest emerged as the most effective algorithm for this task, achieving near-perfect accuracy.
2. The combination of preprocessing and a wide range of n-grams (1-3) yielded the best results.
3. While preprocessing generally improved performance, its impact varied across different models and configurations.
4. The high accuracy achieved by multiple models suggests that lexical features (captured by bag-of-n-grams) are strongly indicative of fake news in this dataset.

## 🔮 Future Work

1. Experiment with more advanced NLP techniques:
   - Word embeddings (e.g., Word2Vec, GloVe)
   - Transformer-based models (e.g., BERT, RoBERTa)
2. Incorporate additional features:
   - Source credibility scores
   - Publication date and time
   - Author information
3. Develop a web-based interface for real-time fake news detection
4. Explore ensemble methods to combine the strengths of different models
5. Investigate the model's performance on different types of fake news (e.g., satire, propaganda)
6. Conduct error analysis to understand the types of articles that are misclassified


## 📚 References

1. Kaggle Dataset: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
2. scikit-learn Documentation: [https://scikit-learn.org/](https://scikit-learn.org/)
3. spaCy Documentation: [https://spacy.io/](https://spacy.io/)
4. Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017). Fake News Detection on Social Media: A Data Mining Perspective. ACM SIGKDD Explorations Newsletter, 19(1), 22-36.
5. Allcott, H., & Gentzkow, M. (2017). Social Media and Fake News in the 2016 Election. Journal of Economic Perspectives, 31(2), 211-236.
