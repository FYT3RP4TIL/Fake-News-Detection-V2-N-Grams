# FakeNews-Detector

## 📰 Project Overview

FakeNews-Detector is a machine learning project aimed at classifying news articles as either real or fake using various natural language processing (NLP) techniques and classification algorithms. The project demonstrates the effectiveness of bag-of-n-grams approaches in text classification tasks.

## 🎯 Objective

The main goal of this project is to address the problem of misinformation by developing a model that can accurately distinguish between real and fake news articles.

## 📊 Dataset

The dataset used in this project is from Kaggle: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

- **Structure**: Two columns - 'Text' and 'label'
- **Text**: Contains the news article or statement
- **Label**: Indicates whether the text is Fake (0) or Real (1)
- **Task Type**: Binary Classification

## 🛠 Methodology

### 1. Data Preprocessing

Two approaches were used:
a) Without preprocessing
b) With preprocessing

Preprocessing steps included:
- Removing stop words
- Removing punctuation
- Applying lemmatization

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

### 2. Feature Extraction

Bag-of-n-grams approach using scikit-learn's CountVectorizer:
- Unigrams
- Bigrams
- Trigrams

### 3. Model Training and Evaluation

Several classification algorithms were implemented and compared:

1. K-Nearest Neighbors (KNN)
   - With Euclidean distance
   - With Cosine similarity
2. Random Forest
3. Multinomial Naive Bayes

## 📈 Results

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

## 🔍 Analysis

1. **KNN Performance**: 
   - Cosine similarity outperformed Euclidean distance
   - Overall performance was lower compared to other algorithms

2. **Random Forest**:
   - Showed excellent performance both with and without preprocessing
   - Achieved near-perfect accuracy (0.99-1.00) in all configurations

3. **Multinomial Naive Bayes**:
   - Performed very well without preprocessing
   - Achieved 0.99 accuracy

4. **Effect of Preprocessing**:
   - Slightly reduced performance for Random Forest with trigrams (0.96 vs 0.99)
   - Maintained perfect performance for Random Forest with 1-3 grams

5. **Best Model**:
   - Random Forest with 1-3 grams and preprocessing achieved perfect scores across all metrics

## 🚀 Conclusion

The FakeNews-Detector project demonstrates the effectiveness of bag-of-n-grams approaches in classifying fake news. Random Forest consistently performed the best, achieving near-perfect or perfect accuracy. The project also highlights that while preprocessing can be beneficial, careful consideration should be given to its impact on model performance.

## 🛠 Tools and Libraries Used

- Python
- pandas: For data manipulation
- scikit-learn: For machine learning models and evaluation
- spaCy: For text preprocessing
- NumPy: For numerical operations

## 🔮 Future Work

1. Experiment with more advanced NLP techniques (e.g., word embeddings, transformers)
2. Incorporate additional features (e.g., source credibility, publication date)
3. Develop a web interface for real-time fake news detection
4. Explore ensemble methods to combine the strengths of different models

## 📚 References

1. Kaggle Dataset: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
2. scikit-learn Documentation: [https://scikit-learn.org/](https://scikit-learn.org/)
3. spaCy Documentation: [https://spacy.io/](https://spacy.io/)
