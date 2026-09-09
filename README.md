# Intelligent Document Understanding System

An NLP-based document classification and information extraction system that analyzes unstructured text, identifies document categories, extracts useful information, and compares Machine Learning and Deep Learning approaches.

## 📌 Project Overview

Organizations deal with large amounts of unstructured textual information such as business documents, reports, articles, and other digital content. Manually reviewing and categorizing these documents can be time-consuming.

This project demonstrates an **Intelligent Document Understanding System** that uses Natural Language Processing (NLP), Machine Learning, and Deep Learning techniques to automatically understand and classify text documents.

The system performs:

* Text cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Text feature engineering
* TF-IDF feature extraction
* Document classification
* Deep Learning-based text classification
* Information extraction
* Model comparison and evaluation
* Saving and loading the best-performing model

---

## 🎯 Objectives

The main objectives of this project are to:

1. Process and clean unstructured document text.
2. Explore document characteristics and category distributions.
3. Extract meaningful numerical and text-based features.
4. Build Machine Learning models for document classification.
5. Build a Deep Learning model using BiLSTM.
6. Compare model performance using Accuracy and Weighted F1-score.
7. Identify the best-performing classification model.
8. Extract useful information such as emails, URLs, phone numbers, and dates.
9. Save the trained model and preprocessing components for future predictions.

---

## 📊 Dataset

The project uses the **Text Classification Documentation** dataset available on Kaggle.

**Dataset source:** Tanishq Dublish — Text Classification Documentation

The dataset contains:

* **2,225 documents**
* **2 columns**

  * `Text` — document text
  * `Label` — document category

### Document Categories

| Label | Category      |
| ----: | ------------- |
|     0 | Politics      |
|     1 | Sport         |
|     2 | Technology    |
|     3 | Entertainment |
|     4 | Business      |

### Dataset Distribution

| Category      | Documents |
| ------------- | --------: |
| Sport         |       511 |
| Business      |       510 |
| Politics      |       417 |
| Technology    |       401 |
| Entertainment |       386 |
| **Total**     | **2,225** |

> **Note:** The dataset is intentionally not included in this GitHub repository. It can be downloaded separately from Kaggle and placed in the appropriate `data/` directory when reproducing the project.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy
* Matplotlib
* Seaborn

### Natural Language Processing

* NLTK
* spaCy
* Regular Expressions

### Machine Learning

* Scikit-learn
* TF-IDF
* Logistic Regression
* Linear Support Vector Machine (Linear SVM)

### Deep Learning

* TensorFlow
* Keras
* BiLSTM

### Model Persistence

* Joblib

### Development Environment

* Jupyter Notebook
* Visual Studio Code

---

## 🔄 Project Workflow

```text
Raw Documents
      ↓
Data Loading
      ↓
Data Inspection
      ↓
Exploratory Data Analysis
      ↓
Text Cleaning & Preprocessing
      ↓
Feature Engineering
      ↓
Train/Test Split
      ↓
TF-IDF Feature Extraction
      ↓
Machine Learning Models
      ├── Logistic Regression
      └── Linear SVM
      ↓
Deep Learning Model
      └── BiLSTM
      ↓
Model Evaluation
      ↓
Model Comparison
      ↓
Best Model Selection
      ↓
Document Information Extraction
      ↓
Prediction Pipeline
      ↓
Save & Load Model
```

---

## 🧹 1. Text Preprocessing

The raw document text is cleaned before model training.

The preprocessing pipeline includes:

* Converting text to lowercase
* Removing URLs
* Removing email addresses
* Removing special characters
* Removing unnecessary whitespace
* Preparing clean text for feature extraction

Example:

```python
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

---

## 🔎 2. Exploratory Data Analysis

EDA was performed to understand the structure and characteristics of the dataset.

The analysis includes:

* Dataset shape
* Column information
* Data types
* Missing values
* Category distribution
* Document length analysis
* Word count analysis
* Most frequently occurring words

Visualizations were created using **Matplotlib** and **Seaborn**.

---

## ⚙️ 3. Feature Engineering

Additional document-level features were created to understand the characteristics of each document.

The engineered features include:

* Text length
* Word count
* Digit count
* Uppercase character count
* Exclamation mark count
* Question mark count

These features provide additional insights into the structure of the documents.

---

## 📝 4. TF-IDF Feature Extraction

The cleaned text was converted into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

The vectorizer was configured with:

* Maximum features: `10,000`
* Minimum document frequency: `2`
* Maximum document frequency: `0.95`
* N-gram range: `(1, 2)`
* Sublinear TF enabled

Both unigrams and bigrams were considered to capture individual words as well as meaningful word combinations.

---

## 🤖 5. Machine Learning Models

Two traditional Machine Learning classification algorithms were implemented.

### Logistic Regression

Logistic Regression was used as a strong baseline model for multi-class document classification.

### Linear SVM

A Linear Support Vector Machine was trained using the TF-IDF representation.

Linear SVM performed better than the other tested models and was selected as the final classification model.

---

## 🧠 6. Deep Learning — BiLSTM

A Bidirectional Long Short-Term Memory (BiLSTM) network was also implemented using TensorFlow/Keras.

The architecture includes:

```text
Input Text
    ↓
Text Vectorization
    ↓
Embedding Layer
    ↓
Bidirectional LSTM
    ↓
Dense Layer
    ↓
Dropout
    ↓
Softmax Output
```

Early stopping was used during training to reduce unnecessary training and help prevent overfitting.

---

## 📈 7. Model Performance

The models were evaluated using:

* Accuracy
* Weighted F1-score
* Classification Report
* Confusion Matrix

### Final Model Comparison

| Rank | Model               |   Accuracy | Weighted F1 |
| ---: | ------------------- | ---------: | ----------: |
| 🥇 1 | **Linear SVM**      | **97.98%** |  **97.98%** |
| 🥈 2 | Logistic Regression |     97.53% |      97.53% |
| 🥉 3 | BiLSTM              |     92.36% |      92.39% |

### Best Model

**Linear SVM** achieved the best overall performance with approximately:

* **97.98% Accuracy**
* **97.98% Weighted F1-score**

Therefore, Linear SVM was selected as the final document classification model.

---

## 🧩 8. Intelligent Document Analysis

In addition to classification, the system extracts useful information from document text.

The extraction pipeline identifies:

* Word count
* Character count
* Email addresses
* URLs
* Phone numbers
* Dates

For example, a document can be passed through the system and produce:

```text
Predicted Category : Politics
Predicted Label    : 0

Document Statistics
------------------------------
Word Count         : 213
Character Count    : 1289

Extracted Information
------------------------------
Emails             : []
URLs               : []
Phone Numbers      : []
Dates              : []
```

This demonstrates how the project goes beyond simple classification and combines **document understanding with information extraction**.

---

## 💾 9. Model Saving and Loading

The final Linear SVM pipeline components were saved using Joblib.

Saved artifacts include:

```text
models/
├── linear_svm_model.pkl
├── tfidf_vectorizer.pkl
└── label_mapping.pkl
```

These files allow the trained classification pipeline to be reused without retraining the model.

> The model files are currently kept locally and are not included in this repository checkpoint.

---

## 📁 Repository Structure

Currently, this repository contains the main Jupyter Notebook:

```text
Intelligent_Document_Understanding/
│
└── Intelligent_Document_Understanding.ipynb
```

The notebook contains the complete workflow from data loading and preprocessing to model training, evaluation, information extraction, and prediction.

For local reproduction, the expected working structure is:

```text
Intelligent_Document_Understanding/
│
├── data/
│   └── text_classification.csv
│
├── models/
│   ├── linear_svm_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── label_mapping.pkl
│
├── output/
│   └── model_comparison.csv
│
└── Intelligent_Document_Understanding.ipynb
```

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
cd Intelligent_Document_Understanding
```

### 2. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk spacy joblib tensorflow tqdm
```

### 3. Download the Dataset

Download the **Text Classification Documentation** dataset from Kaggle.

Place the CSV file inside:

```text
data/
```

The expected filename is:

```text
text_classification.csv
```

### 4. Open the Notebook

Open:

```text
Intelligent_Document_Understanding.ipynb
```

using Jupyter Notebook or Visual Studio Code.

### 5. Run the Cells

Run the notebook cells sequentially to:

* Load the dataset
* Perform EDA
* Clean the text
* Engineer features
* Train classification models
* Train the BiLSTM model
* Compare model performance
* Perform document analysis
* Test predictions

---

## 🔮 Example Prediction

The trained system can classify new documents automatically.

Example:

```python
prediction = predict_with_saved_model(document_text)

print("Predicted Category:", prediction)
```

Possible output:

```text
Predicted Category: Technology
```

---

## 💡 Key Skills Demonstrated

This project demonstrates practical experience in:

* Python Programming
* Data Cleaning
* Exploratory Data Analysis
* Natural Language Processing
* Text Preprocessing
* Feature Engineering
* TF-IDF
* Machine Learning
* Deep Learning
* Model Evaluation
* Classification
* Information Extraction
* Model Persistence
* End-to-End ML Workflow

---

## 📌 Key Takeaways

* Traditional NLP-based Machine Learning models performed very strongly on this dataset.
* Linear SVM achieved the highest classification performance among the tested approaches.
* TF-IDF with unigram and bigram features provided an effective representation of document text.
* BiLSTM demonstrated the application of Deep Learning for text classification.
* Combining classification with information extraction makes the system more useful for practical document-processing scenarios.

---

## 🔮 Future Improvements

The project can be further enhanced by:

* Adding PDF and DOCX document upload support
* Building an interactive Streamlit web application
* Adding Named Entity Recognition (NER)
* Extracting organization names, people, locations, and other entities
* Adding document summarization
* Supporting multiple document formats
* Adding confidence scores to predictions
* Deploying the application online
* Adding OCR support for scanned documents
* Creating a REST API for document classification

---

## 👩‍💻 Author

**Jana Priya**

Data Science Professional | Python | SQL | Machine Learning | NLP | Deep Learning

---

## ⭐ Project Highlights

**Problem:** Manual document classification and information extraction can be time-consuming.

**Solution:** An NLP-powered system that automatically classifies documents and extracts useful information.

**Best Model:** Linear SVM

**Best Accuracy:** **97.98%**

**Domain:** Natural Language Processing / Document Intelligence / Machine Learning

---

## 📄 License

This project is intended for educational, portfolio, and demonstration purposes.
