# Machine Learning Projects

A collection of supervised machine learning projects built with Python and scikit-learn, covering classification tasks across healthcare and finance domains. Crafting models for your financial and health requirements.

---

## Projects

### 1. Heart Disease Prediction
**Notebook:** `notebooks/heart_disease_prediction.ipynb`

Predicts the likelihood of heart disease using clinical features such as age, cholesterol, blood pressure, and chest pain type.

- **Model:** Random Forest Classifier
- **Dataset:** [Heart Disease Dataset](https://gist.github.com/trantuyen082001/1fc2f5c0ad1507f40e721e6d18b34138/raw/heart.csv)
- **Key Steps:** EDA → Correlation heatmap → Train/test split → Model training → Evaluation (accuracy, confusion matrix, classification report)

---

### 2. Loan Approval Prediction
**Notebook:** `notebooks/loan_approval_prediction.ipynb`

Predicts whether a loan application will be approved based on applicant demographics and financial background.

- **Model:** Random Forest Classifier
- **Dataset:** [Loan Approval Dataset](https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/Loan-Approval-Prediction.csv)
- **Key Steps:** Missing value imputation → Label encoding → Feature/target split → Model training → Model export (`.pkl`)

---

### 3. Financial News Sentiment Analysis
**Notebook:** `notebooks/financial_news_sentiment.ipynb`

Classifies financial news headlines as positive or negative using NLP and machine learning.

- **Model:** Logistic Regression with TF-IDF vectorisation
- **Dataset:** [Financial Market News Dataset](https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Financial%20Market%20News.csv)
- **Key Steps:** Text preprocessing (stopword removal, tokenisation) → TF-IDF feature extraction → Model training → Prediction on new headlines

---

## Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook or [Google Colab](https://colab.research.google.com/)

### Installation

```bash
git clone https://github.com/your-username/Machine-Learning.git
cd Machine-Learning
pip install -r requirements.txt
```

### Running the Notebooks

```bash
jupyter notebook notebooks/
```

Or open any `.ipynb` file directly in [Google Colab](https://colab.research.google.com/).

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computation |
| `scikit-learn` | Model training and evaluation |
| `matplotlib` / `seaborn` | Data visualisation |
| `nltk` | Natural language processing |

---

## Project Structure

```
Machine-Learning/
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    ├── heart_disease_prediction.ipynb
    ├── loan_approval_prediction.ipynb
    └── financial_news_sentiment.ipynb
```

---

## License

This project is open source and available under the [MIT License](LICENSE).
