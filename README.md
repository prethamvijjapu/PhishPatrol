# PhishPatrol
# 🔐 PhishPatrol: Redefining Phishing Detection using Machine Learning

![PhishPatrol Logo](#) <!-- Add your logo or project image here -->

## 📌 Project Overview

**PhishPatrol** is a machine learning-based phishing URL detection system designed to identify malicious websites using ensemble models. It leverages lexical, host-based, and content-based URL features to classify phishing attempts with high precision and speed. The core model integrates powerful ML classifiers like XGBoost, AdaBoost, and MLP using a Voting Classifier to ensure robustness and accuracy.

---

## 📊 Key Features

- ✅ URL-based phishing detection
- 🧠 Ensemble learning with Voting Classifier (XGB, AdaBoost, MLP)
- 📈 Achieves over **97.8% accuracy** on benchmark datasets
- 🚀 Real-time prediction interface using **Streamlit**
- 🔍 Feature engineering with EDA, normalization, and selection
- 🛡️ Lightweight and scalable for deployment

---

## ⚙️ Tech Stack

- **Python 3.10+**
- **Scikit-learn**, **XGBoost**, **Optuna**
- **Streamlit** (Frontend for real-time URL input)
- **Pandas**, **Matplotlib**
- **Joblib** (for model serialization)

---

## 🧪 Machine Learning Models

| Model              | Accuracy (Dataset 1) | Accuracy (Dataset 2) |
|-------------------|----------------------|----------------------|
| Decision Tree      | 95.79%               | 97.70%               |
| Random Forest      | 96.97%               | 98.25%               |
| AdaBoost           | 93.62%               | 97.65%               |
| XGBoost            | 97.24%               | 98.80%               |
| MLP                | 97.01%               | 98.05%               |
| Gradient Boost     | 96.88%               | 98.85%               |
| **Voting Classifier** | **97.24%**       | **98.75%**           |

---

## 🧠 Methodology

The project follows a three-module structure:

1. **Data Preparation & Feature Engineering**  
   - Preprocessing two phishing datasets (~21k URLs total)
   - Feature selection: lexical, SSL, domain age, special chars
   - Normalization and encoding

2. **Model Training & Evaluation**  
   - Trained 7 ML classifiers
   - Evaluated using Accuracy, Precision, Recall, F1-Score
   - Hyperparameter tuning via **Optuna**

3. **Voting Ensemble Classifier**  
   - Combined best-performing models (XGB, MLP, RF)
   - Majority-vote-based classification
   - Streamlit GUI for real-time predictions

---

## 🧰 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/phishpatrol.git
   cd phishpatrol
SNAPSHOTS
Fig 4.1: DATASET 1: URL Features Dataset -1
Figure 4.1 presents a similar comparison for Dataset 1. In this chart, the red bar represents phishing URLs, while the green bar represents legitimate ones. The chart indicates that legitimate URLs are slightly more numerous than phishing URLs. 
  
Fig 4.2:  DATASET 2:URL Features Dataset -2
Figure 4.2 presents a similar comparison for Dataset 2. The green bar corresponds to legitimate URLs, and the red bar to phishing URLs. In this dataset, legitimate URLs appear marginally more frequently than phishing URLs.
 AccuracyPrecisionRecallF1 ScoreDecision Tree0.95790.96410.9620.963Random Forest0.96970.96550.9820.974Adaboost0.93620.93580.9530.944XGBoost0.97240.96640.9860.976MLP0.97010.96630.9820.973Gradient Boost0.96880.96110.9850.979Voting Classifier0.97240.96420.9880.976
Table 4.1 FIRST DATASET RESULT
The table 4.1 presents the performance metrics (Accuracy, precision, recall and f1-score) of various machine learning classifier on first dataset among models evaluated. Voting classifier achieved the highest accuracy. 

 AccuracyPrecisionRecall F1 ScoreDecision Tree0.9770.97450.98020.9773Random Forest0.98250.98220.98320.9827MLP0.98050.98410.97730.9807XG Boost0.9880.98430.99210.9882AdaBoost0.97650.97350.98020.9769Gradient Boost0.98850.98530.99210.9827Voting0.98750.98720.99210.9897Table 4.2: SECOND DATASET RESULTS
The table 4.2 presents the performance metrics (Accuracy, precision , recall and f1-score) of various machine learning classifier on first dataset among models evaluated. XG Boost and voting
classifier achieved the highest accuracy. 



Fig 4.3 Performance results for first dataset
Figure 4.3 compares the performance of various classifiers on the first dataset using Accuracy, Precision, Recall, and F1 Score. XGBoost shows the best overall performance, achieving the highest scores across all metrics. Ensemble methods like Random Forest, Gradient Boost, and the Voting Classifier also perform well. Adaboost and Decision Tree have slightly lower scores, indicating comparatively weaker performance.

 
Fig 4.4 Performance results for second dataset
Figure 4.4 compares the performance of various classifiers on the second dataset using Accuracy, Precision, Recall, and F1 Score. XGBoost shows the best overall performance, achieving the highest scores across all metrics. Ensemble methods like Random Forest, Gradient Boost, and the Voting Classifier also perform well. Adaboost and Decision Tree have slightly lower scores, indicating comparatively weaker performance.

  
Fig 4.5 Comparison between Datasets
Figure 4.5 shows an accuracy comparison of different classifiers on two datasets. XGBoost and Gradient Boost achieved the highest accuracy on both datasets (up to 0.99), while MLP showed the most significant improvement, rising from 0.94 to 0.98. Overall, ensemble methods maintained strong and consistent performance across both datasets.


	SAMPLE OUTPUT FOR LEGITIMATE AND ILLEGITIMATE CASE:
            
Fig 4.6: Positive Output
Figure 4.6 displays the output interface of the "PhishPatrol" tool, which is designed to detect suspicious URLs using machine learning. In this instance, the tool correctly identifies "www.google.com" as a legitimate website and returns a positive output.

Fig 4.7 : Negative Output
Fig 4.7:Negative Output This figure demonstrates the response of the PhishPatrol system when an illegitimate or suspicious URL is entered. The tool accurately detects and labels the website as a phishing threat, showcasing its effectiveness in real-time URL classification




