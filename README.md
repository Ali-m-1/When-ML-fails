# When ML Fails — Shortcut Learning & Class Imbalance in Online Shopping Prediction

## Project Overview

This project was developed as part of the **ECC Spring 2026 Mini-Project: When ML Fails**.

The objective of the assignment was not only to train a machine learning model, but to investigate and explain a concrete failure mode through controlled experiments.

Our work focuses on the **Online Shoppers Intention Dataset** and studies two important failure modes:

1. **Shortcut Learning under Temporal Correlations**
2. **Class Imbalance in Purchase Prediction**

The project demonstrates how a high-performing model can still fail under realistic conditions, and how controlled experiments can reveal the causes of these failures.

---

# Dataset

Dataset used:

* **Online Shoppers Purchasing Intention Dataset (UCI 468)**

Task:

* Binary classification of user sessions leading to a purchase (`Revenue = True`) or not (`Revenue = False`).

The dataset contains:

* User browsing behavior
* Page visit statistics
* Session durations
* Traffic source information
* Visitor type
* Temporal information (`Month`, `Weekend`, `SpecialDay`)

---

# Main Research Question

> Does the model rely on temporal features such as `Month` as shortcuts instead of learning robust behavioral signals, leading to degraded generalization under distribution shift?

A secondary investigation studies the effect of severe class imbalance on minority-class detection.

---

# Methodology

## 1. Exploratory Data Analysis (EDA)

The first phase focused on identifying potentially suspicious correlations.

Main observations:

* Strong class imbalance
* Large variation of conversion rate across months
* Temporal dependence in purchasing behavior
* Some contextual features appeared more predictive than behavioral engagement features

Example:

| Month | Conversion Rate |
| ----- | --------------- |
| Feb   | 1.6%            |
| Nov   | 25.3%           |

This suggested that the model could exploit seasonality as a shortcut.

---

## 2. Reference Model

A reference XGBoost classifier was trained using a standard random train/test split.

### Initial Results

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 0.85  |
| Precision | 0.53  |
| Recall    | 0.13  |
| F1-score  | 0.21  |
| ROC-AUC   | 0.76  |

Although the accuracy appeared strong, recall on the minority class was extremely poor.

Feature importance analysis also showed heavy reliance on temporal features such as:

* `Month_Nov`
* `Month_Mar`
* `Month_Sep`

This supported the shortcut-learning hypothesis.

---

## 3. Controlled Experiment — Temporal Distribution Shift

To test the shortcut-learning hypothesis, the random split was replaced with a temporal split.

### Experimental Design

Training data:

* Earlier months

Testing data:

* Later months with different purchasing dynamics

The objective was to evaluate whether performance degraded when temporal correlations changed.

This experiment isolates the effect of seasonality and tests whether the model learned robust behavioral patterns or merely memorized temporal purchase probabilities.

---

## 4. Correction Strategy

Two correction strategies were explored.

### A. Shortcut Reduction

Temporal shortcut features were reduced or removed in order to force the model to rely more heavily on behavioral engagement signals.

### B. Class Imbalance Correction

The second correction targeted the imbalance problem directly using:

* `scale_pos_weight` in XGBoost

This approach penalizes errors on the minority class more strongly.

---

# Main Findings

* Recall improved significantly
* False negatives were reduced by almost half
* The correction improved minority-class detection
* Overall accuracy decreased slightly
* The trade-off between precision and recall became visible

---

# Key Lessons Learned

This project highlighted several important machine learning lessons:

* High accuracy can hide severe failure modes
* Models may exploit contextual shortcuts instead of causal signals
* Aggregate metrics are insufficient for evaluating robustness
* Controlled experiments are essential for diagnosing ML failures
* Fixing symptoms is different from fixing causes

---

# Threats to Validity

Several limitations must be acknowledged:

* Results depend on the train/test split
* Hyperparameters were not fully optimized
* Different imbalance correction methods could lead to different conclusions
* Temporal effects may vary on other datasets
* Some observed improvements may depend on random initialization

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Jupyter Notebook

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

#

---

# Authors

- BARHOUD OTHMANE

- RABBAH MOHAMED ALI

- SERBOUTI YOUSSEF
