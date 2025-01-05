# Customer Churn Prediction

This repository contains the implementation of a **Customer Churn Prediction** system using machine learning and deep learning techniques. The project aims to predict whether a bank customer will leave the institution based on various features, helping banks take proactive measures to retain valuable clients.

---

## Project Overview

### Objective
To develop a predictive model that identifies the probability of customer churn using machine learning and artificial neural networks (ANN). The model helps banks understand customer behavior and reduce churn rates.

### Dataset
The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/code/kmalit/bank-customer-churn-prediction/data). It contains 14 features and 10,200 rows of data related to customer demographics, account information, and activity levels.

#### Key Features:
1. **Credit Score**: Credit score of the customer.
2. **Geography**: Country of the customer.
3. **Gender**: Gender of the customer.
4. **Age**: Customer's age.
5. **Tenure**: Duration of association with the bank.
6. **Balance**: Account balance.
7. **Number of Products**: Number of bank products utilized.
8. **HasCrCard**: Whether the customer has a credit card (binary).
9. **IsActiveMember**: Whether the customer is active (binary).
10. **Estimated Salary**: Customer's salary estimate.
11. **Exited**: Target variable indicating churn (1 if churned, 0 otherwise).

---

## Methods

### Machine Learning Baseline
- **Logistic Regression**:
  - Used as the baseline model for binary classification.
  - Achieved an accuracy of **81%**.

### Deep Learning Model
- **Artificial Neural Network (ANN)**:
  - Architecture:
    - 5 layers: 14, 16, 12, 6, and 1 neuron(s).
    - **Activation Function**: GELU (Gaussian Error Linear Unit).
    - **Optimizer**: Adam.
    - **Loss Function**: Binary Crossentropy.
  - Achieved an accuracy of **87%**.

---

## Evaluation Metrics
The models were evaluated using the following metrics:

1. **Precision**: Measures the quality of positive predictions.
2. **Recall**: Measures the completeness of positive predictions.
3. **F1 Score**: Harmonic mean of precision and recall.

---

## Results

| Model                  | Accuracy |
|------------------------|----------|
| Logistic Regression    | 81%      |
| Logistic Regression (Enhanced) | 82.5%   |
| ANN                    | 87%      |

---

## Key Insights

1. **Baseline Model**: Logistic Regression provided a robust starting point with an 81% accuracy.
2. **Deep Learning**: ANN significantly improved prediction accuracy to 87%.
3. **Limitations**:
   - Small dataset size limited the model's performance.
   - Confidentiality constraints prevented the use of real-world banking data.

---

## Future Work

1. **Dataset Improvements**:
   - Use larger datasets with millions of records.
   - Include customer feedback data for enhanced predictions.
2. **Model Optimization**:
   - Explore other activation functions and optimizers.
   - Experiment with advanced neural network architectures.

---

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the model:
   ```bash
   python main.py
   ```

---

## References

1. [Bank Customer Churn Prediction Dataset - Kaggle](https://www.kaggle.com/code/kmalit/bank-customer-churn-prediction/data)
2. Saishruthi Swaminathan, Logistic Regression - Detailed Overview, [Towards Data Science](https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc)
3. Jason Brownlee, Adam Optimization Algorithm, [Machine Learning Mastery](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
