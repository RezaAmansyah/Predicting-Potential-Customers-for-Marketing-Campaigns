

### **Project:** Predicting Potential Customers for Marketing Campaigns

**1. Project Overview**

* **Business Objective:**
  To optimize a classification model that predicts customers who are most likely to respond to a marketing campaign, while minimizing costs and financial losses from sending campaigns to uninterested customers.

* **Dataset:**
  2,240 rows, 29 initial features. Target variable: `Response` (accepted/not accepted). The data is imbalanced with a ratio of 85:15.

**2. Data Understanding & Preparation**

* **EDA & Initial Insights:**

  * Several features show a strong correlation with campaign response, such as `Ever_Accepted`, `NumStorePurchases`, and `MntMeatProducts`.
  * The dataset’s imbalance (85:15) requires special handling to prevent the model from being biased toward the majority class.

* **Preprocessing:**

  * Applied **RobustScaler** to handle outliers.
  * Encoded categorical features: `Education` (ordinal), `Marital_Status` (one-hot).
  * Used **SMOTENC** for resampling to address data imbalance.

**3. Modeling & Evaluation**

* **Model:** Support Vector Classifier (SVC) with parameters `kernel='rbf'`, `gamma='auto'`, `C=1`, `class_weight=None`.

* **Performance:**

  * Confusion Matrix: TN = 317, TP = 53, FP = 59, FN = 14
  * ROC-AUC = **0.82** → The model has a good ability to distinguish between classes.
  * PR-AUC = **0.65** → Fairly good considering the imbalanced dataset.

* **Top 5 Most Important Features:**

  1. `Ever_Accepted` → 0.067
  2. `NumStorePurchases` → 0.067
  3. `MntMeatProducts` → 0.058
  4. `Recency` → 0.053
  5. `Marital_Status` → 0.032

**4. Conclusion**

* The **SVC model** successfully identified potential customers with an **ROC-AUC score of 0.82**, indicating strong predictive performance.
* The model is **more sensitive to customers who do not respond** to campaigns (high TN) but can still be improved in **capturing potential customers** (relatively low TP).
* The most influential features for campaign response are **Ever_Accepted**, **NumStorePurchases**, and **MntMeatProducts**, which can be used as the basis for customer segmentation.
* Financial risks from sending campaigns to uninterested customers are minimized, but there remains a risk of **loss due to False Negatives** (potential customers not being detected).

**5. Recommendations**

* **Business / Campaign Strategy:**

  * Focus campaign efforts on customers with **high prediction scores** based on key features (`Ever_Accepted`, `NumStorePurchases`, `MntMeatProducts`).
  * Consider **personalized or targeted campaigns** for customers near the borderline score range to reduce False Negatives.

* **Model Improvement:**

  * Experiment with **`class_weight='balanced'`** in SVC or try other algorithms such as **Random Forest** or **XGBoost** to improve recall for the minority class.
  * Perform further **hyperparameter tuning** (e.g., `C`, `gamma`) to optimize the precision-recall trade-off.
  * Evaluate alternative **resampling techniques** (e.g., ADASYN, SMOTE variants) to enhance minority class performance.

* **Monitoring & Deployment:**

  * Implement **model monitoring systems** to track customer distribution changes and real-time model performance.
  * Regularly **retrain the model** to adapt to new customer behavior trends.

* **Data Enhancement:**

  * Add new features related to **recent customer interactions** with previous campaigns or communication channels (email, SMS, etc.) to improve predictive accuracy.
