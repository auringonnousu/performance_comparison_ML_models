### Performance Comparison of tree-based models as part of my thesis.


You can run the code with the following file:

run_performance_comparison.py

Clone the repository
```bash
git clone https://github.com/auringonnousu/performance_comparison_ML_models.git
``` 

Navigate to the cloned directory
```bash
cd performance_comparison_ML_models
``` 

Run the Python script
```bash
python run_performance_comparison.py
```

Or click on this Binder badge:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/auringonnousu/performance_comparison_ML_models/HEAD)

<br>



Conduction of a comparison of classification performance and run-time for **Decision Tree Classifier**, **Random Forest Classifier** and **Gradient Boosting Classifier**.

RandomOverSampler is used to balance the training set.

GridSearch is used to find the best parameters for each model. 
5-fold Cross-validation is performed. 

The performance is evaluated on the test set.

The Built-in Feature Importance is used to find the most important features for each model.

Computing ROC AUC score for each model.

Training of models with only the most important features and parameter.

Steps:

1. Encoding using OneHotEncoder()
2. Applying RandomOverSampler()
3. Running pipeline per model with cv
4. Performing GridSearchCV() within pipeline for each model
5. Training of each model
6. Performing cross-validation on each model
7. Write results to df 
8. Visualization of metrics for current models 
9. Training of each model with best parameter and most important features
10. Evaluation of each model
