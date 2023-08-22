
Performance Comparison of tree-based models
Conduction of a comparison of classification performance and runtime for **Decision Tree Classifier**, **Random Forest Classifier**, and **Gradient Boosting Classifier**.

RandomOverSampler is used to balance the training set.

GridSearchCV is used to find the best parameters for each model. The performance is evaluated on the test set.

Cross-validation is performed. 

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
7. Write results to df ['model_name', 'f1_score', 'accuracy', 'precision', 'recall', 'roc_auc',  'roc_curve', 'auc', 'feature_importances',
                        'best_parameters', 'execution_time', 'mean_cv_score', 'std_dev_cv_score']
8. Visualization of metrics for current models ['feature_importances','execution_time', 'roc_auc']
9. Training of each model with best parameter and most important features
10. Evaluation of each model
