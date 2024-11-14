# ACT Score Pipeline:

## Overview
ACT stands for Aberrations, Contribution of short fragments, and Terminal motif analyses, a machine learning pipeline aiming to predict the outcome of Diffuse Large B-Cell Lymphoma patients by analyzing plasma collected on-treatment. We applied machine learning models to train and validate a model with four genomic and fragmentomic features derived from a single-run of whole genome sequencing data. The four features included were enhanced tumor fraction estimated by ichorCNA, proportion of short fragments (20-150bp), fragment-end integrated analysis (FrEIA) score, and Gini Diversity Index. We benchmarked three models: Random Forest (RF), Logistic Regression, Gaussian Naive Bayes, and Support Vector Machines (SVMs) to evaluate the performance.

Following hyperparameter optimization, the classifier’s parameters were configured to their optimized values, and the model was retrained using the entire training dataset. To ensure robustness, we perturbed the training data using five different seeds and repeated hyperparameter optimization, yielding five alternative optimal models. The classification prediction for the validation sample was derived by averaging the ACT scores produced by these five predictors.

Hyperparameter optimization was performed using a 4-fold cross-validation setting to maximize the prediction accuracy. To ensure robustness, we perturbed the training data with five different random seeds and repeated hyperparameter optimization, yielding five alternative optimal models. The classification prediction for the validation cohort was derived by averaging the ACT scores produced by these five predictors.

