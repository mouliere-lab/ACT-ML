# ACT Score Pipeline:

## Overview
ACT stands for Aberrations, Contribution of short fragments, and Terminal motif analyses—a machine learning pipeline designed to predict the outcomes of Diffuse Large B-Cell Lymphoma (DLBCL) patients by analyzing on-treatment plasma samples. We leveraged machine learning techniques to develop and validate a predictive model using four genomic and fragmentomic features obtained from a single run of whole-genome sequencing data. The four key features were: enhanced tumor fraction estimated by ichorCNA, the proportion of short DNA fragments (20-150bp), the Fragment-End Integrated Analysis (FrEIA) score, and the Gini Diversity Index.

To benchmark performance, we evaluated four classifiers: Random Forest, Logistic Regression, Gaussian Naive Bayes, and Support Vector Machines (SVMs). Each classifier underwent hyperparameter optimization using a 4-fold cross-validation strategy aimed at maximizing predictive accuracy. To ensure the model’s robustness, training data was perturbed using five distinct random seeds, with hyperparameter optimization repeated for each iteration. This process generated five optimal models per classifier. For the validation cohort, the final classification prediction was determined by averaging the ACT scores produced by these five predictive models.

