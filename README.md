# Predicting-the-efficacy-of-a-new-pharmaceutical-drug
In this project, we are using machine learning to develop a model that can predict the efficacy of a new pharmaceutical drug based on its chemical properties. The goal is to help researchers in the pharmaceutical industry to identify potentially effective drugs more efficiently.

We are using a dataset containing information on various pharmaceutical drugs and their efficacy levels, as well as their chemical properties such as molecular weight, polarity, and solubility. We preprocess the data by checking for missing values, removing duplicates, and converting categorical variables to numerical values.

We then engineer new features based on the chemical properties of the drugs using the RDKit library, which generates molecular descriptors that describe the chemical properties of a molecule. We use these features to train a variety of machine learning algorithms such as Random Forest, Gradient Boosting, and Neural Networks. We evaluate each model using cross-validation and select the best performing model based on its accuracy and F1 score.

Finally, we deploy the best performing model as a REST API using Flask, which allows users to input the chemical properties of a new drug and receive a predicted efficacy score. This makes the model accessible to researchers and other stakeholders in the pharmaceutical industry, who can use it to make more informed decisions about which drugs to pursue further development.
