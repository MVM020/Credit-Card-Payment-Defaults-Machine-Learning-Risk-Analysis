# Credit-Card-Payment-Defaults-Machine-Learning-Risk-Analysis
A machine learning project in R analyzing financial and demographic data from 30,000 credit card clients in Taiwan to predict payment default risk.
The project compares three predictive algorithms (Naïve Bayes, K-Nearest Neighbors (KNN), and Neural Networks) against findings from a published research paper (Credit Default Mining Using Combined Machine Learning and Heuristic Approach, 2018).


Dataset & Data Cleaning:Preprocessed 30,000 observations and 25 variables capturing customer demographics (gender, education, marital status), 6 months of repayment history, bill statement amounts, and past payment amounts. Data cleaning included converting files, removing ID variables, setting factor targets, scaling numeric features, and randomizing dataset partitions in R


Machine Learning Models:Implemented and tuned Naïve Bayes, KNN (k=5), and Neural Network models in R using packages like nnet, class, and e1071.


Results & Key Findings:K-Nearest Neighbors: Reached 92.2% accuracy (surpassing the paper's 82.8%) due to feature scaling, randomized partitioning, and optimized K-value selection.  Naïve Bayes: Reached 74.3% accuracy (improving on the benchmark's 67.2%) using Laplace smoothing parameter adjustments.  
Neural Networks: Reached 81.6% accuracy (aligning closely with the paper's 81.0%).  
