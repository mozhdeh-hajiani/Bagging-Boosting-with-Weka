# Bagging-Boosting-with-Weka
This repository contains an implementation of **ensemble learning methods**: Bagging, AdaBoost, and AdaBoost.M1 using **Weka’s C4.5 classifier** as the base learner.
---
##  Project Overview
* Implements **Bagging** and **Boosting** (AdaBoost, AdaBoost.M1) algorithms.
* Uses **C4.5 classifier** from **Weka** as the base learner.
* Handles **missing values** with mean (for numeric) or mode (for nominal).
* Evaluates performance on multiple real datasets.
---
##  Features
* **Bagging** with bootstrapped samples (sampling with replacement).
* **AdaBoost** for binary classification.
* **AdaBoost.M1** for multi-class classification.
  * Train/Test split = **80/20**
  * Number of iterations = **21**
  * Performance metric: **Test accuracy**
---
##  Datasets
* Breast
* Cleveland
* Mammographic
* Automobile
---
##  References
* Weka: [University of Waikato ML Software](http://www.cs.waikato.ac.nz/~ml/weka/)
* UCI/KEEL Dataset Repository: [http://sci2s.ugr.es/keel/missing.php](http://sci2s.ugr.es/keel/missing.php)
