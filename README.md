# Support Vector Machine (SVM) Classification - Task 9

## Project Overview
This repository contains the implementation of a Support Vector Machine (SVM) model using the Iris dataset, comparing Linear and Radial Basis Function (RBF) kernels as part of the Edutech Solution Data Science Internship.

## Core Implementations
* **Data Processing**: Scaled data using `StandardScaler` to prevent feature dominance in distance metrics.
* **Kernel Implementation**: Trained both Linear and RBF versions of `SVC`.
* **Evaluation**: Applied confusion matrices and classification reports to compare performance.

## Kernel Comparison Report
* **Linear Kernel**: Achieved **97% accuracy** on the test set. It misclassified 1 instance of Iris-versicolor as Iris-virginica. It builds a flat hyperplane separation.
* **RBF Kernel**: Achieved **100% accuracy** on the test set. It creates a flexible, curved decision boundary by mapping the data into a higher-dimensional space, handling overlapping distributions perfectly on this test split.

---

## 3. Understanding Hyperplanes and Margins (Model Parameters)

Based on the model parameters extracted from the trained Support Vector Machine (SVC), here is the mathematical and geometric understanding of how our boundaries were formed:

### A. Linear SVC Parameters
* **Hyperplane Coefficients ($w$):** Extracted via `linear_svc.coef_`. These represent the normal vectors defining the orientation of the flat linear hyperplanes. Since Scikit-learn uses a One-vs-One (OvO) strategy for multi-class problems, it sets up **3 separate hyperplanes** to split the 3 flower classes.
* **Intercepts ($b$):** Extracted via `linear_svc.intercept_`. This represents the bias or offset of each hyperplane relative to the coordinate origin.
* **Support Vectors:** Selected using `linear_svc.n_support_`. These are the edge data points resting exactly on the margins that dictate the maximum width of the separation channel.

### B. RBF SVC Parameters
* **Why no Coefficients ($w$)?** The Radial Basis Function (RBF) kernel maps features into an infinite-dimensional space where an explicit flat weight vector cannot be mapped out. Therefore, `coef_` does not exist for the RBF kernel.
* **Dual Coefficients:** Extracted via `rbf_svc.dual_coef_` ($\alpha_i y_i$). Instead of a single feature weight, the model assigns individual weights to the support vector points, using them as geometric anchors to blend a flexible, curved boundary.
* **Support Vectors:** Monitored using `rbf_svc.n_support_` to count the specific borderline samples from each category responsible for creating the curved margins.

---

## Interview Questions & Answers

### 1. What is a Hyperplane?
A hyperplane is the decision boundary separating different classes in a dataset. 
* In a 2D space, it is a line.
* In a 3D space, it is a flat plane.
* In an N-dimensional space, it is an (N-1)-dimensional subspace.

### 2. Explain the 'Kernel Trick'.
The kernel trick is a mathematical approach that allows SVMs to find non-linear boundaries without physically transforming data into a higher dimension. It calculates the similarity relationships between coordinates via inner products directly in the original space, saving massive computational overhead.
