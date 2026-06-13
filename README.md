# KQC7016 Assignment 2: Smart Hospital Data Analytics

This repository contains the code, datasets, and result files for KQC7016 Assignment 2.

## Project Title

**AI for Medicine: Smart Hospital**

## Team Members

* NIE JING
* DAI FENFANG

## Project Overview

This project applies data analytics methods to a smart hospital scenario. The aim is to support hospital operation efficiency, patient risk classification, healthcare service planning, and business intelligence decision-making.

The project is divided into two connected parts:

## Part A: Patient Waiting Time Prediction and Operation Efficiency

Part A focuses on predicting patient waiting time and identifying congestion patterns in a smart hospital.

### Methods Used

* Multiple Linear Regression
* K-Means Clustering

### Main Objectives

* To analyse patient waiting time patterns.
* To predict patient waiting time using hospital operation variables.
* To group patients into low, medium, and high waiting-time patterns.
* To support queue management and staff allocation.

### Main Files

* `PartA_WaitingTime_Analysis.ipynb`
* `smart_hospital_waiting_time.xls`
* `smart_hospital_waiting_time_with_clusters.xls`

### Main Results

The regression model achieved:

* MSE = 65.39
* RMSE = 8.09 minutes
* R² = 0.878

The clustering model identified three patient waiting-time groups:

* Cluster 0: High waiting-time group
* Cluster 1: Low waiting-time group
* Cluster 2: Medium waiting-time group

## Part B: Patient Risk Classification and Service Association Analysis

Part B focuses on classifying patient risk levels and discovering healthcare service usage patterns.

### Methods Used

* Decision Tree Classification
* Association Rule Mining

### Main Objectives

* To classify patients into Low, Medium, and High risk groups.
* To identify service association patterns.
* To support triage planning and healthcare resource allocation.

### Main Files

* `KQC7016_PartB_patient_risk_analysis.py`
* `smart_hospital_partb_patient_risk_dataset.csv`
* `partb_decision_tree_metrics.csv`
* `partb_association_rules.csv`
* `partb_fig_confusion_matrix.png`
* `partb_fig_risk_by_department.png`
* `partb_fig_risk_distribution.png`
* `partb_fig_service_by_risk.png`
* `partb_fig_simplified_decision_tree.png`

### Main Results

The decision tree model achieved:

* Accuracy = 74.4%
* Weighted precision = 73.5%
* Weighted recall = 74.4%
* Weighted F1-score = 73.5%

The association rules showed that high-risk patients were strongly linked with observation, ECG, laboratory tests, and hospital admission.

## Tools Used

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## Dataset Note

The datasets used in this project are simulated smart hospital datasets. They were created for academic analysis and do not contain real patient records.

