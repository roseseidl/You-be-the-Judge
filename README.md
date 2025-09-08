# You-be-the-Judge
Project for The Center for AI Safety's "AI Safety, Ethics, and Society" class. Developing an AI judge based on prompts from the Guardian's "You be the Judge" column.

## Personal Dilemma Verdict Predictor
This project implements an ensemble machine learning framework for predicting whether a person is "guilty" or "innocent" in personal dilemmas taken from the Guardian. It combines Logistic Regression, Random Forests, and a custom PyTorch Neural Network to evaluate performance across multiple runs, using both semantic features and TF-IDF representations of input data.

## The system includes:
Feature engineering tailored for moral dilemmas (responsibility, emotions, remorse indicators, etc.).  
TF-IDF features for accusation and defence texts.  
Comparative and ratio-based semantic features.  
Feature selection using mutual information.  
Model training with cross-validation, hyperparameter tuning, and early stopping.  
Multi-run experiments (default: 10 runs) with test performance tracking.  
Visualization and summary statistics comparing model accuracies.  

## Project Structure
PersonalDilemmaNN: Custom 2-layer neural network with dropout and sigmoid output.  
train_neural_network: Handles cross-validation, hyperparameter search, and early stopping for the NN.  
train_optimized_ensemble: Trains Logistic Regression, Random Forest, and the Neural Network, choosing hyperparameters with CV.  
extract_personal_dilemma_features: Extracts custom moral dilemma features (responsibility, emotion, remorse, etc.).  
create_advanced_features: Creates a combined feature set (TF-IDF + semantic + comparative + ratio).  
run_single_experiment: Runs a full train/val/test split, model training, and evaluation.  
plot_results: Generates bar charts and statistical summaries across multiple runs.  
main: Runs experiments (default: 10) on the dataset and produces plots + summaries.  

## Installation Prerequisites
Python 3.8+  
GPU with CUDA (optional, recommended for Neural Network)  
## Dependencies
Install required libraries:  
bash  
pip install pandas numpy scikit-learn torch matplotlib  

## Dataset
The system expects a CSV file (default: guardian_dataset.csv) with the following columns:  
prosecution: Text of the accusation.  
defence: Text of the defence.  
verdict: Either "guilty" or "innocent".  
confidence:   
If the verdict vote is  
50-65 --> low confidence  
66-85 --> mid confidence  
86-100 --> high confidence  


This project is provided as-is for research and educational purposes.
