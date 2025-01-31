#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 22:58:19 2025

@author: nataliyaportman
"""
import kagglehub
import pandas as pd
from os import path
import os
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

def locate_data():
    # import dataset
    # Download latest version
    loc = kagglehub.dataset_download("jocelyndumlao/cardiovascular-disease-dataset")
    print("Path to dataset files:", loc)
    Path = None
    if len(os.listdir(loc)) != 0:
        for dir in os.listdir(loc):
            fpath = os.path.join(loc,dir)
            os.chdir(fpath)
            extension = 'csv'
            file = glob.glob('*.{}'.format(extension))
            if len(file) !=0:
                Path =fpath
            
    else:
        extension = 'csv'
        file = glob.glob('*.{}'.format(extension))
        if len(file) != 0:
            Path = loc
        
    return file, Path

def load_data(file, Path):
    if not(Path):
        print('data file in csv format does not exist')
        return None
    else:
        dfms = []
        # form a file path
        for fl in file:
            fpath = os.path.join(Path, fl)
            df = pd.read_csv(fpath)
            dfms.append(df)
        return dfms

  
def explore_data(dfms):
     
    if len(dfms)==1:
        df = dfms[0]
    
    df = df.drop('patientid', axis=1)
    # compute correlations between features and the target
    correlations = df.corr()['target'].drop('target')
    # Display the correlations
    print("Feature Correlations with the Output:")
    print(correlations.sort_values(ascending=False))       
    # Sort correlations for better visualization
    correlations_sorted = correlations.sort_values(ascending=False)

    # Create a bar plot for correlations
    plt.figure(figsize=(8, 6))
    sns.barplot(x=correlations_sorted.index, y=correlations_sorted.values, palette='viridis')
    plt.title('Correlation of Features with Target Variable', fontsize=16)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Correlation Coefficient', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    #_______________________________
    print(df[['slope','target']].value_counts()) 
    #train, validate, test = \
    #          np.split(df.sample(frac=1, random_state=42), 
    #                   [int(.7*len(df)), int(.8*len(df))])
    #csv_file_dir = '/Users/nataliyaportman/Documents/GitHub/CardiovascularDiseasePrediction/data/'
    #train_path = os.path.join(csv_file_dir,'train.csv')  # Specify the desired file path
    #val_path = os.path.join(csv_file_dir,'validate.csv')  # Specify the desired file path
    #test_path = os.path.join(csv_file_dir,'test.csv')  # Specify the desired file path
    #train.to_csv(train_path, index=False)  
    #validate.to_csv(val_path, index=False) 
    #test.to_csv(test_path, index=False)
    
       
    #return train, validate, test


def predict_rule_based(feature_vector):
    # feature vector for an individual consists of 12 features
    # feature vector is of Pandas series type
    # convert to dictionary
    feat = feature_vector.to_dict()
    val = feat['slope']
    if val == 1 or val == 0:
        print("Patient has no cardiovascular disease")
        return 0.0
    elif val == 2 or val == 3:
        print("Patient has cardiovascular disease")
        return 1.0
    else:
        print ('Acceptable slope values are 0, 1, 2 and 3. Cannot infer diagnosis')
        return None
 
def evaluate_acc():
    # this code evaluates accuracy of prediction using a test set
    csv_file_dir = '/Users/nataliyaportman/Documents/GitHub/CardiovascularDiseasePrediction/data/'
    os.chdir(csv_file_dir)
    df = pd.read_csv('test.csv')
    s = 0.0
    y_pred = []
    y_true = []
    for i in range(len(df.index)):
        record = df.loc[i,:]
        pred_y = predict_rule_based(record)
        true_y = record['target']
        if true_y == pred_y:
            s = s+1.0
        y_pred.append(pred_y)
        y_true.append(true_y)
    acc = s/len(df.index)
    print("Accuracy of the rule-based prediction algorithm is ", acc)   
    #calculate precision and recall

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    # Print the results
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

if __name__ == '__main__':
    file, Path = locate_data()    
    dfms = load_data(file, Path)   
    explore_data(dfms)     
    #train, validate, test = explore_data(dfms)
    evaluate_acc()
