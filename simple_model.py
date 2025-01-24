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
        if len(file) !=0:
            Path=loc
        
    return file, Path

def load_data(file, Path):
    if not(Path):
        print('data file in csv format does not exist')
        return None
    else:
        dfms=[]
        # form a file path
        for fl in file:
            fpath = os.path.join(Path, fl)
            df = pd.read_csv(fpath)
            dfms.append(df)
        return dfms
file, Path = locate_data()    
dfms = load_data(file, Path) 
  
import numpy as np
def explore_data(dfms):
     
    if len(dfms)==1:
        df=dfms[0]
    
    df = df.drop('patientid', axis=1)
    # compute correlations between features and the target
    correlations = df.corr()['target'].drop('target')
    # Display the correlations
    print("Feature Correlations with the Output:")
    print(correlations.sort_values(ascending=False))       
    
    train, validate, test = \
              np.split(df.sample(frac=1, random_state=42), 
                       [int(.7*len(df)), int(.8*len(df))])
    return train, validate, test

train, validate, test = explore_data(dfms)