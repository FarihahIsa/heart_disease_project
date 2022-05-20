#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 20:04:48 2022

@author: farihahisa
"""

import pickle
import os
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') # GPU or CPU

for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

#%% Paths

SS_SCALER_PATH = os.path.join(os.getcwd(),  'ss_scaler.pkl')
KNN_CLASS_PATH = os.path.join(os.getcwd(), 'knn_classifier.pkl')

#%% loading of setting or model

standard_scaler = pickle.load(open(SS_SCALER_PATH),'rb')
knn_model = pickle.load(open(KNN_CLASS_PATH),'rb')

heart_disease_chance = {0:'negative', 1:'positive'}

model = knn_model
#%% Deployment

patient_info = np.arrays([5,116,74,0,0,25,6,0,201,30]) # true label 0

patient_info_scaled = standard_scaler.transform(np.expand_dims(patient_info,axis=0))


model.predict(patient_info_scaled)

outcome = model.predict(patient_info_scaled)

print(np.argmax(outcome))
print(heart_disease_chance[np.argmax(outcome)])


#%% streamlit

with st.form('Heart Disease Prediction form'):
    st.write("Patient's info")
    sex = int(st.number_input('sex'))
    cp = st.number_input('cp')
    trtbps = st.number_input('trtbps')
    chol= st.number_input('chol')
    fbs = st.number_input('fbs')
    restecg = st.number_input('restecg')
    thalachh = st.number_input('thalachh')
    exng= st.number_input('exng')
    oldpeak = st.number_input('oldpeak')
    slp = st.number_input('slp')
    caa= st.number_input('caa')
    thall = st.number_input('thall')
    age = int(st.number_input('age'))
    
submitted = st.form_submit_button('submit')

if submitted == True:
    patient_info = np.array(['age','trtbps','chol','thalachh','oldpeak'])
    patient_info_scaled = standard_scaler.transform(np.expand_dims(patient_info,
                                                             axis=0))
outcome = model.predict(patient_info_scaled)

st.write(heart_disease_chance[np.argmax(outcome)])

if np.argmax(outcome)==1:
    st.warning('Chance of getting heart disease is higher')
else:
    st.snow()
    st.succes('Chance of getting heart disease is small')