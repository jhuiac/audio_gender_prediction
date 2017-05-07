# -*- coding: utf-8 -*-
"""
Created on Sun May  7 18:05:06 2017

@author: Matthew
"""

import numpy as np
import scipy.io.wavfile as wavfile
from python_speech_features import mfcc

vctk_dir = 'C:\\Users\\Matthew\\Desktop\\AIML\\Binaries\\VCTK-Corpus-Downsampled\\'

# Step 1 Load the Model
from sklearn.externals import joblib
clf = joblib.load('model.pkl')

# Step 2 Feature Extraction on File
input_file = 'C:\\Users\\Matthew\\Desktop\\AIML\\gender_detection\\test.wav'
n_features = 20
sampling_rate, audio_samples = wavfile.read(input_file)
mfcc_feat = mfcc(audio_samples, sampling_rate, numcep=n_features)
input_data = mfcc_feat.mean(axis=0) # Take the mean MFCC's over the whole audio signal

# Step 3 Prediction
print("Prediction: ",clf.predict(input_data))
