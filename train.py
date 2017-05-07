# -*- coding: utf-8 -*-
"""
Created on Sat May  6 08:30:29 2017

@author: Matthew
"""

import os
import numpy as np
import scipy.io.wavfile as wavfile
from python_speech_features import mfcc

vctk_dir = 'C:\\Users\\Matthew\\Desktop\\AIML\\Binaries\\VCTK-Corpus-Downsampled\\'

# Step 1 Data Ingestion
# a) Speaker M/F Status
spkr_dict = {}
with open(''.join((vctk_dir,'speaker-info.txt')), 'r') as spkrfile:
    for line in spkrfile:
        line_data = line.strip().split()
        # Create dictionary mapping of ID: GENDER
        if line_data[0] != 'ID':
                    spkr_dict[line_data[0]] = line_data[2]

# b) Get a list of all files
fileset_list = []
targets = []
for root, dirs, files in os.walk(vctk_dir):
    for file in files:
        if file.endswith(".wav"):
            if spkr_dict.get(file[1:4]) is not None:
                fileset_list.append(os.path.join(root, file))
                # Append M/F target
                targets.append(spkr_dict[file[1:4]])

# Step 2 Feature Extraction
n_features = 20
complete_dataset = np.zeros((len(targets),n_features))
for indx, record in enumerate(fileset_list):
    if indx%1000 == 0:
        print("Processing ", str(indx), '/', str(len(targets)))
    sampling_rate, audio_samples = wavfile.read(record)
    mfcc_feat = mfcc(audio_samples, sampling_rate, numcep=n_features)
    complete_dataset[indx,:] = mfcc_feat.mean(axis=0) # Take the mean MFCC's over the whole audio signal
print('Feature Extraction Complete!')

# Step 3 Shuffle Datasets (Keep target index proper)
shuffle_index = np.random.permutation(complete_dataset.shape[0])
shuffled_dataset = complete_dataset[shuffle_index]
shuffled_targets = np.array(targets)[shuffle_index]

# Step 4 Split Into Training and Test Sets
training_percentage = 0.9
training_records = int(shuffled_dataset.shape[0]*training_percentage)

training_inputs = shuffled_dataset[:training_records,:]
training_targets = shuffled_targets[:training_records]
test_inputs = shuffled_dataset[training_records:,:]
test_targets = shuffled_targets[training_records:]

# Step 5 Training
from sklearn import svm
clf = svm.SVC()
clf.fit(training_inputs, training_targets)

# Step 6 Save the Model for Production
from sklearn.externals import joblib
joblib.dump(clf, 'model.pkl')

# Step 7 Testing
predictions = clf.predict(test_inputs)
accuracy = np.mean(predictions==test_targets)
print("Accuracy: ", accuracy*100., "%")



