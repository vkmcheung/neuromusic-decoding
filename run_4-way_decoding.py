# -*- coding: utf-8 -*-
"""
Decoding drums, instrumentals, vocals, and mixed sources in music using human brain activity with fMRI
--paper accepted to ISMIR 2023

Four-way classification

@code author: Vincent K.M. Cheung
"""


#%% start

import numpy as np
from nilearn.maskers import NiftiMasker
import os.path

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa


labels = ['merged','backing','drums','vocals']

def classify(clf,X_test,y_true):
    y_pred = clf.predict(X_test)
    try:
        y_prob = clf.predict_proba(X_test)
        roc_auc = roc_auc_score(y_true, y_prob, multi_class = 'ovr')
    except:
        y_prob = None
        roc_auc = np.nan
    
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels = labels)
    return acc,roc_auc,cm,y_pred,y_prob

def classify_CNN(clf,X_test,y_true):
    y_prob = clf.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_true, axis=1)

    roc_auc = roc_auc_score(y_true, y_prob, multi_class = 'ovr')
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return acc,roc_auc,cm,y_pred,y_prob

def write_results(classifier, mask, vp, acc, roc_auc, cm):        
    print(classifier+' decoding with '+mask+' leaving out sub-'+f'{vp:02}')
    print([acc, roc_auc])
    print('\n\n')
    
    import csv
    fname = 'results_4-way_decoding.csv' 
    header = ['classifier','mask','leftout-subject', 'accuracy','roc-auc','confusion_matrix']
    data = [classifier, mask, vp, acc, roc_auc, cm]
    
    if os.path.isfile(fname) == False:        
        with open(fname, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)    
    with open(fname, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)

# preprocessing pipeline
pipe = Pipeline([
    ('remove_0_features',VarianceThreshold()),
    ('robust_scaler',RobustScaler()),            
    ])


#### get data

for mask in ['auditory_bil', 'auditory_L','auditory_R', 'primary_visual_bil','somatosensory_and_motor_bil']:

    # load mask    
    maskfile = mask+'.nii' 
    
    # instantiate masker
    masker = NiftiMasker(maskfile,standardize_confounds=False)
        
    # load data
    allXfile = mask+'_data.npy' 
    allfnamefile = mask +'_fname.npy' 
    
    print('loading data for '+mask)
    allX = np.load(allXfile)
    allfname = np.load(allfnamefile)


#### process data


    # prepare labels
    label_subject = np.array([x.split('_')[0][-2:] for x in allfname], dtype='int')
    label_song = np.array([x.split('_')[4][-3:] for x in allfname], dtype='int')
    label_rating = np.array([x.split('_')[3][-1] for x in allfname], dtype='int')
    label_dict = {labels[0]  : 0,
                  labels[1]  : 1,
                  labels[2]  : 2,
                  labels[3]  : 3}
    label_version = np.array([x.split('_')[-1][8:-4] for x in allfname])
    label_version_num = np.array([label_dict[x] for x in label_version], dtype='int')




#### RF and SVM
    
    for vp in np.unique(label_subject):
        
        
        # split into train and test        
        Xtr = allX[label_subject != vp, :]
        y_train = label_version[label_subject != vp]
        
        Xte = allX[label_subject == vp, :]
        y_true = label_version[label_subject == vp]   
        
        
        X_train = pipe.fit_transform(Xtr,y_train)
        X_test = pipe.transform(Xte)

            
        # evaluate models        
        from sklearn.ensemble import RandomForestClassifier
        classifier = 'RF'
        clf = RandomForestClassifier(random_state=42, n_jobs=-1,
                  ).fit(X_train, y_train)
        acc,roc_auc,cm,y_pred,y_prob = classify(clf,X_test,y_true)
        write_results(classifier,mask,vp,acc,roc_auc,cm) #write results                 
        
        from sklearn.svm import SVC
        classifier = 'SVM'
        clf = SVC(random_state=42, probability=True,
                  kernel = 'linear',
                  max_iter=10000).fit(X_train, y_train)      
        acc,roc_auc,cm,y_pred,y_prob = classify(clf,X_test,y_true)
        write_results(classifier,mask,vp,acc,roc_auc,cm) #write results  
        
        
        
#### CNN

        # split into train, test, and validation
        valvp0 = np.random.permutation(np.setdiff1d(label_subject,vp))[0] # random validation subject from train
        valvp1 = np.random.permutation(np.setdiff1d(label_subject,vp))[1] # random validation subject from train
        
        Xtr = allX[(label_subject != vp) & (label_subject != valvp0) & (label_subject != valvp1), :]
        y_train = label_version_num[(label_subject != vp) & (label_subject != valvp0) & (label_subject != valvp1)] # use numeric labels for onehot later
        
        Xva = allX[(label_subject == valvp0) | (label_subject == valvp1), :]
        y_validation = label_version_num[(label_subject == valvp0) | (label_subject == valvp1)]   
        
        Xte = allX[label_subject == vp, :]
        y_true = label_version_num[label_subject == vp]   
        

        X_train = pipe.fit_transform(Xtr,y_train)
        X_validation = pipe.transform(Xva)
        X_test = pipe.transform(Xte)
        
        
        
        ### keras CNN model
        keras.backend.clear_session()
        
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, len(labels))
        y_validation = keras.utils.to_categorical(y_validation, len(labels))
        y_true = keras.utils.to_categorical(y_true, len(labels))
        
        classifier = 'CNN'
        inputs = keras.Input(shape=(np.shape(X_train)[1],1))
        x = layers.Conv1D(96, kernel_size=4, strides=4)(inputs)
        def res2(x):
            fx = layers.Conv1D(96, kernel_size=7, padding='same')(x)
            fx = layers.LayerNormalization()(fx)
            fx = layers.Conv1D(384, kernel_size=1, padding='same')(fx)
            fx = keras.activations.gelu(fx)
            fx = layers.Conv1D(96, kernel_size=1, padding='same')(fx)
            out = layers.Add()([x,fx])
            out = layers.ReLU()(out)
            return out
        x = res2(x)
        x = layers.Dense(1024)(x) 
        x = layers.Dense(512)(x)
        x = layers.Dense(256)(x)
        x = layers.Flatten()(x)
        outputs = layers.Dense(len(labels), activation="softmax")(x)
                
        
        model = keras.Model(inputs = inputs, outputs = outputs)        
        batch_size = 512
        callback = keras.callbacks.EarlyStopping(patience=25, verbose=0, restore_best_weights=True)        
        optimizer = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        model.fit(X_train, y_train, batch_size=batch_size, epochs=200, validation_data=(X_validation,y_validation), callbacks = [callback], verbose = 0)
        acc,roc_auc,cm,y_pred,y_prob = classify_CNN(model,X_test,y_true)
        write_results(classifier,mask,vp,acc,roc_auc,cm) #write results 