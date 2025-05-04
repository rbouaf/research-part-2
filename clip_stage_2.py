import pickle
import numpy as np
import sys
import os
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

def main():
    # Load the train-test splits for image paths and determine labels
    paths_split = pickle.load(open('train_test_paths.pickle', 'rb'))
    train_paths = paths_split['train_imgs']
    val_paths = paths_split['test_imgs']
    
    # Load extracted features
    train_features_dict = pickle.load(open('stage_1_clip_features_train.pickle', 'rb'))
    val_features_dict = pickle.load(open('stage_1_clip_features_val.pickle', 'rb'))
    
    # Build feature arrays for train and validation based on the paths
    train_features = []
    train_labels = []
    for p in train_paths:
        if p in train_features_dict:
            train_features.append(train_features_dict[p])
            # Label: 0 if '/left/' in path, 1 if '/right/' in path
            train_labels.append(0 if '/left/' in p else 1)
    
    val_features = []
    val_labels = []
    for p in val_paths:
        if p in val_features_dict:
            val_features.append(val_features_dict[p])
            val_labels.append(0 if '/left/' in p else 1)
    
    train_features = np.stack(train_features, axis=0)
    val_features = np.stack(val_features, axis=0)
    
    # Define a pipeline: scaling + Linear SVM (using class balancing)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LinearSVC(dual=False, class_weight='balanced', verbose=0))
    ])
    
    # Train classifier on stage 1 features
    pipe.fit(train_features, train_labels)
    
    # Predict on validation set
    predictions = pipe.predict(val_features)
    print(classification_report(val_labels, predictions, target_names=['left', 'right']))
    print("Accuracy =", accuracy_score(val_labels, predictions))

if __name__ == '__main__':
    main()
