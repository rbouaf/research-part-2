#!/usr/bin/env python3
import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

def load_and_diagnose():
    # Load your train/test splits and extracted features
    paths = pickle.load(open('train_test_paths.pickle', 'rb'))
    all_features = pickle.load(open('stage_1_features.pickle', 'rb'))
    train_paths = paths['train_imgs']
    test_paths  = paths['test_imgs']

    # Figure out which paths we actually have features for
    found_train = [p for p in train_paths if p in all_features]
    miss_train  = [p for p in train_paths if p not in all_features]
    found_test  = [p for p in test_paths  if p in all_features]
    miss_test   = [p for p in test_paths  if p not in all_features]

    # Print a quick summary
    print(f"Train paths total: {len(train_paths)}")
    print(f" → features found: {len(found_train)}, missing: {len(miss_train)}")
    if miss_train:
        print("  Missing train (first 10):", miss_train[:10])
    print(f"Test  paths total: {len(test_paths)}")
    print(f" → features found: {len(found_test)}, missing: {len(miss_test)}")
    if miss_test:
        print("  Missing test (first 10):", miss_test[:10])

    # Bail out early if nothing to train or test on
    if not found_train:
        raise RuntimeError("No training features found! Check your stage_1_features.pickle")
    if not found_test:
        raise RuntimeError("No testing  features found! Check your stage_1_features.pickle")

    # Stack into arrays
    X_train = np.stack([all_features[p] for p in found_train], axis=0)
    y_train = ['left' if '/left/' in p else 'right' for p in found_train]

    X_test  = np.stack([all_features[p] for p in found_test], axis=0)
    y_test  = ['left' if '/left/' in p else 'right' for p in found_test]

    return X_train, y_train, X_test, y_test

def main():
    # Load data and run diagnostics
    X_train, y_train, X_test, y_test = load_and_diagnose()

    # Build pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    LinearSVC(dual=False, class_weight='balanced', verbose=2))
    ])

    # Train once
    print("\n=== Fitting SVM on train data ===")
    pipe.fit(X_train, y_train)

    # Evaluate
    print("\n=== Evaluating on test data ===")
    preds = pipe.predict(X_test)
    print(classification_report(y_test, preds, target_names=['left','right'], digits=3))
    print("Accuracy:", accuracy_score(y_test, preds))

    # Save final classifier
    pickle.dump(pipe, open('classifier.pickle', 'wb'))
    print("\nSaved final model to classifier.pickle")

if __name__ == '__main__':
    main()
