import numpy as np
import pickle
import os
import sys
import json
import pdb, random
import time
from PIL import Image
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
def main():
    paths = pickle.load(open('train_test_paths.pickle', 'rb'))
    all_features = pickle.load(open('stage_1_features.pickle', 'rb'))
    train_paths = paths['train_imgs']
    test_paths = paths['test_imgs']
    train_features = np.stack([all_features[tp] for tp in train_paths if tp in all_features], axis=0)
    train_labels = ['left' if '/left/' in tp else 'right' if '/right/' in tp else sys.exit(-1) for tp in train_paths if tp in all_features]
    test_features = np.stack([all_features[tp] for tp in test_paths if tp in all_features], axis=0)
    test_labels = ['left' if '/left/' in tp else 'right' if '/right/' in tp else sys.exit(-1) for tp in test_paths if tp in all_features]
    if not os.path.isfile('classifier.pickle'):
        pipe_lrSVC = Pipeline([('scaler', StandardScaler()), ('clf', LinearSVC(dual=False, class_weight='balanced', verbose=2))])
        pipe_lrSVC.fit(train_features, train_labels)
        predictions = pipe_lrSVC.predict(test_features)
        print(classification_report(y_true=test_labels, y_pred=predictions))
        pickle.dump(pipe_lrSVC, open('classifier.pickle', 'wb'))
    else:
        pipe_lrSVC = pickle.load(open('classifier.pickle', 'rb'))
        predictions = pipe_lrSVC.predict(test_features)
        predictions = ['left' if 'left' in p else 'right' for p in predictions]
        print(classification_report(test_labels, predictions, target_names=['left', 'right'], digits=3))
        print('Accuracy ={}\n'.format(accuracy_score(test_labels, predictions)))
if __name__ == '__main__':
    main()
