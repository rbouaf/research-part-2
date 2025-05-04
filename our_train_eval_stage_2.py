import os
import sys
import signal
import pickle
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

def load_data():
    paths = pickle.load(open('train_test_paths.pickle', 'rb'))
    all_feats = pickle.load(open('stage_1_features.pickle', 'rb'))
    train_paths = paths['train_imgs']
    test_paths  = paths['test_imgs']

    X_train = np.stack([all_feats[p] for p in train_paths if p in all_feats], axis=0)
    y_train = ['left' if '/left/' in p else 'right' for p in train_paths if p in all_feats]

    X_test  = np.stack([all_feats[p] for p in test_paths if p in all_feats], axis=0)
    y_test  = ['left' if '/left/' in p else 'right' for p in test_paths if p in all_feats]

    return X_train, y_train, X_test, y_test

def main():
    X_train, y_train, X_test, y_test = load_data()

    ckpt_path = 'classifier_checkpoint.pickle'
    final_path = 'classifier.pickle'
    max_iters  = 10000  # total number of liblinear iterations
    step_iters = 2000   # how many iterations per fit()

    # Build our pipeline with warm_start
    scaler = StandardScaler()
    svc    = LinearSVC(dual=False,
                       class_weight='balanced',
                       max_iter=max_iters,
                       verbose=2)
    pipe = Pipeline([('scaler', scaler), ('clf', svc)])

    # If we have a checkpoint, load it and pick up
    start_iter = 0
    if os.path.exists(ckpt_path):
        print("Loading checkpoint from", ckpt_path)
        pipe = pickle.load(open(ckpt_path, 'rb'))
        start_iter = pipe.named_steps['clf'].max_iter

    # SIGTERM handler
    def _save_and_exit(signum, frame):
        print("\nSIGTERM received—saving checkpoint at iter", 
              pipe.named_steps['clf'].max_iter, "…")
        pickle.dump(pipe, open(ckpt_path, 'wb'))
        sys.exit(0)

    signal.signal(signal.SIGTERM, _save_and_exit)

    # Now train in a loop of small fits
    for iters in range(start_iter + step_iters, max_iters + 1, step_iters):
        print(f"\n=== training up to {iters} iters ===")
        pipe.named_steps['clf'].max_iter = iters
        pipe.fit(X_train, y_train)

        # checkpoint after each chunk
        print("Saving checkpoint at iter", iters)
        pickle.dump(pipe, open(ckpt_path, 'wb'))

    # final model
    print("\n=== Training complete! ===")
    pickle.dump(pipe, open(final_path, 'wb'))
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    # evaluation
    preds = pipe.predict(X_test)
    print(classification_report(y_test, preds, target_names=['left','right'], digits=3))
    print("Accuracy:", accuracy_score(y_test, preds))

if __name__ == '__main__':
    main()
