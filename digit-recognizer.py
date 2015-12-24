# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 22:03:30 2015
@author: k.sozykin
based on : http://lenguyenthedat.com/minimal-data-science-3-mnist-neuralnet/
           https://github.com/lenguyenthedat/kaggle-for-fun/blob/master/digit-recognizer/digit-recognizer.py
"""
import pandas as pd
import scipy as sp
import csv
from time import time
from os import path,makedirs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
#from sknn import mlp
def main():
    ok = 'ok'
    # params
    pd.options.mode.chained_assignment = None
    sample = 0
    target = 'Label' if sample else 'label'
    # reading data
    base_path = 'C:\\Users\\Gogol\\Desktop\\kaggle\\Digit Recognizer\\data'
    if not sample:
        train = pd.read_csv(path.join(base_path,'train.csv'))
        test = pd.read_csv(path.join(base_path,'test.csv'))
    else:
        data = pd.read_csv(path.join(base_path,'train-200.csv'))
        # show digits first 10
        i = 0
        for line in open(path.join(base_path,'train-200.csv')):
            # ignore laberl
            if i > 0 and i < 10:
                im = sp.array([int(elem) for elem in line.strip().split(',')])[:-1].reshape((28,28))
                plt.imshow(im)
                plt.gray()
                plt.show()
            i += 1
        # split data into test and train data 25 % and 75 %
        data['is_train'] = sp.random.uniform(0, 1, len(data)) <= 0.75
        train = data[data['is_train'] == True]
        test  = data[data['is_train'] == False]
    # get_features name
    features = test.columns.tolist()
    classifiers = []
    if sample:
        features.remove('is_train')
        features.remove('Label')
    # Create array of classifiers
    classifiers = []
    classifiers.append(RandomForestClassifier(n_estimators=256, max_depth=64))
    # Train
    for classifier in classifiers:
        st = time()
        classifier.fit(sp.array(train[list(features)]),train[target])
        fn = time()
        print classifier.__class__.__name__,'with training time:',fn-st
    if sample:
        a_val =  accuracy_score(test[target].values,classifier.predict(sp.array(test[features])))
        print classifier.__class__.__name__,'Accuracy Score:',a_val
    else:
        cnt = 0
        for classifier in classifiers:
            cnt += 1
            res_path = 'C:\\Users\\Gogol\\Desktop\\kaggle\\Digit Recognizer\\result'
            if not path.exists(res_path):
                makedirs(res_path)
            csvfile = path.join(res_path,classifier.__class__.__name__ + '-'+ str(cnt) + '-submit.csv')
            print(csvfile)
            pred  = classifier.predict(sp.array(test[features]))
            with open(csvfile,'w') as out:
                writer = csv.writer(out,lineterminator='\n')
                writer.writerow(['ImageId',target])
                m = len(pred)
                for i in range(m):
                    writer.writerow([i+1,pred[i]])
if __name__ == '__main__':
    main()

