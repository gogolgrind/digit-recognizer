# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 22:03:30 2015
@author: k.sozykin
based on : http://lenguyenthedat.com/minimal-data-science-3-mnist-neuralnet/
           https://github.com/lenguyenthedat/kaggle-for-fun/blob/master/digit-recognizer/digit-recognizer.py
           https://www.kaggle.com/cyberzhg/digit-recognizer/sklearn-pca-svm/files
"""
from __future__ import print_function
import pandas as pd
import scipy as sp
import csv
from time import time
import sklearn.metrics
from os import path,makedirs,name as os_name
from sklearn.ensemble import RandomForestClassifier  
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler,RobustScaler

# params
sample = 0
scale = 0
is_win = os_name == 'nt'
pd.options.mode.chained_assignment = None
is_pca = 0
target = 'Label' if sample else 'label'

if not is_win:
    from sknn import mlp
    
from datetime import datetime

# disable warnings 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():
    # reading data
    base_path = ''
    if is_win:
        base_path = 'C:\\Users\\Gogol\\Desktop\\kaggle\\Digit Recognizer\\data'
    else:
        base_path = '/home/cvlab/Рабочий стол/kaggle/Digit Recognizer/data'
    print('Data reading starts')
    if not sample:
        train = pd.read_csv(path.join(base_path,'train.csv'))
        test = pd.read_csv(path.join(base_path,'test.csv'))
    else:
        data = pd.read_csv(path.join(base_path,'train-1000.csv'))
        # split data into test and train data 25 % and 75 %
    if sample:
        data['is_train'] = sp.random.uniform(0, 1, len(data)) <= 0.75
        train = data[data['is_train'] == True]
        test  = data[data['is_train'] == False]
    print('Data reading ends')
    # get_features name
    features = test.columns.tolist()

    if sample:
        features.remove('is_train')
        features.remove(target)
        pass
    # scaling. Dont use it with SVM
    if scale:
        print('Data scaling starts')
        # StandardScaler get warnings. Why ?
        scaler = RobustScaler()
        for col in features:
            scaler.fit(list(train[col])+list(test[col]))
            train[col] = scaler.transform(train[col])
            test[col] = scaler.transform(test[col])
        print('Data scaling ends\n \n')
    # Create array of classifiers
    classifiers = []
    classifiers.append(RandomForestClassifier(n_estimators=256, max_depth=64))
#    classifiers.append(SVC(kernel ='rbf',C = 2))
#    classifiers.append(mlp.Classifier(
#            layers=[
#                mlp.Layer('Rectifier', units=100),
#                mlp.Layer("Tanh", units=100),
#                mlp.Layer("Sigmoid", units=100),
#                mlp.Layer('Softmax')],
#            learning_rate=0.001,
#            learning_rule='momentum',
#            learning_momentum=0.9,
#            batch_size=100,
#            valid_size=0.01,
#            n_stable=20,
#            n_iter=200,
#            verbose=True))

#    classifiers.append(mlp.Classifier(
#                layers=[
#                    mlp.Convolution("Rectifier", channels=16, kernel_shape=(5,5)),
#                    mlp.Convolution("Rectifier", channels=8, kernel_shape=(2,2)),
#                    mlp.Convolution("Rectifier", channels=8, kernel_shape=(2,2)),
#                    mlp.Layer('Rectifier', units=100),
#                    mlp.Layer('Softmax')],
#                learning_rate=0.001,
#                learning_rule='momentum',
#                learning_momentum=0.9,
#                batch_size=100,
#                valid_size=0.01,
#                n_stable=20,
#                n_iter=50,
#                verbose=True))
            
   
    # Train
    cnt = 0
    pca = PCA(n_components=35,whiten=True)
    for classifier in classifiers:
        class_name = classifier.__class__.__name__
        print('Training with %s  starts' % class_name)
        st = time()
        labels = train[target]
        train_data = sp.array(train[list(features)])
        if class_name == 'SVC' or is_pca:
            train_data = pca.fit_transform(train_data)
        classifier.fit(train_data,labels)
        fn = time()
        print (class_name,'with training time:',fn-st)
        print('Training with %s ends' % class_name)
        print('Eval with %s starts' % class_name)
        a_val = 0
        test_data = sp.array(test[features])
        if class_name == 'SVC' or is_pca:
            test_data = pca.transform(test_data)
        pred  = classifier.predict(test_data)
        if sample:
            a_val =  sklearn.metrics.accuracy_score(test[target].values,pred)
            print (classifier.__class__.__name__,'Accuracy Score:',a_val)
        else:
            res_path = ''
            if is_win:
                res_path = 'C:\\Users\\Gogol\\Desktop\\kaggle\\Digit Recognizer\\result'
            else:
                res_path = '/home/cvlab/Рабочий стол/kaggle/Digit Recognizer/result'
            if not path.exists(res_path):
                makedirs(res_path)
            classifier_name = classifier.__class__.__name__
            if classifier_name == 'classifier':
                classifier_name = 'DNN_'+ classifier_name
            classifier_name = str(cnt) + '_' + classifier_name + '_' +  datetime.now().strftime("%Y-%m-%d_%H-%M")
            csvfile = path.join(res_path,classifier_name + '_' + 'submit.csv')
            with open(csvfile,'w') as out:
                writer = csv.writer(out,lineterminator='\n')
                writer.writerow(['ImageId',target])
                m = len(pred)
                for i in range(m):
                    pred_val = pred[i] if type(pred[i]) != sp.ndarray else pred[i][0]
                    writer.writerow([i+1,pred_val])
        print('Eval ends\n')
        cnt += 1
if __name__ == '__main__':
    main()
