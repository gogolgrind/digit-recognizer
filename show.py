# -*- coding: utf-8 -*-

import scipy as sp
import matplotlib.pyplot as plt
from os import path,name as os_name

def show_dig(im):
    plt.imshow(im)
    plt.gray()
    plt.show()
    
def main():
    i = 0
    base_path = ''
    if  os_name == 'nt':
        base_path = 'C:\\Users\\Gogol\\Desktop\\kaggle\\Digit Recognizer\\data'
    else:
        base_path = '/home/cvlab/Рабочий стол/kaggle/Digit Recognizer/data'
    imgs = []
    for line in open(path.join(base_path,'test-200.csv')):
        # ignore laberl
        if i > 0:
            im = sp.array([int(elem) for elem in line.strip().split(',')]).reshape((28,28))
            imgs.append(im)
        i += 1
    show_dig(imgs[59])
if __name__ == '__main__':
    main()
