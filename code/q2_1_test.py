from q2_1 import *
import data
import numpy as np
import q2_0
import time

def kFold_test():
    train_data, train_labels, test_data, test_labels = data.load_all_data('../a2digits')
    print (k_fold(train_data,train_labels,10))

def query_test():

    train_data, train_labels, test_data, test_labels = data.load_all_data('../a2digits')
    #print (test_labels)
    knn = KNearestNeighbor(train_data, train_labels)
    for i in range(test_data.shape[0]):

        predicted_label = knn.query_knn(test_data[i],14)#test optimal 14
        print (predicted_label)
        q2_0.visualize(test_data[i].reshape([1,-1]))

def main ():
    while 1:

        functions={'kFold':kFold_test,'query':query_test}

        functions[input('Please input the test function name (kFold for kFold_test): ')]()

if __name__ == '__main__':
    main();