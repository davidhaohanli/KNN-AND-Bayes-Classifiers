'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
#Import pyplot - plt.imshow is useful!
#import matplotlib.pyplot as plt
#from collections import Counter

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        dist=self.l2_distance(test_point)
        ind=dist.argsort()#find the kth smallest ind of dist
        hashmap={'Dummy':[0,float('inf')]};
        res='Dummy'
        #find the kth smallest dist and count for the most frequently counted one
        #if die happens, then pull out the one with least dist sum
        for i in range(k):
            if self.train_labels[ind[i]] not in hashmap:
                hashmap[self.train_labels[ind[i]]]=[1,dist[ind[i]]];
            else:
                hashmap[self.train_labels[ind[i]]][0]+=1;
                hashmap[self.train_labels[ind[i]]][0] += dist[ind[i]];
            if hashmap[res][0] < hashmap[self.train_labels[ind[i]]][0]:
                res=self.train_labels[ind[i]];
                continue
            if hashmap[res][0] == hashmap[self.train_labels[ind[i]]][0]:
                if hashmap[res][1]>hashmap[self.train_labels[ind[i]]][1]:
                    res=self.train_labels[ind[i]]
        return res

def cross_validation(knn, k_range=np.arange(1,16)):
    res=['Dummy',float('-inf')]
    for k in k_range:
        # Loop over folds
        # Evaluate k-NN
        # ...
        accuracy=k_fold(knn.train_data, knn.train_labels, k);
        if accuracy > res[1]:
            res=[k,accuracy];
    return res;

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    res=np.zeros(eval_data.shape[0]);
    for i in range(eval_data.shape[0]):
        res[i] = knn.query_knn(eval_data[i],k)==eval_labels[i];
    return res.mean();

def k_fold(data,label,k,nFold=10):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           K in the number of folds
    '''
    ## TODO

    pieceLen = data.shape[0] // nFold;
    ran=np.arange(data.shape[0]);
    accuracies=np.zeros(nFold);
    for i in range(nFold):
        train_piece=np.concatenate((ran[0:i*pieceLen],ran[(i+1)*pieceLen:-1]));
        #print (train_piece)
        data_train=data[train_piece];
        label_train=label[train_piece];
        data_test=data[i*pieceLen:(i+1)*pieceLen];
        label_test=label[i*pieceLen:(i+1)*pieceLen];
        knn=KNearestNeighbor(data_train,label_train);
        accuracies[i]=classification_accuracy(knn,k,data_test,label_test);
    return accuracies.mean();



def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('../a2digits')
    #print (test_labels)
    knn = KNearestNeighbor(train_data, train_labels)

    print ('Accuracy for train data with k=1: ',classification_accuracy(knn,1,train_data,train_labels),\
           '\nAccuracy for train data with k=15: ',classification_accuracy(knn,15,train_data,train_labels), \
           '\nAccuracy for test data with k=1: ',classification_accuracy(knn,1,test_data,test_labels),\
           '\nAccuracy for test data with k=15: ',classification_accuracy(knn,15,test_data,test_labels))

    [k,loss]=cross_validation(knn)
    print ('Optimal K for KNN and the corresponding mean k_fold loss: ',k ,'&',loss)
    print ('Accuracy for train data with optimal k: ',classification_accuracy(knn,k,train_data,train_labels))
    print ('Accuracy for test data with optimal k: ', classification_accuracy(knn, k, test_data, test_labels))

if __name__ == '__main__':
    main()
