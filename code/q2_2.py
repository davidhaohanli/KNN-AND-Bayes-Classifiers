'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
import q2_0
#import scipy.linalg as al
#import matplotlib.pyplot as plt

labels=np.arange(10)

def compute_mean_mles(train_data):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    for label in range(10):
        means[label]=np.mean(train_data[label],axis=0)
    # Compute means
    return means

def compute_sigma_mles(train_data,means):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    cov=np.zeros((10,64,64))

    for label in range(10):
        this_data = train_data[label]-means[label]
        cov[label] = this_data.T.dot(this_data)/this_data.shape[0]+0.01*np.identity(64)
    return cov

def deShuffle(train_data, train_labels,shuffled=True):
    # data label sorting (de-shuffle), use linear sort algorithm (counting sort)
    if shuffled:
        data_clean = np.zeros((10, train_data.shape[0]//10, 64));
        cur = np.zeros(10);
        for i in range(train_data.shape[0]):
            # print (train_labels[i])
            label = int(train_labels[i])
            data_clean[label,int(cur[label])] = train_data[i];
            cur[label] += 1;
    else:
        return train_data.reshape((10,-1,64));
    return data_clean;

def plot_cov_diagonal(covariances):
    # Plot the diagonal of each covariance matrix side by side
    cov_diag = np.zeros((10, 64))
    for i in range(10):
        cov_diag[i] = np.diag(covariances[i])
        # ...
    q2_0.visualize(cov_diag, np.arange(10), timerSet=False)

def comp_logZ(cov):
    logZ=np.zeros(10)
    for i in range(10):
        #print(cov[i].shape)
        logZ[i] = np.log(np.sqrt(((2*np.pi)**64)*np.linalg.det(cov[i])));
    return logZ

def original_likelihood(digits, means, covariances):
    likelihood = np.zeros((digits.shape[0],10))
    for label in range(10):
        x_minus_mean = digits-means[label];
        for n,data in enumerate(x_minus_mean):
            likelihood[n][label] = (2*np.pi)**(-64/2)*np.dot(np.linalg.det(covariances[label]),\
                                    np.exp(-1/2*np.dot(np.dot(data.T,np.linalg.inv(covariances[label])),data)))
    return likelihood/likelihood.sum(axis=1).reshape((-1,1));

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    return np.log(original_likelihood(digits,means,covariances))
    gen_likelihood=np.zeros((digits.shape[0],10))
    for i in range(10):
        x_minus_mean=digits-means[i]
        for n,data in enumerate(x_minus_mean):
            gen_likelihood[n][i] = -(logZ[i]+0.5*np.dot(np.dot(data.T,np.linalg.inv(covariances[i])),data))
    return gen_likelihood

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    #logZ=comp_logZ(covariances);
    return generative_likelihood(digits,means,covariances)+np.log(1/10);

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''

    cond_hd = conditional_likelihood(digits,means,covariances)
    return cond_hd[np.arange(digits.shape[0]),np.vectorize(int)(labels)].sum()/digits.shape[0],cond_hd

def classify_data(digits, means, covariances,con_hd=False):
    '''
    Classify new points by taking the most likely posterior class
    '''

    # Compute and return the most likely class
    if con_hd is False:
        return np.argmax(conditional_likelihood(digits, means, covariances),axis=1)
    else:
        return np.argmax(con_hd,axis=1)

def accuracy(labels,digits,means,covariance,con_hd=False):

    if con_hd is False:
        return np.equal(labels,classify_data(digits,means,covariance)).mean();
    else:
        return np.equal(labels,classify_data(digits,means,covariance,con_hd)).mean();

def leading_eig(cov):

    ld_eig=np.zeros((10,64))
    for label in range(10):
        #print (label)
        w,v = np.linalg.eig(cov[label])
        ld_eig[label] = v[np.argmax(w)]
        #print (ld_eig[label])
    #print (ld_eig.shape)
    return ld_eig

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data(shuffle=False)
    # Fit the model
    data_clean=deShuffle(train_data,train_labels,shuffled=False)
    means = compute_mean_mles(data_clean)
    covariances = compute_sigma_mles(data_clean, means)
    plot_cov_diagonal(covariances)
    q2_0.visualize(leading_eig(covariances),labels)
    train_hd=avg_conditional_likelihood(train_data, train_labels, means, covariances)
    test_hd=avg_conditional_likelihood(test_data, test_labels, means, covariances)
    print('Train_data: ')
    print('Average conditional likelihood for train data in correct class is: ',\
          train_hd[0])
    print('\nTest_data: ')
    print('Average conditional likelihood for test data in correct class is: ', \
          test_hd[0])
    print('\nThe accuracy for train data is: ',accuracy(train_labels, train_data, means, covariances,train_hd[1]));
    print('The accuracy for test data is: ',accuracy(test_labels, test_data, means, covariances,test_hd[1]));

    # Evaluation

if __name__ == '__main__':
    main()