'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
import q2_0
#import scipy.linalg as al
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

labels=np.arange(10)

def compute_mean_mles(train_data):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    for label in range(10):
        for i in range(64):
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
        for sample in range(train_data.shape[1]):
            train_data[label][sample]-=means[label];
    for label in range(10):
        #print ('label: ',label)
        for x in range(64):
            for y in range(64):
                #print ('dimension',x,y)
                for sample1 in range(train_data.shape[1]):
                        cov[label][x][y]+=train_data[label][sample][x]*train_data[label][sample][y];

    # Compute covariances
    cov/=(train_data.shape[1]**2);
    for i in range(10):
        cov[i]+=0.01*np.identity(64);
    return cov

def deShuffle(train_data, train_labels):
    # data label sorting (de-shuffle), use linear sort algorithm (counting sort)
    data_clean = np.zeros((10, train_data.shape[0] // 10, 64));
    cur = np.zeros(10);
    for i in range(train_data.shape[0]):
        # print (train_labels[i])
        label = int(train_labels[i])
        data_clean[label][int(cur[label])] = train_data[i];
        cur[label] += 1;
    return data_clean;

def plot_cov_diagonal(covariances):
    # Plot the diagonal of each covariance matrix side by side
    cov_diag = np.zeros((10, 64))
    for i in range(10):
        cov_diag[i] = np.diag(covariances[i])
        # ...
    q2_0.visualize(cov_diag, np.arange(10), False)

def comp_logZ(cov):
    logZ=np.zeros(10)
    for i in range(10):
        #print(cov[i].shape)
        logZ[i] = np.log(np.sqrt(((2*np.pi)**64)*np.linalg.det(cov[i])));
    return logZ

def generative_likelihood(digits, means, covariances,logZ):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
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
    logZ=comp_logZ(covariances);
    return generative_likelihood(digits,means,covariances,logZ)+1/10;

def avg_conditional_likelihood(digits, labels, means, covariances,stem):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    for i in labels:
        print('Average conditional likelihood for '+stem+'data in class',i,': ', cond_likelihood[i].mean())
    # Compute as described above and return
    return None

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''

    # Compute and return the most likely class
    return np.argmax(conditional_likelihood(digits, means, covariances),axis=1)


def accuracy(labels,digits,means,covariance):
    return np.equal(labels,classify_data(digits,means,covariance)).mean();

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data()
    # Fit the model
    data_clean=deShuffle(train_data,train_labels)
    means = compute_mean_mles(data_clean)
    covariances = compute_sigma_mles(data_clean, means)
    plot_cov_diagonal(covariances)
    print('Train_data: ')
    avg_conditional_likelihood(train_data, labels, means, covariances,data.TRAIN_STEM)
    print('\nTest_data: ')
    avg_conditional_likelihood(test_data, labels,means, covariances, data.TEST_STEM)
    print('The accuracy for train data is: ',accuracy(train_labels, train_data, means, covariances));
    print('The accuracy for test data is: ',accuracy(test_labels, test_data, means, covariances));

    # Evaluation

if __name__ == '__main__':
    main()