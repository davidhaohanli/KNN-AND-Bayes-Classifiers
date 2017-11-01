'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
import q2_0
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

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

    data_xy=np.zeros((10,train_data.shape[1]**2,64,64))
    for label in range(10):
        for x in range(64):
            for sample1 in range(train_data.shape[1]):
                for y in range(64):
                    for sample2 in range(train_data.shape[1]):
                        data_xy[label][sample1+sample2][x][y]\
                            =train_data[label][sample1][x]-train_data[label][sample2][y];
                        #print (data_xy[label][sample1+sample2][x][y])
    # Compute covariances
    for label in range(10):
        cov[label]=np.mean(label,axis=0);
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
    for i in range(10):
        cov_diag = np.diag(covariances[i])
        # ...

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    return None

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    return None

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return
    return None

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    pass

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    data_clean=deShuffle(train_data,train_labels)
    means = compute_mean_mles(data_clean)
    covariances = compute_sigma_mles(data_clean, means)
    covDiag=np.zeros(10,64)
    for label in range(10):
        for i in range(64):
            covDiag[label][i] = covariances[label][i][i]

    q2_0.visualize(covDiag,np.arrange(10),False)

    # Evaluation

if __name__ == '__main__':
    main()