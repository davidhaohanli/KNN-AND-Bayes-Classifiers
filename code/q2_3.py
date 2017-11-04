'''
Question 2.3 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
import q2_2
import q2_0
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

labels=np.arange(10)

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    eta = np.zeros((10, 64))
    for label in range(train_data.shape[0]):
        for dim in range(64):
            eta[label][dim]=(train_data[label][:][dim].sum()+1)/(train_data.shape[1]+2)
    return eta

def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    q2_0.visualize(class_images,np.arange(10))
        # ...

def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    generated_data = np.zeros((10, 64))
    plot_images(generated_data)

def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array 
    '''
    gen_hd=np.zeros((bin_digits.shape[0],10));
    for label in range(10):
        for n in range(bin_digits.shape[0]):
            gen_hd[n][label]= (bin_digits[n]*np.log(eta[label])+(1-bin_digits[n])*np.log(1-eta[label])).sum()
    return gen_hd

def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    return generative_likelihood(bin_digits,eta)+np.log(1/10);

def avg_conditional_likelihood(bin_digits,labels, eta,stem):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    for i in labels:
        print('Average conditional likelihood for '+stem+'data in class',i,': ', cond_likelihood[i].mean())
    # Compute as described above and return
    return None

def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    return np.argmax(conditional_likelihood(bin_digits, eta),axis=1)
    # Compute and return the most likely class

def accuracy(labels,bin_digits,eta):
    return np.equal(labels, classify_data(bin_digits,eta)).mean();


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data(shuffle=False)
    data_clean = q2_2.deShuffle(binarize_data(train_data),train_labels,shuffled=False)
    # Fit the model
    eta= compute_parameters(data_clean)
    #print (np.log(eta))
    #print (eta.shape)
    # Evaluation
    plot_images(eta)

    print('Train_data: ')
    avg_conditional_likelihood(data_clean, labels, eta, data.TRAIN_STEM)
    print('\nTest_data: ')
    avg_conditional_likelihood(data_clean, labels, eta, data.TEST_STEM)
    print('\nThe accuracy for train data is: ',accuracy(train_labels, train_data, eta));
    print('The accuracy for test data is: ',accuracy(test_labels, test_data,eta));

    generate_new_data(eta)

if __name__ == '__main__':
    main()
