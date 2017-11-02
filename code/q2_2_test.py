from q2_2 import *
import data
import os
import numpy as np


def mean_test():
    train_data, train_labels, _,_ = data.load_all_data('../a2digits')
    #print (train_data.shape)
    mean=compute_mean_mles(deShuffle(train_data, train_labels));
    print (mean);
    os.system('touch ../temp_data/mean')
    np.savetxt('../temp_data/mean',mean,delimiter=',')
    return mean

    pass;

def covariance_test():
    train_data, train_labels, _,_ = data.load_all_data()
    #print (train_data.shape)
    cov=compute_sigma_mles(deShuffle(train_data, train_labels),compute_mean_mles(deShuffle(train_data, train_labels)));
    print (cov);
    os.system('touch ../temp_data/cov')
    np.savetxt('../temp_data/cov', cov.reshape((10,-1)),delimiter=',')
    return cov
    pass;

def load_mean_cov():
    if not os.path.isfile('../temp_data/mean'):
        print ('calculating mean')
        mean = mean_test();
    else:
        print ('loading mean')
        mean = np.loadtxt('../temp_data/mean', delimiter=',')
    if not os.path.isfile('../temp_data/cov'):
        print ('calculating cov')
        cov = covariance_test();
    else:
        print ('loading cov')
        cov = np.loadtxt('../temp_data/cov', delimiter=',').reshape((10,64,64))

    return mean,cov

def read_and_plot_test():
    mean,cov=load_mean_cov();
    plot_cov_diagonal(cov)
    pass;

def con_hd():
    train_data, train_labels, _, _ = data.load_all_data()
    mean, cov = load_mean_cov();
    con_likelihood = conditional_likelihood(train_data, mean, cov);
    print('Average conditional likelihood: ', con_likelihood.mean(axis=0))

def main():

    while 1:

        functions={'cov':covariance_test,'mean':mean_test,'plot':read_and_plot_test,\
                   'con_hd':con_hd}

        functions[input('Please input the test function name (cov, mean): ')]()

if __name__ == "__main__":
    main()