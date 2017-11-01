from q2_2 import *
import data
import numpy as np


def mean_test():
    train_data, train_labels, _,_ = data.load_all_data('../a2digits')
    #print (train_data.shape)
    print (compute_mean_mles(deShuffle(train_data, train_labels)).shape)
    pass;

def covariance_test():
    train_data, train_labels, _,_ = data.load_all_data('../a2digits')
    #print (train_data.shape)
    print (compute_sigma_mles(deShuffle(train_data, train_labels),compute_mean_mles(deShuffle(train_data, train_labels))))
    pass;

def main():

    while 1:

        functions={'cov':covariance_test,'mean':mean_test}

        functions[input('Please input the test function name (cov, mean): ')]()

if __name__ == "__main__":
    main()