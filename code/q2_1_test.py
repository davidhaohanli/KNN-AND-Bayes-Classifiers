from q2_1 import *
import data
import numpy as np


def kFold_test():
    train_data, train_labels, test_data, test_labels = data.load_all_data('../a2digits')
    print (k_fold(train_data,train_labels,10))

def main ():
    while 1:

        functions={'kFold':kFold_test}

        functions[input('Please input the test function name (kFold for kFold_test): ')]()

if __name__ == '__main__':
    main();