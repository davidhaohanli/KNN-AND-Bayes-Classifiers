import q2_3
import data
import q2_2
import q2_0
import numpy as np

def mean_test():
    train_data, train_labels, test_data, test_labels = data.load_all_data(shuffle=False)
    data_clean = q2_2.deShuffle(q2_3.binarize_data(train_data),train_labels,shuffled=False)
    print (data_clean.shape)
    # Fit the model
    eta= q2_3.compute_parameters(data_clean)
    print (eta.shape)
    q2_3.plot_images(eta)

def classify_test():
    train_data, train_labels, test_data, test_labels = data.load_all_data(shuffle=True)
    train_data = q2_3.binarize_data(train_data);
    test_data = q2_3.binarize_data(test_data);
    data_clean = q2_2.deShuffle(train_data,train_labels,shuffled=True)
    eta = q2_3.compute_parameters(data_clean)
    predict=q2_3.classify_data(train_data,eta)
    for n in range(train_data.shape[0]):
        print ('predict: ',predict[n],'label: ',train_labels[n],' ',predict[n]==train_labels[n])
        q2_0.visualize(train_data[n].reshape(1,-1),timerSet=True);

def cond_hd_test():
    train_data, train_labels, test_data, test_labels = data.load_all_data(shuffle=True)
    train_data = q2_3.binarize_data(train_data);
    test_data = q2_3.binarize_data(test_data);
    data_clean = q2_2.deShuffle(train_data, train_labels, shuffled=True)
    eta = q2_3.compute_parameters(data_clean)
    con_hd=q2_3.conditional_likelihood(train_data, eta);
    for n in range(train_data.shape[0]):
        print ('predict: ',con_hd[n],'label: ',train_labels[n])
        q2_0.visualize(train_data[n].reshape(1,-1),timerSet=True);

def main():

    while 1:

        functions={'mean':mean_test,'classify':classify_test, 'con_hd':cond_hd_test,'q':exit}

        functions[input('Please input the test function name (classify, mean, q for exit): ')]()

if __name__ == "__main__":
    main()