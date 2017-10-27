'''
Question 2.0 Skeleton Code

Here you should load the data and plot
the means for each of the digit classes.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
'''
def plot_means(train_data, train_labels):
    means = []
    for i in range(0, 10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        # Compute mean of class i

    # Plot all means on same axis
    all_concat = np.concatenate(means, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()
'''
def visualize(X,features=np.array(['Just One Dimension'])):
    plt.figure(figsize=(20, 5))
    feature_count = features.shape[0]

    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        plt.imshow(X[i].reshape((8,8)),cmap='gray')
        #plt.xlabel(features[i])
        # plt.ylabel('target y')
        # TODO: Plot feature i against y

    plt.tight_layout()
    plt.show()

def main ():
    train_data,_, _, _=data.load_all_data_from_zip('../a2digits.zip','../',shuffle=False);
    #print (train_data.shape,train_labels.shape,test_data.shape,test_labels.shape)
    step=700;
    mean=np.zeros((10,64));
    for i in range(10):
        mean[i]=np.mean(train_data[i*step:(i+1)*step],axis=0);
    #print (mean.shape)
    visualize(mean,np.arange(0,10))

if __name__ == '__main__':
    main()
