import numpy as np


class KNN:
    def __init__(self, k=1):
        self.k = k


    def fit(self, X, y):
        self.train_X = X
        self.train_y = y


    def predict(self, X, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)


    def compute_distances_two_loops(self, X):

        num_train = self.train_X.shape[0]
        num_test = X.shape[0]

        dists = np.zeros((num_test, num_train), np.float32)

        for i_test in range(num_test):
            for i_train in range(num_train):
                dists[i_test, i_train] = np.sum(np.abs(X[i_test] - self.train_X[i_train]))

        return dists
                


    def compute_distances_one_loop(self, X):

        num_train = self.train_X.shape[0]
        num_test = X.shape[0]

        dists = np.zeros((num_test, num_train), np.float32)

        for i_test in range(num_test):
            
            dists[i_test, :] = np.sum(np.abs(X[i_test] - self.train_X), axis= 1) # не уверен в этом
        return dists

    def compute_distances_no_loops(self, X):

        num_train = self.train_X.shape[0]
        num_test = X.shape[0]

        dists = np.zeros((num_test, num_train), np.float32)
        #честно говорю подсмотрел в инете, долго голову ломал.... очень....
        dists = np.sum(np.abs(X[:, np.newaxis, :] - self.train_X[np.newaxis, :, :]), axis=2)
        return dists


    def predict_labels_binary(self, dists):
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)
        for i in range(num_test):
            pred[i] = np.sum(self.train_y[
                np.argsort(dists[i])[:self.k]
                ]) > (self.k / 2)
        return pred

    def predict_labels_multiclass(self, dists):
        
        num_test = dists.shape[0]
        #pred = np.zeros(num_test, np.int)
        pred = np.zeros(num_test, dtype= int)

        for i in range(num_test):
            labl = self.train_y[
                np.argsort(dists[i])[:self.k]
                ]
            pred[i] = np.argmax(np.bincount(labl))
        return pred
