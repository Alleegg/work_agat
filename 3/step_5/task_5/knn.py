import numpy as np


class KNN:
    """
    K-классификатор ближайших соседей, использующий потери L1
    """
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
        '''
        Вычисляет расстояние L1 от каждой выборки X до каждой обучающей выборки 
        Векторизует некоторые вычисления, поэтому используется только 1 цикл

        Аргументы:
        X, np-массив (num_test_samples, num_features) - образцы для запуска
        
        Возвращается:
        расстояния, np-массив (num_test_samples, num_train_samples) - массив
           с расстояниями между каждым тестом и каждой тренировочной выборкой
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            pass
        return dists

    def compute_distances_no_loops(self, X):
        '''
        Вычисляет расстояние L1 от каждой выборки X до каждой обучающей выборки
        Полностью векторизует вычисления с помощью numpy

        Аргументы:
        X, np-массив (num_test_samples, num_features) - образцы для запуска
        
        Возвращается:
        расстояния, np-массив (num_test_samples, num_train_samples) - массив
           с расстояниями между каждым тестом и каждой тренировочной выборкой
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        # Using float32 to to save memory - the default is float64
        dists = np.zeros((num_test, num_train), np.float32)
        # TODO: Implement computing all distances with no loops!
        pass

    def predict_labels_binary(self, dists):
        '''
        Возвращает предсказания модели для случая бинарной классификации
        
        Аргументы:
        расстояния, np-массив (num_test_samples, num_train_samples) - массив
           с расстояниями между каждым тестом и каждой тренировочной выборкой

        Возвращается:
        pred, np-массив bool (num_test_samples) - двоичные прогнозы
        для каждого тестового образца
        '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)
        for i in range(num_test):
            # TODO: Implement choosing best class based on k
            # nearest training samples
            pass
        return pred

    def predict_labels_multiclass(self, dists):
        '''
        Возвращает предсказания модели для случая многоклассовой классификации
        
        Аргументы:
        расстояния, np-массив (num_test_samples, num_train_samples) - массив
           с расстояниями между каждым тестом и каждой тренировочной выборкой

        Возвращается:
        pred, np-массив int (num_test_samples) - прогнозируемый индекс класса
        для каждой тестовой выборки
        '''
        num_test = dists.shape[0]
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.int)
        for i in range(num_test):
            # TODO: Implement choosing best class based on k
            # nearest training samples
            pass
        return pred
