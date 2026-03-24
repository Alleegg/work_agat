import numpy as np
import scipy.io as io
import os


def load_data_mat(filename, max_samples, seed=42):
    '''
    Загружает массивы numpy из файла .mat

    Возвращается:
    X, np-массив (num_samples, 32, 32, 3) - изображения
    y, np-массив из int (num_samples) - меток
    '''
    raw = io.loadmat(filename)
    X = raw['X']  # nd.array of [32, 32, 3, n_samples]
    y = raw['y']  # nd.array of [n_samples, 1]
    X = np.moveaxis(X, [3], [0])
    y = y.flatten()
    # измените значение класса 0 на 0
    y[y == 10] = 0

    np.random.seed(seed)
    samples = np.random.choice(np.arange(X.shape[0]),
                               max_samples,
                               replace=False)
    
    return X[samples].astype(np.float32), y[samples]


def load_svhn(folder, max_train, max_test):
    '''
    Загружает набор данных SVHN из файла

    Аргументы:


    Возвращает:
    train_X, np array (num_train, 32, 32, 3) — обучающие изображения
    train_y, np array of int (num_train) — обучающие метки
    test_X, np array (num_test, 32, 32, 3) — тестовые изображения
    test_y, np array of int (num_test) — тестовые метки
    '''
    train_X, train_y = load_data_mat(os.path.join(folder, "train_32x32.mat"), max_train)
    test_X, test_y = load_data_mat(os.path.join(folder, "test_32x32.mat"), max_test)
    return train_X, train_y, test_X, test_y


def random_split_train_val(X, y, num_val, seed=42):
    '''
Случайным образом разбивает набор данных на обучающие и валидационные
    
    Аргументы:
    X - np-массив с выборками
    y - np-массив с метками
    num_val - количество выборок, которые нужно поместить в исходное значение проверки
- случайное начальное значение

    Возвращается:
    train_X, np-массив (num_train, 32, 32, 3) - обучающие изображения
    train_y, np-массив из int (num_train) - обучающих меток
    val_X, np-массив (num_val, 32, 32, 3) - изображения для проверки
    val_y, np-массив int (num_val) - метки для проверки
    '''
    np.random.seed(seed)
    
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    train_indices = indices[:-num_val]
    train_X = X[train_indices]
    train_y = y[train_indices]

    val_indices = indices[-num_val:]
    val_X = X[val_indices]
    val_y = y[val_indices]

    return train_X, train_y, val_X, val_y
    
    
