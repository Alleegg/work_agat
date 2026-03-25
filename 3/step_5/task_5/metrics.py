import numpy as np

def binary_classification_metrics(prediction, ground_truth):

    if len(prediction) == len(ground_truth):

        #main == np.zeros((2, 2), np.int_)
        TP, TN, FN, FP = 0,0,0,0
        for pred in range(prediction):
            for truth in range(ground_truth):
                if pred == 0 and truth == 0:
                    TN += 1
                elif pred == 1 and truth == 1:
                    TP += 1
                elif pred == 0 and truth == 1:
                    FN += 1
                elif pred == 1 and truth == 0:
                    FP += 1
                
        # main = np.array([TP, FN],
        #                 [FP, TN])
        '''
        Аргументы:
        prediction, np-массив значений bool (num_samples) - предсказания модели
        ground_truth, np-массив значений bool (num_samples) - истинные метки
        '''
        precision = 0
        recall = 0
        accuracy = 0
        f1 = 0

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)

        return precision, recall, f1, accuracy

    else:
        raise ValueError('')


def multiclass_accuracy(prediction, ground_truth):

    '''
    Аргументы:
    prediction, np-массив int (num_samples) - модельные предсказания
    ground_truth, np-массив int (num_samples) - истинные метки
    '''

    if len(prediction) == len(ground_truth):
        general = len(prediction)

        count_true = 0
        for pred in range(prediction):
            for truth in range(ground_truth):
                if pred == truth: count_true += 1

        accuracy = count_true / general

        return accuracy

    else:
        raise ValueError('')