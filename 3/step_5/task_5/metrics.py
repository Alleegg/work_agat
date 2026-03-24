def binary_classification_metrics(prediction, ground_truth):
    '''
    Вычисляет показатели для бинарной классификации

    Аргументы:
    предсказание, np-массив значений bool (num_samples) - предсказания модели
    ground_truth, np-массив значений bool (num_samples) - истинные метки

    Возвращается:
    точность, отзыв, f1, accuracy - показатели классификации
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: внедряйте показатели!
    # Несколько полезных ссылок:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Вычисляет показатели для многоклассовой классификации

    Аргументы:
    предсказание, np-массив int (num_samples) - модельные предсказания
    ground_truth, np-массив int (num_samples) - истинные метки

    Возвращается:
    точность - отношение точных прогнозов к общему количеству выборок.
    '''
    # TODO: Обеспечение точности вычислений
    return 0
