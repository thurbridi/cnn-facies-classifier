import numpy as np
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, jaccard_score)


def mean_class_accuracy(cm):
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)

    class_acc = (tp + tn) / (tp + tn + fp + fn)
    return np.mean(class_acc)


def evaluate(y_true, y_pred):
    print('Evaluating model...\n')

    matrix = confusion_matrix(y_true, y_pred)
    accuracy = mean_class_accuracy(matrix)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    fwiu = jaccard_score(y_true, y_pred, average='weighted')

    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    return {
        'mean-accuracy': accuracy,
        'precision-macro': precision,
        'recall-macro': recall,
        'f1-score-macro': f1,
        'confusion-matrix': matrix,
        'fwiu': fwiu,
    }
