import numpy as np


def class_frequency(y, n_classes):
    empty = {_class: 0 for _class in range(n_classes)}

    unique, counts = np.unique(y, return_counts=True)
    frequency = dict(zip(unique, counts))

    return {**empty, **frequency}
