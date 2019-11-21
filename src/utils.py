import numpy as np


def class_frequency(y, n_classes):
    empty = {_class: 0 for _class in range(n_classes)}

    unique, counts = np.unique(y, return_counts=True)
    frequency = dict(zip(unique, counts))

    return {**empty, **frequency}


def remove_borders(cube, image_size):
    shape = cube.shape
    half = int(image_size / 2)
    return cube[(half - 1):shape[0] - half, (half - 1):shape[1] - half, (half - 1):shape[2] - half]
