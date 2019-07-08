import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(matrix, classes,
                          title=None,
                          cmap=plt.cm.Blues):

    fig, ax = plt.subplots()
    im = ax.imshow(matrix, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(matrix.shape[1]),
           yticks=np.arange(matrix.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if matrix[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def plot_classes_freq(class_freq, classes, title=None):
    fig, ax = plt.subplots()
    n_classes = len(classes)
    idx = np.arange(n_classes)

    ax.bar(idx, class_freq.values())

    ax.set(
        title=title,
        xticks=idx,
        xticklabels=classes,
        ylabel='Examples',
    )

    fig.tight_layout()
    return ax
