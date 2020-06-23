import keras
import argparse
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np
from sklearn.model_selection import train_test_split
from src.sequences import F3Sequence
from sklearn.utils.class_weight import compute_class_weight


def split_indexes(cube, sample_size):
    shape = cube.shape

    indexes = [
        (iline, xline, depth)
        for iline in range(shape[0] - sample_size + 1)
        for xline in range(shape[1] - sample_size + 1)
        for depth in range(shape[2] - sample_size + 1)
    ]

    train_indexes, val_indexes = train_test_split(
        indexes, test_size=0.2, shuffle=True)
    return train_indexes, val_indexes


def train(sample_size, model_name):
    seismic_cube = np.load('data/raw/train_seismic.npy')
    facies_cube = np.load('data/raw/train_labels.npy')

    class_weights = compute_class_weight(
        'balanced', np.unique(facies_cube), facies_cube.flatten())

    batch_size = 1000
    input_shape = (sample_size, sample_size, 3)
    n_classes = len(np.unique(facies_cube))

    model = Sequential()

    model.add(Conv2D(32, (7, 7), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (5, 5), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(200))
    model.add(Activation('relu'))

    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    opt = keras.optimizers.Adam(
        lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    train_indexes, val_indexes = split_indexes(seismic_cube, sample_size)

    training_sequence = F3Sequence(
        seismic_cube, facies_cube, train_indexes, batch_size, sample_size)
    validation_sequence = F3Sequence(
        seismic_cube, facies_cube, val_indexes, batch_size, sample_size)

    history = model.fit_generator(
        generator=training_sequence,
        validation_data=validation_sequence,
        class_weight=class_weights,
        use_multiprocessing=False,
        epochs=20,
        workers=8,
        verbose=1,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, min_delta=0.01, restore_best_weights=True)],
    )

    # trained-model-f3-48-full
    model.save(f'models/{model_name}')
    print('Model saved!')

    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'sample_size',
        type=int,
        help='size of dataset images (image_size x image_size)',
        default=32
    )

    parser.add_argument(
        'filename',
        type=str,
        help='filename of the model when saving',
        default='last_f3.h5'
    )

    args = parser.parse_args()

    train(args.sample_size, args.filename)
