import numpy as np

from keras.utils import Sequence, to_categorical


def normalize_cube(cube):
    cube -= np.min(cube)
    cube /= np.max(cube)
    return cube


class F3Sequence(Sequence):
    def __init__(self, seismic_cube, facies_cube, indexes, batch_size,
                 sample_size, shuffle=True):
        self.seismic_cube = normalize_cube(seismic_cube)
        self.facies_cube = facies_cube
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.n_classes = len(np.unique(facies_cube))
        self.shuffle = shuffle
        self.indexes = indexes
        self.mid_idx = int((self.sample_size - 1) / 2)

    def __len__(self):
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, idx):
        dim = (self.sample_size, self.sample_size)

        X = np.empty((self.batch_size, *dim, 3), dtype=float)
        y = np.empty((self.batch_size), dtype=int)

        batch_indexes = self.indexes[idx * self.batch_size:
                                     (idx + 1) * self.batch_size]
        for i, coord in enumerate(batch_indexes):
            image, label = self.get_example(coord)
            X[i, ] = image
            y[i] = label

        return X, to_categorical(y, num_classes=self.n_classes)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def get_example(self, coord):
        # samples a single example with upper left coord as reference

        x, y, z = coord
        sample = self.seismic_cube[
            x:x + self.sample_size,
            y:y + self.sample_size,
            z:z + self.sample_size,
        ]

        image = np.moveaxis(np.array([
            sample[self.mid_idx, :, :],
            sample[:, self.mid_idx, :],
            sample[:, :, self.mid_idx]
        ]), 0, -1)

        label = self.facies_cube[
            x + self.mid_idx,
            y + self.mid_idx,
            z + self.mid_idx
        ]

        return image, label


class F3PredictSequence(Sequence):
    def __init__(self, seismic_cube, indexes, batch_size, sample_size):
        self.seismic_cube = normalize_cube(seismic_cube)
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.indexes = indexes
        self.mid_idx = int((self.sample_size - 1) / 2)

    def __len__(self):
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, idx):
        dim = (self.sample_size, self.sample_size)

        X = np.empty((self.batch_size, *dim, 3), dtype=float)

        batch_indexes = self.indexes[idx * self.batch_size:
                                     (idx + 1) * self.batch_size]
        for i, coord in enumerate(batch_indexes):
            image = self.get_example(coord)
            X[i, ] = image

        return X

    def get_example(self, coord):
        # samples a single example with upper left coord as reference
        x, y, z = coord
        sample = self.seismic_cube[
            x:x + self.sample_size,
            y:y + self.sample_size,
            z:z + self.sample_size,
        ]

        image = np.moveaxis(np.array([
            sample[self.mid_idx, :, :],
            sample[:, self.mid_idx, :],
            sample[:, :, self.mid_idx]
        ]), 0, -1)

        return image
