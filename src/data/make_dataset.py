import os
import scipy.io as sio
import numpy as np
import h5py


def save_dataset(path, X_train, Y_train, X_test, Y_test):
    f = h5py.File(path, "w")
    f.create_dataset('test/X', data=X_test)
    f.create_dataset('test/Y', data=Y_test)
    f.create_dataset('train/X', data=X_train)
    f.create_dataset('train/Y', data=Y_train)


def sample_well_locations(n_wells, x_range, y_range):
    well_locations = []
    x_coords = np.random.choice(x_range, n_wells)
    y_coords = np.random.choice(y_range, n_wells)

    for i in range(n_wells):
        well_locations.append((x_coords[i], y_coords[i]))
    return well_locations


def main():
    seed = 42
    n_wells = 10
    image_size = 32

    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, '../../data/raw/stanford6_truncated.mat')
    dataset_path = os.path.join(
        dirname, '../../data/interim/stanford6_truncated_rgb.h5')

    data = sio.loadmat(filename)
    seismic_cube = data['sismica_input']
    facies_cube = data['facies_output']

    height, width, depth = seismic_cube.shape

    x_range = width - image_size + 1
    y_range = depth - image_size + 1
    z_range = height - image_size + 1

    np.random.seed(seed)
    well_locations = sample_well_locations(n_wells, x_range, y_range)

    m_train = n_wells * z_range
    m_test = z_range * x_range * y_range

    seismic_cube -= np.min(seismic_cube)
    seismic_cube /= np.max(seismic_cube)
    seismic_cube *= 255

    X_train = np.empty((m_train, image_size, image_size, 3), dtype='uint8')
    Y_train = np.empty((m_train, 1), dtype='int8')

    X_test = np.empty((m_test, image_size, image_size, 3), dtype='uint8')
    Y_test = np.empty((m_test, 1), dtype='int8')

    mid_idx = int(np.median(range(image_size)))

    i, j = 0, 0
    for x in range(x_range):
        for y in range(y_range):
            for z in range(z_range):
                sample = seismic_cube[
                    z:z+image_size,
                    x:x+image_size,
                    y:y+image_size,
                ]

                image = np.moveaxis(
                    np.array([
                        sample[mid_idx, :, :],  # Red    -> height
                        sample[:, mid_idx, :],  # Green  -> width
                        sample[:, :, mid_idx],  # Blue   -> depth
                    ]),
                    0,
                    -1
                )

                facies = facies_cube[
                    z:z+image_size,
                    x:x+image_size,
                    y:y+image_size,
                ][mid_idx, mid_idx, mid_idx]

                X_test[i] = image
                Y_test[i] = facies
                i += 1

                if (x, y) in well_locations:
                    X_train[j] = image
                    Y_train[j] = facies
                    j += 1

    print(f'Wells sampled: {n_wells}')
    print(f'Sampled well locations:\n{well_locations}\n')

    print(f'X_train shape: {X_train.shape}')
    print(f'Y_train shape: {Y_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'Y_test shape: {Y_test.shape}\n')

    print(f'Saving dataset as "{dataset_path}"...')
    save_dataset(
        dataset_path,
        X_train, Y_train,
        X_test, Y_test
    )
    print('Done!')


if __name__ == '__main__':
    main()
