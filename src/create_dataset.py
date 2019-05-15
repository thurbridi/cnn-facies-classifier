import os
import scipy.io as sio
import numpy as np

if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, '../data/stanford6_truncated.mat')

    data = sio.loadmat(filename)
    seismic_cube = data['sismica_input']
    facies_cube = data['facies_output']

    image_size = 32
    n_wells = 10

    height, width, depth = seismic_cube.shape

    m = ((height - image_size + 1)
         * (width - image_size + 1)
         * (depth - image_size + 1))

    X = np.empty((m, 3, image_size, image_size))
    Y = np.empty((m, 1))
    mid_idx = int(np.median(range(image_size)))
    i = 0
    for z in range(height - image_size + 1):
        for x in range(width - image_size + 1):
            for y in range(depth - image_size + 1):
                sample = seismic_cube[
                    z:z+image_size,
                    x:x+image_size,
                    y:y+image_size,
                ]

                image = np.array([
                    sample[mid_idx, :, :],
                    sample[:, mid_idx, :],
                    sample[:, :, mid_idx],
                ])

                facies = facies_cube[
                    z:z+image_size,
                    x:x+image_size,
                    y:y+image_size,
                ][mid_idx, mid_idx, mid_idx]

                X[i] = image
                Y[i] = facies
                i = i + 1

    print(X.shape)
    print(Y.shape)
