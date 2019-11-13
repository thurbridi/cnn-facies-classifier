import os
import argparse
import numpy as np
import keras
from src import F3PredictSequence


def pred_indexes(cube, sample_size):
    shape = cube.shape

    indexes = [
        (iline, xline, depth)
        for iline in range(shape[0] - sample_size + 1)
        for xline in range(shape[1] - sample_size + 1)
        for depth in range(shape[2] - sample_size + 1)
    ]

    return indexes


def pred_sequence(input_name, out_name, model_name, sample_size):
    dirname = os.path.dirname(__file__)
    seismic_cube = np.load(os.path.join(
        dirname, '../data/raw/', input_name)
    )

    model = keras.models.load_model(os.path.join(
        dirname, '../models/', model_name))

    indexes = pred_indexes(seismic_cube, sample_size)
    batch_size = 1000
    sequence = F3PredictSequence(
        seismic_cube, indexes, batch_size, sample_size)

    pred_list = np.argmax(model.predict_generator(
        sequence, workers=4, verbose=1), axis=-1)

    shape = seismic_cube.shape
    x_range = shape[0] - sample_size + 1
    y_range = shape[1] - sample_size + 1
    z_range = shape[2] - sample_size + 1

    pred_cube = np.zeros(shape=(x_range, y_range, z_range), dtype=np.int8)

    for i, pred in enumerate(pred_list):
        pred_cube[indexes[i]] = pred

    print(f'Cube saved as: /models/{out_name}')
    np.save(os.path.join(
        dirname, f'../models/{out_name}'),
        pred_cube
    )


def pred_volume(filename):
    dirname = os.path.dirname(__file__)

    seismic_cube = np.load(os.path.join(
        dirname, '../data/raw/', filename)
    )

    (iline, xline, depth) = seismic_cube.shape
    sample_size = 32
    mid_idx = int((sample_size - 1)/2)

    x_range = iline - sample_size + 1
    y_range = xline - sample_size + 1
    z_range = depth - sample_size + 1
    print(f'Predicting {x_range * y_range * z_range} points')

    seismic_cube -= np.min(seismic_cube)
    seismic_cube /= np.max(seismic_cube)

    predict = np.zeros(shape=(x_range, y_range, z_range), dtype=np.int8)

    model = keras.models.load_model(os.path.join(
        dirname, '../models/', model_name))
    model.summary()

    print(f'Creating prediction cube...')
    for x in range(x_range):
        for y in range(y_range):
            for z in range(z_range):
                sample = seismic_cube[
                    x:x+sample_size,
                    y:y+sample_size,
                    z:z+sample_size,
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

                predict[x, y, z] = np.argmax(
                    model.predict(np.array([image])), axis=-1)

    for volume_id in ['train', 'test1', 'test2']:
        if volume_id in filename:
            out_name = f'f3-{volume_id}-pred.npy'

    print(f'Cube saved as: models/{out_name}')
    np.save(os.path.join(
        dirname, f'../models/{out_name}'),
        predict
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'modelname',
        type=str,
        help='filename of the model to use for prediction'
    )

    parser.add_argument(
        'image_size',
        type=int,
        help='image size (n) used by the model (n, n, 3)'
    )

    parser.add_argument(
        'input',
        type=str,
        help='filename of the seismic volume you want to predict'
    )

    parser.add_argument(
        'out',
        type=str,
        help='filename of the output volume'
    )

    args = parser.parse_args()

    input_name = args.input
    output_name = args.out
    model_name = args.modelname
    image_size = args.image_size

    pred_sequence(input_name, output_name, model_name, image_size)
