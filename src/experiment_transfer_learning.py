import keras
from keras import Model
from keras.layers import GlobalAveragePooling2D, Dense

import numpy as np
import h5py
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

with h5py.File('data/processed/stanford6_32.h5', 'r') as dataset:
    x_train = np.array(dataset['train/X'])
    y_train = np.array(dataset['train/Y'])
    x_val = np.array(dataset['val/X'])
    y_val = np.array(dataset['val/Y'])
    x_test = np.array(dataset['test/X'])
    y_test = np.array(dataset['test/Y'])

classnames = {
    0: 'Floodplain',
    1: 'Pointbar',
    2: 'Channel',
    3: 'Boundary',
}

m = x_train.shape[0]
num_classes = 4

mobilenet_v2 = keras.applications.MobileNetV2(
    input_shape=(32, 32, 3), include_top=False)

temp = mobilenet_v2.output
temp = GlobalAveragePooling2D()(temp)
temp = Dense(256, activation='relu')(temp)
out = Dense(4, activation='softmax')(temp)

model = Model(inputs=mobilenet_v2.input, outputs=out)


for layer in mobilenet_v2.layers[:-2]:
    layer.trainable = False

model.summary()

opt = keras.optimizers.Adam(
    lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

batch_size = 32
epochs = 10

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=epochs,
    shuffle=True,
)

model.save('models/mobilenetv2_32.h5')
print('Model saved!')


print('Evaluating model...\n')
y_pred = np.argmax(model.predict(x_test), axis=-1)
y_true = np.argmax(y_test, axis=-1)

precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

matrix = confusion_matrix(y_true, y_pred)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

print(f'Precision: \t{precision}')
print(f'Recall: \t{recall}')
print(f'F1-Score: \t{f1}')
