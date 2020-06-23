import numpy as np
import h5py
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, GlobalAveragePooling2D, Conv2D, MaxPooling2D
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

input_shape = x_train.shape[1:]

model = Sequential()

model.add(Conv2D(128, (3, 3), padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(64, (3, 3), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(32, (3, 3), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))


model.add(GlobalAveragePooling2D())

model.add(Dense(256))
model.add(Activation('relu'))

model.add(Dense(256))
model.add(Activation('relu'))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt = keras.optimizers.Adam(
    lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

model.summary()

batch_size = 32
epochs = 80

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=epochs,
    shuffle=True,
)

model.save('models/trained_model_32.h5')
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
