import numpy as np
import h5py
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


with h5py.File('data/processed/f3_32.h5', 'r') as dataset:
    x_train = np.array(dataset['train/X'])
    y_train = np.array(dataset['train/Y'])
    x_val = np.array(dataset['val/X'])
    y_val = np.array(dataset['val/Y'])

classnames = {
    0: 'Upper North',
    1: 'Middle North',
    2: 'Lower North',
    3: 'Chalk/Rijnland',
    4: 'Scruff',
    5: 'Zechstein',
}

m = x_train.shape[0]
n_classes = len(classnames)

input_shape = x_train.shape[1:]

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

model.summary()

batch_size = 32
epochs = 80

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=epochs,
    shuffle=True,
    callbacks=[keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, min_delta=0.01, restore_best_weights=True)]
)

model.save('models/trained-model-f3-32.h5')
print('Model saved!')
