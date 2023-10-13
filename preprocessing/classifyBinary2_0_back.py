import utils as ut
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

datafilepath = "../data/datasets/dataset_1/bands/"

def import_subject(subject, pattern2, delta=False, alpha=False, beta=False):
    files = ut.find_files_with_ext('.npy', datafilepath)
    files = [s for s in files if f"_{subject}_" in s and pattern2 in s]
    data = [np.load(x) for x in files]
    
    if delta == False:
        for x in data:
            for y in x[0]:
                y[0] = np.zeros(68, dtype=float)
    if alpha == False:
        for x in data:
            for y in x[0]:
                y[1] = np.zeros(68, dtype=float)
    if beta == False:
        for x in data:
            for y in x[0]:
                y[2] = np.zeros(68, dtype=float)
    return data

def classify(data_one, data_two, save_name):
    labels_one = np.zeros((data_one[0].shape[0],))
    labels_two = np.ones((data_two[0].shape[0],))
    
    data = np.concatenate((data_one[0], data_two[0]), axis=0)
    labels = np.concatenate((labels_one, labels_two), axis=0)
    
    labels = to_categorical(labels, num_classes=2)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

    model = keras.Sequential([
        layers.Input(shape=(63, 3, 68)),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 1)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 1)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    checkpoint = ModelCheckpoint('save_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, verbose=1, restore_best_weights=True)

    history = model.fit(x_train, y_train,
                        epochs=100,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=[checkpoint, early_stopping])

    model.load_weights('save_model.h5')
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    with open('model_performance.txt', 'a') as file:
        file.write(f"{save_name}, {score[1]}\n")

# Example usage for Subject 1 and 2
for subject in [1, 2]:
    for frequency_band in [("delta", True, False, False), ("alpha", False, True, False), ("beta", False, False, True)]:
        band_name, delta, alpha, beta = frequency_band
        data_zero_back = import_subject(subject, "zeroback", delta=delta, alpha=alpha, beta=beta)
        data_one_back = import_subject(subject, "twoback", delta=delta, alpha=alpha, beta=beta)
        
        classify(data_zero_back, data_one_back, f"TwoClassSub{subject}_{band_name}_zerobacktwoback")

