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
        print(files)
        data = []
        for x in files:
            data.append(np.load(x))
        print(len(data))
        for y in data:
            print(y.shape)
       
        if delta==False:
            for x in data:
                for y in x[0]: #y is the three arrays for delta, alpha, and beta
                    y[0] = np.zeros(68, dtype=float) #zeros out delta
        if alpha==False:
            for x in data:
                for y in x[0]: #y is the three arrays for delta, alpha, and beta
                    y[1] = np.zeros(68, dtype=float) #zeros out alpha
        if beta==False:
            for x in data:
                for y in x[0]: #y is the three arrays for delta, alpha, and beta
                    y[2] = np.zeros(68, dtype=float) #zeros out beta
        return data

def classify(data_one, data_two, data_three, save_name):
    
    labels_one = np.ones((data_one[0].shape[0],)) * 0  
    labels_two = np.ones((data_two[0].shape[0],)) * 1  
    labels_three = np.ones((data_three[0].shape[0],)) * 2  

    data = np.concatenate((data_one[0], data_two[0], data_three[0]), axis=0)
    labels = np.concatenate((labels_one, labels_two, labels_three), axis=0)

    labels = to_categorical(labels, num_classes=3)

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
        layers.Dense(3, activation='softmax')
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
    test_acc = score[1]

    with open('model_performance.txt', 'a') as file:
        file.write(f"{save_name}, {test_acc}\n") 


##Subject One
data_one = import_subject(1, "zeroback", delta=True)
data_two = import_subject(1, "oneback", delta=True)
data_three = import_subject(1, "twoback", delta=True)
classify(data_one, data_two, data_three, "ThreeClassSub1_delta")

data_one = import_subject(1, "zeroback", alpha=True)
data_two = import_subject(1, "oneback", alpha=True)
data_three = import_subject(1, "twoback", alpha=True)
classify(data_one, data_two, data_three, "ThreeClassSub1_alpha")

data_one = import_subject(1, "zeroback", beta=True)
data_two = import_subject(1, "oneback", beta=True)
data_three = import_subject(1, "twoback", beta=True)
classify(data_one, data_two, data_three, "ThreeClassSub1_beta")

data_one = import_subject(1, "zeroback", delta=True, alpha=True)
data_two = import_subject(1, "oneback", delta=True, alpha=True)
data_three = import_subject(1, "twoback", delta=True, alpha=True)
classify(data_one, data_two, data_three, "ThreeClassSub1_deltaAlpha")

data_one = import_subject(1, "zeroback", delta=True, beta=True)
data_two = import_subject(1, "oneback", delta=True, beta=True)
data_three = import_subject(1, "twoback", delta=True, beta=True)
classify(data_one, data_two, data_three, "ThreeClassSub1_deltaBeta")

data_one = import_subject(1, "zeroback",alpha=True, beta=True)
data_two = import_subject(1, "oneback",alpha=True, beta=True)
data_three = import_subject(1, "twoback",alpha=True, beta=True)
classify(data_one, data_two, data_three, "ThreeClassSub1_alphaBeta")

data_one = import_subject(1, "zeroback",delta=True,alpha=True, beta=True)
data_two = import_subject(1, "oneback",delta=True,alpha=True, beta=True)
data_three = import_subject(1, "twoback",delta=True,alpha=True, beta=True)
classify(data_one, data_two, data_three, "ThreeClassSub1_deltaalphaBeta")

##Subject Two
data_one = import_subject(2, "zeroback", delta=True)
data_two = import_subject(2, "oneback", delta=True)
data_three = import_subject(2, "twoback", delta=True)
classify(data_one, data_two, data_three, "ThreeClassSub2_delta")

data_one = import_subject(2, "zeroback", alpha=True)
data_two = import_subject(2, "oneback", alpha=True)
data_three = import_subject(2, "twoback", alpha=True)
classify(data_one, data_two, data_three, "ThreeClassSub2_alpha")

data_one = import_subject(2, "zeroback", beta=True)
data_two = import_subject(2, "oneback", beta=True)
data_three = import_subject(2, "twoback", beta=True)
classify(data_one, data_two, data_three, "ThreeClassSub2_beta")

data_one = import_subject(2, "zeroback", delta=True, alpha=True)
data_two = import_subject(2, "oneback", delta=True, alpha=True)
data_three = import_subject(2, "twoback", delta=True, alpha=True)
classify(data_one, data_two, data_three, "ThreeClassSub2_deltaAlpha")

data_one = import_subject(2, "zeroback", delta=True, beta=True)
data_two = import_subject(2, "oneback", delta=True, beta=True)
data_three = import_subject(2, "twoback", delta=True, beta=True)
classify(data_one, data_two, data_three, "ThreeClassSub2_deltaBeta")

data_one = import_subject(2, "zeroback",alpha=True, beta=True)
data_two = import_subject(2, "oneback",alpha=True, beta=True)
data_three = import_subject(2, "twoback",alpha=True, beta=True)
classify(data_one, data_two, data_three, "ThreeClassSub2_alphaBeta")

data_one = import_subject(2, "zeroback",delta=True,alpha=True, beta=True)
data_two = import_subject(2, "oneback",delta=True,alpha=True, beta=True)
data_three = import_subject(2, "twoback",delta=True,alpha=True, beta=True)
classify(data_one, data_two, data_three, "ThreeClassSub2_deltaalphaBeta")








