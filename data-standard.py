import pandas as pd
from collections import deque
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import  ModelCheckpoint
import time
from sklearn import preprocessing
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


SEQ_LENGTH=240
FUTURE_PRIDICT=10
EPOCHS = 10  # how many passes through our data
BATCH_SIZE = 64  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
NAME = f"{SEQ_LENGTH}-SEQ-{FUTURE_PRIDICT}-PRED-{int(time.time())}"

def preprocess_df(df):
    df = df.drop("future", 1)  # don't need this anymore.

    for col in df.columns:  # go through all of the columns
        if col != "target":  # normalize all ... except for the target itself!
            df[col] = df[col].pct_change()  # pct change "normalizes" the different currencies (each crypto coin has vastly diff values, we're really more interested in the other coin's movements)
            df.dropna(inplace=True)  # remove the nas created by pct_change
            df = df.replace([np.inf, -np.inf], np.nan)
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)  # scale between 0 and 1.

    df.dropna(inplace=True)
    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=SEQ_LENGTH)
    for i in df.values:  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days) == SEQ_LENGTH:  # make sure we have 60 sequences!
            sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!

    random.shuffle(sequential_data)  # shuffle for good measure.
    buys = []  # list that will store our buy sequences and targets
    sells = []  # list that will store our sell sequences and targets

    for seq, target in sequential_data:  # iterate over the sequential data
        if target == 0:  # if it's a "not buy"
            sells.append([seq, target])  # append to sells list
        elif target == 1:  # otherwise if the target is a 1...
            buys.append([seq, target])  # it's a buy!

    random.shuffle(buys)  # shuffle the buys
    random.shuffle(sells)  # shuffle the sells!

    lower = min(len(buys), len(sells))  # what's the shorter length?

    buys = buys[:lower]  # make sure both lists are only up to the shortest length.
    sells = sells[:lower]  # make sure both lists are only up to the shortest length.

    sequential_data = buys + sells  # add them together
    random.shuffle(
        sequential_data)  # another shuffle, so the model doesn't get confused with all 1 class then the other.

    X = []
    y = []
    print("x an y")
    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    return np.array(X), np.array(y)  # return X and y...and make X a numpy array!

def classify(future,current):
    if float(future)> float(current):
        return 1
    elif float(future)< float(current):
        return -1
    else:
        return 0


data1= pd.read_csv('BTTUSDT-formated.csv')
data1.drop(labels='Unnamed: 0',axis=1,inplace=True)
data1['future']= data1['price'].shift(-FUTURE_PRIDICT)
data1['target']=list(map(classify,data1['price'],data1['future']))
ind= sorted(data1.index.values)
treshold= ind[-int(0.05*len(ind))]

validation_data=data1[(data1.index>= treshold)]
data1=data1[(data1.index<= treshold)]

train_x, train_y = preprocess_df(data1)
validation_x, validation_y = preprocess_df(validation_data)

print(f"train data: {len(train_x)} validation: {len(validation_x)}")

model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

print("first layer")
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())
print("2 layer")
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())
print("3 layer")
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
print("4 layer")
model.add(Dense(2, activation='softmax'))


opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
print("opt")
# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)



filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                                      mode='max')) # saves only the best ones

# Train model
history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[ checkpoint],
)

# Score model
score = model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Save model
model.save("models/{}".format(NAME))
