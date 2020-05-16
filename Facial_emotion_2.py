import pandas as pd
import sys, os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D,BatchNormalization,Activation
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau,TensorBoard,EarlyStopping,ModelCheckpoint
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import pickle


df=pd.read_csv('fer2013.csv')

# print(df.info())
# print(df["Usage"].value_counts())

# print(df.head())
X_train,Y_train,X_test,Y_test=[],[],[],[]
#path='fer2013'
#mylist=os.listdir(path)

for index, row in df.iterrows():
    val=row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
           X_train.append(np.array(val,'float32'))
           Y_train.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
           X_test.append(np.array(val,'float32'))
           Y_test.append(row['emotion'])
    except:
        print(f"error occured at index :{index} and row:{row}")
        
        
num_features=64
num_labels=7
batch_size=64
epochs=50
width,height=48,48

X_train = np.array(X_train,'float32')
Y_train = np.array(Y_train,'float32')
X_test = np.array(X_test,'float32')
Y_test = np.array(Y_test,'float32')

Y_train=to_categorical(Y_train, num_classes=num_labels)
Y_test=to_categorical(Y_test, num_classes=num_labels)

X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test, axis=0)

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)

X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)



#def my_model():
#    
#    NoOfFilters=60
#    SizeOfFilter1=(5,5)
#    SizeOfFilter2=(3,3)
#    SizeOfPool=(2,2)
#    NoOfNodes=500
#    model=Sequential()
#    model.add((Conv2D(NoOfFilters,SizeOfFilter1,input_shape=X_train.shape[1:],activation='relu')))
#    model.add((Conv2D(NoOfFilters,SizeOfFilter1,activation='relu')))  
#    model.add(MaxPooling2D(pool_size=SizeOfPool))
#    model.add((Conv2D(NoOfFilters//2,SizeOfFilter2,activation='relu'))) 
#    model.add((Conv2D(NoOfFilters//2,SizeOfFilter2,activation='relu')))
#    model.add(MaxPooling2D(pool_size=SizeOfPool))
#    model.add(Dropout(0.5))
#    
#    model.add(Flatten())
#    model.add(Dense(NoOfNodes,activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(7,activation='softmax')) 
#    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
#    return model
#
#model=my_model()
#print(model.summary())

def my_model():
    #64 epoch 50
    model = Sequential()

    model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
    model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    
    model.add(Dense(2*2*2*num_features, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2*2*num_features, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2*num_features, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_labels, activation='softmax'))
    model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
    return model
model=my_model()
print(model.summary())

lr_reducer=ReduceLROnPlateau(monitor='val_loss',factor=0.9,patience=3,verbose=1)
#tensorboard=TensorBoard(log_dir='./logs')

early_stopper=EarlyStopping(monitor='val_loss',min_delta=0,patience=8,verbose=1,mode='auto')
checkpointer = ModelCheckpoint(r'C:\Users\Harshit\Desktop\Facial emotion Recognition\model_trained.p', monitor='val_loss', verbose=1, save_best_only=True)

callbacks=[lr_reducer, early_stopper, checkpointer]
model.fit(X_train,Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, Y_test),
          callbacks=callbacks,
          shuffle=True)

scores = model.evaluate(np.array(X_test), np.array(Y_test), batch_size=batch_size)
print("Loss: " + str(scores[0]))
print("Accuracy: " + str(scores[1]))

###save model
pickle_out=open("model_trained_1.p","wb")
pickle.dump(model,pickle_out)
pickle_out.close()