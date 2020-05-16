import cv2
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

pickle_in=open("model_trained_64.p","rb")
model=pickle.load(pickle_in)

df=pd.read_csv('fer2013.csv')

X_test,Y_test=[],[]

for index, row in df.iterrows():
    val=row['pixels'].split(" ")
    try:
        if 'PrivateTest' in row['Usage']:
           X_test.append(np.array(val,'float32'))
           Y_test.append(row['emotion'])
    except:
        print(f"error occured at index :{index} and row:{row}")

X_test = np.array(X_test,'float32')
Y_test = np.array(Y_test,'float32')

#Y_test=to_categorical(Y_test, num_classes=num_labels)

##cannot produce
#normalizing data between oand 1

X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test, axis=0)

#X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)

X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)


pred = model.predict_classes(X_test)
print(accuracy_score(Y_test, pred))