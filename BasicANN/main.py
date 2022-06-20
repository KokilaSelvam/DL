import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import accuracy_score




dataset=pd.read_csv('Churn_Modelling.csv',delimiter=',')
# print(dataset.head())
# print(dataset.shape)

X=dataset.iloc[:,3:13]
Y=dataset.iloc[:,13]

X=X.drop(['Geography','Gender'],axis=1)

Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.20,random_state=0)

l_sc=StandardScaler()
Xtrain=l_sc.fit_transform(Xtrain)
Xtest=l_sc.transform(Xtest)

model=Sequential()
model.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu',input_dim=8))
model.add(Dropout(0.3))
model.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1,kernel_initializer='glorot_uniform',activation='sigmoid'))
model.compile(optimizer='Adamax',loss='binary_crossentropy',metrics=['accuracy'])
model_history=model.fit(Xtrain,Ytrain,validation_split=0.33,batch_size=10,epochs=100)

#summarize history for accuracy
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()

#summarize history for Loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

y_pred=model.predict(Xtest)
y_pred=(y_pred>0.5)
score=accuracy_score(Ytest,y_pred)
print(score)
model.summary()
