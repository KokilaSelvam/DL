import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.layers import Dropout
from sklearn.metrics import accuracy_score



dataset=pd.read_csv('Churn_Modelling.csv',delimiter=',')
# print(dataset.head())
# print(dataset.shape)

X=dataset.iloc[:,3:13]
Y=dataset.iloc[:,13]

# X=X.drop(['Geography','Gender'],axis=1)
#


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
len1=LabelEncoder()
X.iloc[:,1]=len1.fit_transform(X.iloc[:,1])
len2=LabelEncoder()
X.iloc[:,2]=len2.fit_transform(X.iloc[:,2])
onehot=OneHotEncoder()
X=onehot.fit_transform(X).toarray()
X=X[:,1:]

Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.20,random_state=0)

l_sc=StandardScaler()
Xtrain=l_sc.fit_transform(Xtrain)
Xtest=l_sc.transform(Xtest)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def create_model(layers,activation):
    model=Sequential()
    for i ,nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=Xtrain.shape[1]))
            model.add(Activation(activation))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
    model.add(Dense(1))
    model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model
model=KerasClassifier(build_fn=create_model,verbose=0)

layers=[[20],[40,20],[45,30,15]]
activation=['sigmoid','relu']
param_grid=dict(layers=layers,activation=activation,batch_size=[128,256],epochs=[30])
grid=GridSearchCV(model,param_grid=param_grid)
result=grid.fit(Xtrain,Ytrain)

print("Best score",result)
# print("Best param",result.best_params_)