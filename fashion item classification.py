# Remember the 10 classes decoding is as follows:
# 0 => T-shirt/top
# 1 => Trouser
# 2 => Pullover
# 3 => Dress
# 4 => Coat
# 5 => Sandal
# 6 => Shirt
# 7 => Sneaker
# 8 => Bag
# 9 => Ankle boot
import pandas as pd
import numpy as np
fashion_train_df=pd.read_csv('fashion-mnist_train.csv',sep=',')
fashion_test_df=pd.read_csv('fashion-mnist_test.csv',sep=',')
training = np.array(fashion_train_df, dtype = 'float32')
testing = np.array(fashion_test_df, dtype='float32')
X_train = training[:,1:]/255
y_train = training[:,0]
X_test = testing[:,1:]/255
y_test = testing[:,0]
from sklearn.model_selection import train_test_split
X_train,X_validate,y_train,y_validate=train_test_split(X_train,y_train,test_size=0.2,random_state=12345)
X_train = X_train.reshape(X_train.shape[0], *(28, 28, 1))
X_test = X_test.reshape(X_test.shape[0],*(28,28,1))
X_validate = X_validate.reshape(X_validate.shape[0],*(28,28,1))
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.optimizers import Adam
cnn_model = Sequential()
cnn_model.add(Conv2D(64,3,3,input_shape=(28,28,1),activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(3,3)))
cnn_model.add(Flatten())
cnn_model.add(Dense(output_dim=32,activation='relu'))
cnn_model.add(Dropout(p=0.2))
cnn_model.add(Dense(output_dim=32,activation='relu'))
cnn_model.add(Dropout(p=0.2))
cnn_model.add(Dense(output_dim=10,activation='sigmoid'))
cnn_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001),metrics=['accuracy'])
epochs=100
cnn_model.fit(X_train,y_train,batch_size=512,nb_epoch=epochs,verbose=1,validation_data=(X_validate,y_validate))
evaluation = cnn_model.evaluate(X_test, y_test)
print('Test Accuracy : {:.3f}'.format(evaluation[1]))
predicted_classes = cnn_model.predict_classes(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted_classes)
from sklearn.metrics import classification_report
num_classes=10
target_names = ["Class{}".format(i) for i in range(num_classes)]
print(classification_report(y_test, predicted_classes, target_names = target_names))