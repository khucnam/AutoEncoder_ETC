import keras
import numpy as np
# import matplotlib.pyplot as plt
#%matplotlib inline
from tensorflow.keras.models import Sequential
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, add
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras import regularizers
from keras.regularizers import l2
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Conv1D, MaxPooling1D, AveragePooling1D, UpSampling1D
from keras.utils import np_utils
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.metrics import roc_curve
import os, math
from sklearn.metrics import confusion_matrix, matthews_corrcoef, classification_report, roc_auc_score, accuracy_score
from keras.optimizers import Adam


def labelToOneHot(label):# 0--> [1 0], 1 --> [0 1]
    label = label.reshape(len(label), 1)
    label = np.append(label, label, axis = 1)
    label[:,0] = label[:,0] == 0;
    return label

def classificationPerformanceByThreshold(threshold, y_pred, y_test):
    """
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, Y_pred)
    print("some y_pred ",Y_pred[:5])
    tn, fp, fn, tp = cm.ravel()
    """
    auc=roc_auc_score(y_test, y_pred)
    y_test = np.argmax(y_test, axis = 1)
    
    
    Y_pred = np.empty_like(y_pred)
    for i in range(len(y_pred)):
        if y_pred[i][0]>=threshold:
            Y_pred[i]=np.array([1,0]) #assign as class pos
        else:
            Y_pred[i]=np.array([0,1]) #assign as class neg
    
    Y_pred = np.argmax(Y_pred, axis = 1)
    
    
    cm = confusion_matrix(y_test, Y_pred, labels = [0,1])
   
    tn=cm[0][0]
    fn=cm[1][0]
    tp=cm[1][1]
    fp=cm[0][1]
    
    if float(tp)+float(fn)==0:
        TPR=round(float(tp)/0.00000001,3)
    else:
        TPR=round(float(tp)/(float(tp)+float(fn)),3)
    
    if float(fp)+float(tn)==0:
        FPR=round(float(fp)/(0.00000001),3)
    else:
        FPR=round(float(fp)/(float(fp)+float(tn)),3)
    
    if float(tp) + float(fp) + float(fn) + float(tn)==0:
        accuracy = round((float(tp) + float(tn))/(0.00000001),3)    
    else:
        accuracy = round((float(tp) + float(tn))/(float(tp) + float(fp) + float(fn) + float(tn)),3)
        
    if float(tn) + float(fp)==0:
        specitivity=round(float(tn)/(0.00000001),3)
    else:
        specitivity=round(float(tn)/(float(tn) + float(fp)),3)
        
    if float(tp) + float(fn)==0:
        sensitivity = round(float(tp)/(0.00000001),3)
    else:
        sensitivity = round(float(tp)/(float(tp) + float(fn)),3)
    
    if float(tp) + float(fp)==0:
        precision = round(float(tp)/(0.00000001),3)
    else:
        precision = round(float(tp)/(float(tp) + float(fp)),3)
    
    if math.sqrt((float(tp)+float(fp))*(float(tp)+float(fn))*(float(tn)+float(fp))*(float(tn)+float(fn)))==0:
        mcc = round((float(tp)*float(tn) - float(fp)*float(fn))/0.00000001,3)
    else:
        mcc = round((float(tp)*float(tn) - float(fp)*float(fn))/math.sqrt(
                                                                    (float(tp)+float(fp))
                                                                    *(float(tp)+float(fn))
                                                                    *(float(tn)+float(fp))
                                                                    *(float(tn)+float(fn))
                                                                    ),3)
    balAcc=(sensitivity+specitivity)/2
    if (sensitivity+precision)==0:
        f_measure = round(2*sensitivity*precision/(0.00000001),3)
    else:
        f_measure = round(2*sensitivity*precision/(sensitivity+precision),3)
    
    return accuracy, specitivity, sensitivity, mcc, tp, tn, fp, fn, TPR, FPR, balAcc, precision, f_measure, auc
    
def CAE():

    # Encoder
    input_sig = Input(shape=(300,1))
    e = Conv1D(128,12, strides=1, activation='relu', padding="same")(input_sig)
    e1 = MaxPooling1D(2)(e)
    e2 = Conv1D(64,6, strides=1, activation='relu', padding="same")(e1)
    e3 = MaxPooling1D(2)(e2)
    

    # LAY DATA O LAYER NAY DE FEED VAO 1D CNN BEN DUOI
    #flat = Flatten()(x3) 
    
    # Decoder
    d3 = UpSampling1D(2)(e3)
    d2 = Conv1D(128,6,strides=1, activation='relu', padding="same")(d3)
    d1 = UpSampling1D(2)(d2)
    decoded = Conv1D(1,12,strides=1, activation='relu', padding="same")(d1)

    autoencoder = Model(input_sig, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mse')
    print("^^^^^^^^^^^^^^^^ autoencoder ^^^^^^^^^^^^^^^^ \n",autoencoder.summary())
    return autoencoder
    
def extract_layers(main_model, starting_layer_ix, ending_layer_ix):
    # create an empty model
    new_model = Sequential()
    for ix in range(starting_layer_ix, ending_layer_ix + 1):
        curr_layer = main_model.get_layer(index=ix)
        # copy this layer over to the new model
        new_model.add(curr_layer)
    print("^^^^^^^^^^^^^^^^ AE extracted model ^^^^^^^^^^^^^^^^ \n",new_model.summary())
    return new_model
    
def CNN(latentFeature):
    # define model
    input_sig = Input(shape=(latentFeature.shape[1]*latentFeature.shape[2],1))
    x1 = Conv1D(64,48, activation='relu', padding='same')(input_sig)
    x2 = MaxPooling1D(2)(x1)
    x3 = Conv1D(32,46, activation='relu', padding='same')(x2)
    x4 = Dropout(0.5)(x3)
    x5 = MaxPooling1D(2)(x4)
    x6 = Dropout(0.5)(x5)
    flat = Flatten()(x6)
    d1 = Dense(512, activation='relu')(flat)
    output_layer = Dense(2, activation='softmax')(d1)

    cnn= Model(input_sig, output_layer)
    print("^^^^^^^^^^^^^^^^ CNN ^^^^^^^^^^^^^^^^ \n",cnn.summary())
    return cnn
    
def trainAndPredict(pathToSaveModel, trainFile, testFile, AE_epochs, AE_batch_size, CNN_epochs, CNN_batch_size, CNN_lr, wd, fold):
    
    #xu ly train input
    dataset=pd.read_csv(trainFile,header=None)
    X_train = dataset.iloc[:, 0:wd*20].values
    X_train=X_train.reshape(X_train.shape[0],wd*20,1)
    y_train = dataset.iloc[:, wd*20].values
    y_train = labelToOneHot(y_train)
    print("X_train shape ",X_train.shape)
    print("y_train shape = ",y_train.shape)
    
    #lay train input tren neg only
    X_train_neg=dataset[dataset[dataset.shape[1]-1]==0]
    X_train_neg = X_train_neg.iloc[:, 0:wd*20].values
    X_train_neg=X_train_neg.reshape(X_train_neg.shape[0],wd*20,1)
    print("X_train_neg shape ",X_train_neg.shape)
    
    #xu ly test input
    dataset=pd.read_csv(testFile,header=None)
    X_test = dataset.iloc[:, 0:wd*20].values
    X_test=X_test.reshape(X_test.shape[0],wd*20,1)
    y_test = dataset.iloc[:, wd*20].values
    y_test = labelToOneHot(y_test)
    print("X_test shape ",X_test.shape)
    print("y_test shape = ",y_test.shape)
    
    #train AE
    autoencoder=CAE()
    autoencoder.compile(metrics=['mse', "accuracy"],loss='mean_squared_error', optimizer='Adam')
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=30)
    cp = ModelCheckpoint(pathToSaveModel+"/fold"+str(fold)+"_epoch"+str(AE_epochs)+".batch_size"+str(AE_batch_size)+".AE.h5", monitor='loss', mode='min', verbose=1, save_best_only=True)
    AE_history = autoencoder.fit(X_train_neg, X_train_neg, batch_size=AE_batch_size, epochs=AE_epochs, verbose=1, callbacks=[cp])
    AE_saved_model = load_model(pathToSaveModel+"/fold"+str(fold)+"_epoch"+str(AE_epochs)+".batch_size"+str(AE_batch_size)+".AE.h5")
    starting_layer_ix=0
    ending_layer_ix=4 # bottle neck can lay
    #lay ve best half AE
    half_CAE = extract_layers(AE_saved_model, starting_layer_ix, ending_layer_ix)
    
    
    #train CNN tren train data
    train_latentFeature=half_CAE.predict(X_train)
    print("train_latentFeature shape ",train_latentFeature.shape)
    train_latentFeature=train_latentFeature.reshape(train_latentFeature.shape[0],train_latentFeature.shape[1]*train_latentFeature.shape[2],1)
    print("train_latentFeature reshape ",train_latentFeature.shape)
    test_latentFeature=half_CAE.predict(X_test)
    print("test_latentFeature shape ",test_latentFeature.shape)
    test_latentFeature=test_latentFeature.reshape(test_latentFeature.shape[0],test_latentFeature.shape[1]*test_latentFeature.shape[2],1)
    print("test_latentFeature reshape ",test_latentFeature.shape)
    cnnModel = CNN(train_latentFeature)
    adam = Adam(lr=CNN_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    cnnModel.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=30)
    cp = ModelCheckpoint(pathToSaveModel+"/fold"+str(fold)+"_epoch"+str(CNN_epochs)+".batch_size"+str(CNN_batch_size)+".CNN.h5", monitor='loss', mode='min', verbose=2, save_best_only=True)
    CNN_history = cnnModel.fit(train_latentFeature, y_train, batch_size=CNN_batch_size, epochs=CNN_epochs, verbose=1, callbacks=[cp, es])
    CNN_saved_model = load_model(pathToSaveModel+"/fold"+str(fold)+"_epoch"+str(CNN_epochs)+".batch_size"+str(CNN_batch_size)+".CNN.h5")
    y_pred=CNN_saved_model.predict(test_latentFeature)
    return y_pred, y_test
    

def run(pathToSaveModel, folder, inputFileTrain, inputFileTest, rsFile, AE_epochs, AE_batch_size, CNN_epochs, CNN_batch_size, CNN_lr, wd, fold):
  trainFile=inputFileTrain
  testFile=inputFileTest    
  y_pred, y_test = trainAndPredict(pathToSaveModel, trainFile, testFile, AE_epochs, AE_batch_size, CNN_epochs, CNN_batch_size, CNN_lr, wd, fold)

  
  #Ghi ket qua ra file
  f2=open(rsFile,"a")
  f2.write("Threshold ,Aaccuracy ,Specitivity ,Sensitivity ,MCC ,Precision ,tp ,tn ,fp ,fn ,TPR ,FPR ,balAcc ,precision , f_measure ,AUC\n")
  f2.close()
  threshold=0.01
  while threshold<1.01:
    accuracy, specitivity, sensitivity, mcc, tp, tn, fp, fn, TPR, FPR, balAcc, precision, f_measure, auc = classificationPerformanceByThreshold(threshold, y_pred, y_test)
    f2=open(rsFile,"a")
    f2.write(str(threshold)+", "+str(accuracy)+", "+str(specitivity)+", "+str(sensitivity)+", "+str(mcc)+", "+str(precision)+", "+str(tp)+", "+str(tn)+", "+str(fp)+", "+str(fn)+", "+str(TPR)+", "+str(FPR)+", "+str(balAcc)+", "+str(precision)+", "+str(f_measure)+", "+str(auc)+"\n")
    f2.close()
    threshold+=0.002
    
bindingTypes=["FAD","FMN"]
wd=15
for bdType in bindingTypes:
  folder=bdType+"/pssm features wd 15"
  pathToSaveModel=bdType+"/pssm features wd 15/Saved models"
  for AE_epoch in [2]:
    for AE_batch_size in [64]:
        for CNN_epoch in [2]:
            for CNN_batch_size in [64]:
                for CNN_lr in [0.1, 0.01, 0.001]:
                      for fold in [1,2,3,4,5]:
                        inputFileTrain=folder+"/input.fold.train"+str(fold)+".csv"
                        inputFileTest=folder+"/input.fold.test"+str(fold)+".csv" 
                        rsFile=folder+"/CAE_CNN results/architecture 1/epoch"+str(CNN_epoch)+"_batchsize"+str(CNN_batch_size)+"_lr"+str(CNN_lr)+"_fold.result"+str(fold)+".csv"
                        run(pathToSaveModel, folder, inputFileTrain, inputFileTest, rsFile, AE_epoch, AE_batch_size, CNN_epoch, CNN_batch_size, CNN_lr, wd, fold)
                      inputFileTrain=folder+"/input.train.csv"
                      inputFileTest=folder+"/ind.test.csv" 
                      rsFile=folder+"/CAE_CNN results/architecture 1/epoch"+str(CNN_epoch)+"_batchsize"+str(CNN_batch_size)+"_lr"+str(CNN_lr)+"_ind.result.csv"
                      run(pathToSaveModel, folder, inputFileTrain, inputFileTest, rsFile, AE_epoch, AE_batch_size, CNN_epoch, CNN_batch_size, CNN_lr, wd, 0)
                      

        
