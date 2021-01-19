import keras
import numpy as np
# import matplotlib.pyplot as plt
#%matplotlib inline

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, add
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras import regularizers
from keras.regularizers import l2
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.utils import np_utils
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


def labelToOneHot(label):# 0--> [1 0], 1 --> [0 1]
    label = label.reshape(len(label), 1)
    label = np.append(label, label, axis = 1)
    label[:,0] = label[:,0] == 0;
    return label
    

def CAE():
    x = Input(shape=(16, 20,1)) 

    # Encoder
    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    pool1 = MaxPooling2D((2, 2), padding='same')(conv1_1)
    conv1_3 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)

    #bottle neck
    h = MaxPooling2D((2, 2), padding='same')(conv1_3)


    # Decoder
    conv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(h)
    up1 = UpSampling2D((2, 2))(conv2_1)
    conv2_2 = Conv2D(1, (3,3),  padding='same', activation='relu')(up1)

    #output
    r = UpSampling2D((2, 2))(conv2_2)

    autoencoder = Model(inputs=x, outputs=r)
    autoencoder.compile(optimizer='adadelta', loss='mae')
    print(autoencoder.summary())

    return autoencoder
    
def trainAModel(pathToSaveModel, trainFile, testFile, epochs, batch_size, wd):
    dataset=pd.read_csv(trainFile,header=None)
    X_train = dataset.iloc[:, 0:wd*20].values
    y_train = dataset.iloc[:, wd*20].values
    padding=np.zeros([1,20,1])
    
    X_train=X_train.reshape(X_train.shape[0],wd,20,1)
    print("X_train shape before concat ",X_train.shape)
    
    X=np.zeros([X_train.shape[0],16,20,1])
    for i in range(X_train.shape[0]):
        # print("X_train[i] shape before concat ",X_train[i].shape)
        # print("padding shape ",padding.shape)
        temp = np.concatenate((X_train[i], padding))
        X[i]=temp
        # print("X_train[i] shape after concat ",temp.shape)
    X_train=X
    print("X_train shape after concatenate ",X_train.shape)
         
    
    y_train=labelToOneHot(y_train)
    print("X_train shape = ",X_train.shape)
    print("y_train shape = ",y_train.shape)
    
    dataset=pd.read_csv(testFile,header=None)
    X_test = dataset.iloc[:, 0:wd*20].values
    y_test = dataset.iloc[:, wd*20].values
    X_test=X_test.reshape(X_test.shape[0],wd,20,1)
    
    print("X_test shape before concat ",X_test.shape)
    X=np.zeros([X_test.shape[0],16,20,1])
    for i in range(X_test.shape[0]):
        # print("X_train[i] shape before concat ",X_train[i].shape)
        # print("padding shape ",padding.shape)
        temp = np.concatenate((X_test[i], padding))
        X[i]=temp
        # print("X_train[i] shape after concat ",temp.shape)
    X_test=X
    print("X_test shape after concatenate ",X_test.shape)
    print("y_test shape = ",y_test.shape)
    
    autoencoder=CAE()
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=30)

    cp = ModelCheckpoint(pathToSaveModel+"/epoch"+str(epoch)+".batchsize"+str(batch_size)+".AE.h5", monitor='loss', mode='min', verbose=1, save_best_only=True)
    history = autoencoder.fit(X_train, X_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[cp, es])
    saved_model = load_model(pathToSaveModel+"/epoch"+str(epoch)+".batchsize"+str(batch_size)+".AE.h5")
    
    decoded_pssm= saved_model.predict(X_test)
    print("decoded_pssm shape ",decoded_pssm.shape)
    return autoencoder, decoded_pssm, X_test, y_test
    
def RecErr(predicted_decode, X_test, y_test):
       
    
    #mse
    predicted_decode_temp=np.zeros([predicted_decode.shape[0],300])
    for i in range(predicted_decode.shape[0]):
        temp=predicted_decode[i].reshape(16,20)
        predicted_decode_temp[i]=temp[:15,:].reshape(1,300)
        
    #X_test
    X_test_temp=np.zeros([X_test.shape[0],300])
    for i in range(X_test.shape[0]):
        temp=X_test[i].reshape(16,20)
        X_test_temp[i]=temp[:15,:].reshape(1,300)
        
    mse = np.mean(np.power(X_test_temp - predicted_decode_temp, 2), axis=1)
    
    print("X_test_temp shape ", X_test_temp.shape, " type ",type(X_test_temp))
    print("predicted_decode_temp ", predicted_decode_temp.shape, " type ", type(predicted_decode_temp))
    print("mse shape", mse.shape)
    print("mse values ",mse)
    print("y_test shape", y_test.shape)
    error_df = pd.DataFrame({'Reconstruction_error': mse, 'True_class': y_test})
    print(error_df[error_df.True_class == 1] )
    print(error_df[error_df.True_class == 0] )
    
    
    print("error_df shape ",error_df.shape)

    precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class, 
                                                               error_df.Reconstruction_error)
    return predicted_decode_temp, error_df, precision_rt, recall_rt, threshold_rt, mse
    
from sklearn.metrics import roc_curve, auc 
import os, math
def run(pathToSaveModel, folder, inputFileTrain, inputFileTest, rsFile, epoch, batch_size):
  # os.chdir(folder)
   
  trainFile=inputFileTrain
  testFile=inputFileTest    
  wd=15
  autoencoder, decoded_pssm, X_test, y_test = trainAModel(pathToSaveModel, trainFile, testFile, epoch, batch_size, wd)
  # print(decoded_pssm)
  predicted_decode_temp, error_df, precision_rt, recall_rt, threshold_rt, mse= RecErr(decoded_pssm, X_test, y_test)
  # print("precision_rt :", precision_rt)
  # print("recall_rt ", recall_rt)
  # print("mse ", mse)
  # print("threshold ",threshold_rt)  

  false_pos_rate, true_pos_rate, thresholds = roc_curve(error_df.True_class, error_df.Reconstruction_error)
  roc_auc = auc(false_pos_rate, true_pos_rate,)
  # plt.figure(figsize=(10,8))
  # plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f'% roc_auc)
  # plt.plot([0,1],[0,1], linewidth=3)
  # plt.xlim([-0.01, 1])
  # plt.ylim([0, 1.01])
  # plt.legend(loc='lower right')
  # plt.title('Receiver operating characteristic curve (ROC)')
  # plt.ylabel('True Positive Rate')
  # plt.xlabel('False Positive Rate')

  # plt.show()
  from sklearn.metrics import confusion_matrix, matthews_corrcoef, classification_report, roc_auc_score, accuracy_score
  #Ghi ket qua ra file
  f2=open(rsFile,"a")
  f2.write("Threshold ,Aaccuracy ,Specitivity ,Sensitivity ,MCC ,Precision ,tp ,tn ,fp ,fn ,TPR ,FPR ,balAcc ,AUC\n")
  f2.close()

  for i in range(len(threshold_rt)):
    threshold=threshold_rt[i]
    precision=precision_rt[i]
    
    # Prepair data for confusion matrix
    pred_y = [1 if e > threshold else 0 for e in error_df.Reconstruction_error.values]
    # print ('The accuracy is: ',str(round(accuracy_score(error_df.True_class, pred_y)*100, 4)),' \n')

    # print ('The confusion matrix: ')
    cm = confusion_matrix(error_df.True_class, pred_y)
    # print(cm, '\n')
    tn, fp, fn, tp = confusion_matrix(error_df.True_class, pred_y).ravel()
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
    
    if math.sqrt((float(tp)+float(fp))*(float(tp)+float(fn))*(float(tn)+float(fp))*(float(tn)+float(fn)))==0:
        mcc = round((float(tp)*float(tn) - float(fp)*float(fn))/0.00000001,3)
    else:
        mcc = round((float(tp)*float(tn) - float(fp)*float(fn))/math.sqrt(
                                                                    (float(tp)+float(fp))
                                                                    *(float(tp)+float(fn))
                                                                    *(float(tn)+float(fp))
                                                                    *(float(tn)+float(fn))
                                                                    ),3)
    balAcc=round((sensitivity+specitivity)/2,3)

    f2=open(rsFile,"a")
    f2.write(str(threshold)+", "+str(accuracy)+", "+str(specitivity)+", "+str(sensitivity)+", "+str(mcc)+", "+str(precision)+", "+str(tp)+", "+str(tn)+", "+str(fp)+", "+str(fn)+", "+str(TPR)+", "+str(FPR)+", "+str(balAcc)+", "+str(roc_auc)+"\n")
    f2.close()
    
bindingTypes=["FAD","FMN"]
pathToSaveModel=""
for bdType in bindingTypes:
  folder=bdType+"/pssm features wd 15"
  pathToSaveModel=bdType+"/pssm features wd 15/Saved models"
  for epoch in [200]:
    for batch_size in [16]:
      for fold in [1,2,3,4,5]:
        inputFileTrain=folder+"/input.fold.train"+str(fold)+".csv"
        inputFileTest=folder+"/input.fold.test"+str(fold)+".csv" 
        rsFile=folder+"/CAE results/architecture 1-MAE loss/epoch"+str(epoch)+"_batchsize"+str(batch_size)+"_fold.result"+str(fold)+".csv"
        run(pathToSaveModel, folder, inputFileTrain, inputFileTest, rsFile, epoch, batch_size)
      inputFileTrain=folder+"/input.train.csv"
      inputFileTest=folder+"/ind.test.csv" 
      rsFile=folder+"/CAE results/architecture 1-MAE loss/epoch"+str(epoch)+"_batchsize"+str(batch_size)+"_ind.result.csv"
      run(pathToSaveModel, folder, inputFileTrain, inputFileTest, rsFile, epoch, batch_size)
      

    
