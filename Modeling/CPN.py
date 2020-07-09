from ModelConfig import *
from keras import Model
from keras.layers import Dense, Input, BatchNormalization, Concatenate,Add
from tensorflow.keras.callbacks import CSVLogger
from SolarNN5_4.testMetric.loadTVdata import loadData,chooseTestdata
from keras import regularizers
from keras.optimizers import Adam,Nadam,Adamax,Adadelta,Adagrad
import numpy as np
import matplotlib.pyplot as plt
import sys,time
from keras.callbacks import LearningRateScheduler
import keras.backend as K
import pandas as pd



class solarNN():
    def __init__(self,traindataNum,itera, inputDim, outputDim, featrueNum=1,
                 hiddenLayersNum=5, learningRate=0.001,
                 eachLayersNum=None,cross=False):
        if featrueNum > 4:
            print('featureNum must less than 4!')
            sys.exit()
        self.traindataNum=traindataNum
        self.itera=itera
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.hiddenLayersNum = hiddenLayersNum
        self.featureNum = featrueNum
        self.learningRate = learningRate
        self.scalerFlag = False
        self.val_loss = []  # 初始化验证集的loss值
        self.loss = []  # 初始化训练集的loss
        self.mean_absolute_error = []  # 初始化训练集的均方误差
        self.val_mean_absolute_error = []  # 初始化验证集的均方误差
        if cross is True:
            self.getData()

        if eachLayersNum is None:
            if self.hiddenLayersNum == 5:
                self.eachLayersNum = [7, 10, 10, 6, 3]
            elif self.hiddenLayersNum == 6:
                self.eachLayersNum = [7, 10, 12, 10, 6, 3]
            elif self.hiddenLayersNum == 7:
                self.eachLayersNum = [7, 9, 12, 12, 9, 6, 3]
            elif self.hiddenLayersNum == 8:
                self.eachLayersNum = [7, 8, 9, 12, 10, 9, 6, 3]
            elif self.hiddenLayersNum == 9:
                self.eachLayersNum = [6, 8, 10, 12, 15, 10, 8, 6, 3]
            elif self.hiddenLayersNum == 10:
                self.eachLayersNum = [6, 7, 9, 12, 15, 12, 10, 8, 5, 2]
            elif self.hiddenLayersNum == 12:
                self.eachLayersNum = [6, 7, 8, 9, 12, 15, 12, 10, 8, 6, 3, 2]
            elif self.hiddenLayersNum == 11:
                self.eachLayersNum = [6, 7, 9, 12, 15, 12, 10, 8, 6, 5, 2]
            elif self.hiddenLayersNum == 12:
                self.eachLayersNum = [6, 8, 10, 15, 20, 20, 15, 10, 8, 6, 4, 2]
                # self.eachLayersNum=[6,8,10,12,15,20,15,12,10,6,4,2]
            elif self.hiddenLayersNum == 13:
                self.eachLayersNum = [6, 7, 10, 12, 15, 18, 25, 20, 15, 10, 6, 4, 3]
            elif self.hiddenLayersNum == 4:
                self.eachLayersNum = [6, 8, 5, 3]
            elif self.hiddenLayersNum == 25:
                self.eachLayersNum = [6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 60, 80, 50, 30, 20, 10, 8, 6,
                                      4,
                                      4, 3]
            else:
                self.eachLayersNum = [7, 8, 9, 12, 10, 9, 6, 3]
        elif isinstance(eachLayersNum, list):
            self.eachLayersNum = eachLayersNum

    def buildParallelModel(self):
        input = Input(shape=(self.inputDim,))
        layer = BatchNormalization()(input)
        # regularPara = 0
        regularPara=0.001
        layer = Dense(8, kernel_regularizer=regularizers.l1_l2(l1=regularPara,l2=regularPara), activation='relu')(layer)
        layer = Dense(10, activation='relu')(layer)
        layer = Dense(12,activation='relu')(layer)
        ######### Parallel Layer 1
        parallelLayer_1_toadd = Dense(12, activation='relu')(layer)
        parallelLayer_1 = Dense(10, activation='relu')(parallelLayer_1_toadd)
        parallelLayer_1 = Dense(10, activation='relu')(parallelLayer_1)
        parallelLayer_1 = Dense(12, activation='relu')(parallelLayer_1)
        parallelLayer_1 = Add()([parallelLayer_1_toadd, parallelLayer_1])
        parallelLayer_1 = Dense(5, activation='relu')(parallelLayer_1)

        ######### Parallel Layer 2
        parallelLayer_2_toadd = Dense(12, activation='relu')(layer)
        parallelLayer_2 = Dense(10, activation='relu')(parallelLayer_2_toadd)
        parallelLayer_2 = Dense(10, activation='relu')(parallelLayer_2)
        parallelLayer_2 = Dense(12, activation='relu')(parallelLayer_2)
        parallelLayer_2 = Add()([parallelLayer_2_toadd, parallelLayer_2])
        parallelLayer_2 = Dense(5, activation='relu')(parallelLayer_2)
        ######### Parallel Layer 3
        parallelLayer_3_toadd = Dense(12, activation='relu')(layer)
        parallelLayer_3 = Dense(10, activation='relu')(parallelLayer_3_toadd)
        parallelLayer_3 = Dense(10, activation='relu')(parallelLayer_3)
        parallelLayer_3 = Dense(12, activation='relu')(parallelLayer_3)
        parallelLayer_3 = Add()([parallelLayer_3_toadd, parallelLayer_3])
        parallelLayer_3 = Dense(5, activation='relu')(parallelLayer_3)

        # concatenate the parallel layers
        layer = Concatenate()([parallelLayer_1, parallelLayer_2, parallelLayer_3])
        layer=Dense(18,  activation='relu')(layer)
        layer = Dense(20,  activation='relu')(layer)
        layer = Dense(12, activation='relu')(layer)
        layer = Dense(8,  activation='relu')(layer)
        l2r=0.001
        layer = Dense(5,kernel_regularizer=regularizers.l2(l2r), activation='relu')(layer)
        output = Dense(self.outputDim,kernel_regularizer=regularizers.l2(l2r), activation='relu')(layer)
        self.model = Model(input, output)
        self.model.compile(optimizer=Adam(self.learningRate), loss='mse', metrics=['mae','mape'])
        #loss='mse' metrics=['mae']

    def buildParallelModelWithoutRes(self):
        input = Input(shape=(self.inputDim,))
        layer = BatchNormalization()(input)
        # regularPara = 0
        regularPara = 0.000
        layer = Dense(8, kernel_regularizer=regularizers.l1_l2(l1=regularPara, l2=regularPara), activation='relu')(
            layer)
        layer = Dense(10, activation='relu')(layer)
        layer = Dense(12, activation='relu')(layer)
        ######### Parallel Layer 1
        parallelLayer_1_toadd = Dense(12, activation='relu')(layer)
        parallelLayer_1 = Dense(10, activation='relu')(parallelLayer_1_toadd)
        parallelLayer_1 = Dense(10, activation='relu')(parallelLayer_1)
        parallelLayer_1 = Dense(8, activation='relu')(parallelLayer_1)
        #parallelLayer_1 = Add()([parallelLayer_1_toadd, parallelLayer_1])
        parallelLayer_1 = Dense(5, activation='relu')(parallelLayer_1)

        ######### Parallel Layer 2
        parallelLayer_2_toadd = Dense(12, activation='relu')(layer)
        parallelLayer_2 = Dense(10, activation='relu')(parallelLayer_2_toadd)
        parallelLayer_2 = Dense(10, activation='relu')(parallelLayer_2)
        parallelLayer_2 = Dense(8, activation='relu')(parallelLayer_2)
        #parallelLayer_2 = Add()([parallelLayer_2_toadd, parallelLayer_2])
        parallelLayer_2 = Dense(5, activation='relu')(parallelLayer_2)
        ######### Parallel Layer 3
        parallelLayer_3_toadd = Dense(12, activation='relu')(layer)
        parallelLayer_3 = Dense(10, activation='relu')(parallelLayer_3_toadd)
        parallelLayer_3 = Dense(10, activation='relu')(parallelLayer_3)
        parallelLayer_3 = Dense(8, activation='relu')(parallelLayer_3)
        #parallelLayer_3 = Add()([parallelLayer_3_toadd, parallelLayer_3])
        parallelLayer_3 = Dense(5, activation='relu')(parallelLayer_3)

        # concatenate the parallel layers
        layer = Concatenate()([parallelLayer_1, parallelLayer_2, parallelLayer_3])
        layer = Dense(18, activation='relu')(layer)
        layer = Dense(20, activation='relu')(layer)
        layer = Dense(12, activation='relu')(layer)
        layer = Dense(8, activation='relu')(layer)
        l2r = 0.000
        layer = Dense(5, kernel_regularizer=regularizers.l2(l2r), activation='relu')(layer)
        output = Dense(self.outputDim, kernel_regularizer=regularizers.l2(l2r), activation='relu')(layer)
        self.model = Model(input, output)
        #self.model.compile(optimizer=Adam(self.learningRate), loss='mse', metrics=['mae', 'mape'])
        self.model.compile(optimizer=Adamax(self.learningRate), loss='mse', metrics=['mae', 'mape'])
        # loss='mse' metrics=['mae']

    def scheduler(self,epoch):
        # 每隔100个epoch，学习率减小为原来的9/10
        if epoch % epoch_div == 0 and epoch != 0 and epoch>=epoch_mt:
            lr = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr, lr * lr_dec)
            print("lr changed to {}".format(lr * lr_dec))
        return K.get_value(self.model.optimizer.lr)

    def runModel2(self,index=None):
        batchsize = batchsize
        reduce_lr = LearningRateScheduler(self.scheduler)
        logger=r'tmp_logger_%d_cross_%d.csv'%(self.traindataNum,index)
        history = self.model.fit(x=self.trainX, y=self.trainY, validation_data=(self.valX, self.valY),
                                 batch_size=batchsize, epochs=self.itera,verbose=0,
                                 callbacks=[CSVLogger(filename=logger,separator=",", append=False),
                                            reduce_lr])
        self.history=history
        name=history.history.keys()
        val_name = [x for x in name if x.startswith('val')]
        loss_name=list(name-val_name)
        self.val_loss = history.history[val_name[0]]
        self.val_mean_absolute_error = history.history[val_name[1]]
        self.loss = history.history[loss_name[0]]
        self.mean_absolute_error = history.history[loss_name[1]]
        return history

    def testModel(self):

        predict_result = self.model.predict(x=self.testX.values)
        real_result = self.testY.values
        self.MAE = np.abs(predict_result - real_result).mean(axis=0)
        self.MSE = np.power(predict_result - real_result, 2).mean(axis=0)
        self.MAPE = np.abs((real_result - predict_result) / predict_result).mean(axis=0)
        self.comparedResult = np.hstack((predict_result, real_result))
        return self.comparedResult, self.MAE, self.MSE, self.MAPE

    def calculMetrics(self):
        Ib={}
        Ib['MAE']=np.abs(self.comparedResult[:,0] - self.comparedResult[:,0+3]).mean()
        Ib['MSE']=np.power(self.comparedResult[:,0] - self.comparedResult[:,0+3], 2).mean()
        Ib['MAPE']=np.abs((self.comparedResult[:,0] - self.comparedResult[:,0+3])
                          / self.comparedResult[:,0+3]).mean()
        Ir={}
        Ir['MAE'] = np.abs(self.comparedResult[:, 1] - self.comparedResult[:, 1 + 3]).mean()
        Ir['MSE'] = np.power(self.comparedResult[:, 1] - self.comparedResult[:, 1 + 3], 2).mean()
        Ir['MAPE'] = np.abs((self.comparedResult[:, 1] - self.comparedResult[:, 1 + 3])
                            / self.comparedResult[:, 1 + 3]).mean()
        It={}
        It['MAE'] = np.abs(self.comparedResult[:, 2] - self.comparedResult[:, 2 + 3]).mean()
        It['MSE'] = np.power(self.comparedResult[:, 2] - self.comparedResult[:, 2 + 3], 2).mean()
        It['MAPE'] = np.abs((self.comparedResult[:, 2] - self.comparedResult[:, 2 + 3])
                            / self.comparedResult[:, 2 + 3]).mean()
        self.modelMetrics=[Ib,Ir,It]
        return Ib,Ir,It

    def getData(self):
        self.trainData, self.testData,self.valData, self.colName = loadData(self.traindataNum)
        self.trainY = self.trainData[self.colName[5:5 + 3]]  # 每次训练只对前三个电流特征进行训练
        self.trainX = self.trainData[self.colName[0:5]]
        self.testY = self.testData[self.colName[5:5 + 3]]
        self.testX = self.testData[self.colName[:5]]
        self.valY = self.valData[self.colName[5:5 + 3]]
        self.valX = self.valData[self.colName[0:5]]

    def getCrossData(self,train_data,val_test_data,colName):
        valData=val_test_data.iloc[:int(val_test_data.shape[0]/2)]
        testData = val_test_data.iloc[int(val_test_data.shape[0] / 2):]
        self.trainData, self.testData, self.valData, self.colName = train_data,testData,valData,colName
        self.trainY = self.trainData[self.colName[5:5 + 3]]  # three features for training only
        self.trainX = self.trainData[self.colName[0:5]]
        self.testY = self.testData[self.colName[5:5 + 3]]
        self.testX = self.testData[self.colName[:5]]
        self.valY = self.valData[self.colName[5:5 + 3]]
        self.valX = self.valData[self.colName[0:5]]

    def plotLossMeanerror(self):
        xrange = range(len(self.loss))
        plt.figure(1)
        plt.plot(xrange, self.loss, label='loss')
        plt.plot(xrange, self.val_loss, label='val_loss')
        plt.legend()
        plt.show()
        plt.figure(2)
        plt.plot(xrange, self.mean_absolute_error, label='mean_absolute_error')
        plt.plot(xrange, self.val_mean_absolute_error, label='val_mean_absolute_error')
        plt.legend()
        plt.show()

    def saveModel(self, path=None):
        if path is None:
            self.model.save(r'feature%d.h5' % self.featureNum)
        else:
            self.model.save(filepath=path)

    def plotEveryFeature(self):
        #plot Isbtm compared result
        plt.figure(3)
        MAPE_Isbtm=1-np.mean(np.abs(self.comparedResult[:,0]-self.comparedResult[:,3])/np.abs(self.comparedResult[:,0]))
        MAE_Isbtm = np.mean(np.abs(self.comparedResult[:, 0] - self.comparedResult[:, 3]))
        MSE_Isbtm = np.power((self.comparedResult[:, 0] - self.comparedResult[:, 3]),2).mean()
        plt.scatter(range(self.comparedResult[:,0].__len__()),self.comparedResult[:,0],color='red',label='Isbtm Predict Value')
        plt.scatter(range(self.comparedResult[:, 3].__len__()), self.comparedResult[:, 3], color='blue', label='Isbtm Real Value')
        plt.legend()
        plt.title('Isbtm MAPE is %.3f\n'%MAPE_Isbtm+
                  'Isbtm MAE is %.3f\n'%MAE_Isbtm+
                  'Isbtm MSE is %.3f\n'%MSE_Isbtm)
        plt.show()
        # plot Isbtm compared result
        plt.figure(4)
        MAPE_Iscr = 1 - np.mean(np.abs(self.comparedResult[:, 1] - self.comparedResult[:, 4]) / np.abs(self.comparedResult[:, 1]))
        MAE_Iscr = np.abs(self.comparedResult[:, 1] - self.comparedResult[:, 4]).mean()
        MSE_Iscr = np.power((self.comparedResult[:, 1] - self.comparedResult[:, 4]),2).mean()
        plt.scatter(range(self.comparedResult[:,1].__len__()),self.comparedResult[:,1],color='red',label='Iscr Predict Value')
        plt.scatter(range(self.comparedResult[:, 4].__len__()), self.comparedResult[:, 4], color='blue', label='Iscr Real Value')
        plt.legend()
        plt.title('Iscr MAPE is %.3f\n'%MAPE_Iscr+
                  'Iscr MAE is %.3f\n'%MAE_Iscr+
                  'Iscr MSE is %.3f\n'%MSE_Iscr)
        plt.show()

        #plot Isctop result
        plt.figure(5)
        MAPE_Isctop = 1 - np.mean(np.abs(self.comparedResult[:, 2] - self.comparedResult[:, 5]) / np.abs(self.comparedResult[:, 2]))
        MAE_Isctop = np.abs(self.comparedResult[:, 2] - self.comparedResult[:, 5]).mean()
        MSE_Isctop = np.power((self.comparedResult[:, 2] - self.comparedResult[:, 5]), 2).mean()
        plt.scatter(range(self.comparedResult[:,2].__len__()),self.comparedResult[:,2],color='red',label='Isctop Predict Value')
        plt.scatter(range(self.comparedResult[:, 5].__len__()), self.comparedResult[:, 5], color='blue', label='Isctop Real Value')
        plt.legend()
        plt.title('Istop MAPE is %.3f\n'%MAPE_Isctop+
                  'Istop MAE is %.3f\n'%MAE_Isctop+
                  'Istop MSE is %.3f\n'%MSE_Isctop)
        plt.show()

def splitCrossData(index=None):
    traindata, testdata, valdata, colName = loadData(10000)
    data = pd.concat([traindata, testdata, valdata], axis=0, ignore_index=True)
    part=int(12500/5)
    if index==None:
        npdata=data.to_numpy()
        np.random.shuffle(npdata)
        data=pd.DataFrame(npdata,columns=data.columns.to_list())
        val_test_index_start=int(12500*0.8)
        val_test_index_end=12500
    else:
        val_test_index_start = part * (index - 1)
        val_test_index_end = part * index
    val_test_data=data.iloc[val_test_index_start:val_test_index_end]
    train_data=pd.concat([data.iloc[:val_test_index_start],data.iloc[val_test_index_end:]],axis=0,ignore_index=True)
    return train_data,val_test_data,colName



if __name__ == '__main__':
    MAEs,MSEs,MAPEs,his=[],[],[],[]
    it=800
    testNum=[10000]
    id=id # must list or None
    #chooseTestdata()
    for index in id:
        train_data, val_test_data,colName=splitCrossData(index=index)
        for n in testNum:
            featureNum = 2
            outputDim = 3
            NN = solarNN(traindataNum=n,itera=it,inputDim=5, outputDim=outputDim,
                              hiddenLayersNum=25,
                              learningRate=lr,
                              featrueNum=featureNum)
            NN.getCrossData(train_data, val_test_data,colName)
            print('Building model.....')
            NN.buildParallelModelWithoutRes()
            print('Running model......')
            history = NN.runModel2(index=index)
            his.append(history)
            try:
                result, MAE, MSE, MAPE = NN.testModel()
                MAEs.append(MAE)
                MSEs.append(MSE)
                MAPEs.append(MAPE)
            except:
                with open('records.txt', 'a+') as f:
                    f.write('%d dataset can not converged \n'%n)
                print('%d dataset can not converged \n'%n)
                continue
            ibm,irm,itm=NN.calculMetrics()
            m=[ibm,irm,itm]
            for mm in m:
                values=str(list(mm.values())).replace('[','').replace(']','')+'\n'
                with open('records_%d.txt'%it,'a+') as f:
                    f.write(values)
            with open('records_%d.txt'%it,'a+') as f:
                f.write('\n')
