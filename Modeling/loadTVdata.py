import numpy as np
import pandas as pd

def chooseTestdata(filename='WHUT_data_more.npy'):
    colName=[]
    with open('parametername.txt','r') as f:
        lines=f.readlines()
        for name in lines:
            colName.append(name.replace('\n',''))
    arr=np.load(filename)
    np.random.shuffle(arr)
    np.save('testData.npy',arr[10500:12500])
    np.save('allTrainValData.npy',arr[:10500])

def loadData(numOfData=None,filename='WHUT_data_more.npy'):
    colName=[]
    with open('parametername.txt','r') as f:
        lines=f.readlines()
        for name in lines:
            colName.append(name.replace('\n',''))
    testdata=np.load('testData.npy') # 2000 of test have been split
    allTrainData=np.load('allTrainValData.npy')
    assert (numOfData!=None and numOfData<=10000)
    #np.random.shuffle(allTrainData)
    testdata=pd.DataFrame(testdata,columns=colName)
    traindata=pd.DataFrame(allTrainData[:numOfData],columns=colName)
    valdata = pd.DataFrame(allTrainData[10000:10500], columns=colName) #validation data
    #data=pd.DataFrame(arr,columns=colName)
    # data=pd.DataFrame(np.load(filename),columns=colName)
    return traindata,testdata,valdata,colName

if __name__=='__main__':
    chooseTestdata()
    traindata,testdata,valdata,colName=loadData(200)