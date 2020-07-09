import os
import h5py
import numpy as np
import pandas as pd

def readDataFromFile():
    rootDir='WHUT_data_more'
    fileList=os.listdir(rootDir)
    f=lambda x:rootDir+'\\'+x
    fileList=list(map(f,fileList))
    colName=list(h5py.File(fileList[0],'r').keys())
    if not os.path.exists('parametername.txt'):
        with open('parametername.txt','w') as f:
            for name in colName:
                f.write(name+'\n')
    #solarData=np.zeros(shape=(len(fileList),9),)
    solarData=pd.DataFrame(np.zeros(shape=(len(fileList),9)),columns=colName)
    for num,file in enumerate(fileList):
        print('Now is {}/{}'.format(num,len(fileList)))
        data = h5py.File(file, 'r')
        count=0
        for key,val in data.items():
            solarData.values[num][count]=list(val)[0][0]
            count+=1
    solarData[colName[0:5]] = solarData[colName[0:5]] * 10e7
    np.save('%s.npy'%rootDir,solarData)
    print('npy save success!')
    return solarData,colName

def loadData(filename='./Modeling/WHUT_data_more.npy'):
    colName=[]
    with open('./Modeling/parametername.txt','r') as f:
        lines=f.readlines()
        for name in lines:
            colName.append(name.replace('\n',''))
    arr=np.load(filename)
    np.random.shuffle(arr)
    data=pd.DataFrame(arr,columns=colName)
    # data=pd.DataFrame(np.load(filename),columns=colName)
    return data,colName


if __name__=='__main__':
    #data=readDataFromFile()
    solarData, colName=loadData()