from loadData import loadData
import numpy as np

def maxindata(weight=[1,3,1,14]):
    data,name=loadData()
    Eff=data[name[5]]*weight[0]-data[name[6]]*weight[1]+data[name[7]]*weight[2]-data[name[8]].abs()*weight[3]
    return data.values[Eff.values.argmax()],Eff.values.max()

if __name__=='__main__':
    #result,eff=maxindata()
    data,name=loadData()


