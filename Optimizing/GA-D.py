from config.GAConfig import *

import warnings,time
warnings.filterwarnings('ignore')
from sko.GA import GA
from keras import models
import numpy as np
from bestInData import maxindata
import matplotlib.pyplot as plt
from datetime import datetime

model=models.load_model('5to3.h5')

weight = weight

def calculEffFunc(thickness):
    global bestResult,bestPredictCircuit,bestResultRecords
    hglass, hito, hmapbi3, hpcbm, hpedot =thickness
    thickness=[hglass,hito,hmapbi3,hpcbm,hpedot]
    #########离散问题求解
    if flag=='D':
        currentPro=model.predict(np.array([thickness]))
        Isctop=currentPro[0][0]
        Iscr=currentPro[0][1]
        Iscbtm=currentPro[0][2]
        Ka=Isctop-Iscbtm
        Eff=weight[0]*Isctop-weight[1]*Iscr+weight[2]*Iscbtm-weight[3]*np.abs(Ka)
        if Eff>bestResult:
            bestResultRecords.append(Eff)
            bestResult=Eff
            bestPredictCircuit = [Iscbtm, Iscr, Isctop, Ka]

        if abs(thickness[0] * 10 - round(thickness[0] * 10)) > 0.1 \
                or abs(thickness[1] * 10 - round(thickness[1] * 10)) > 0.1 \
                or abs(thickness[2] * 10 - round(thickness[2] * 10)) > 0.1 \
                or abs(thickness[3] * 10 - round(thickness[3] * 10)) > 0.1 \
                or abs(thickness[4] * 10 - round(thickness[4] * 10)) > 0.1:
            #Eff = -5
            pen=-5
            bestResult = -(Eff+pen)
            bestPredictCircuit = [0, 0, 0, 0]
            return -(Eff+pen)

        return -Eff

def GAmethod(max_iter=50,size_pop=500):
    lb = lb
    ub = ub
    ga = GA(func=calculEffFunc, n_dim=5, size_pop=size_pop, max_iter=max_iter, lb=lb, ub=ub, precision=1e-7,prob_mut=prob_mut) #只求解最小值

    best_x, best_y = ga.run()
    Y_history = ga.all_history_Y
    return best_x,best_y,Y_history,ga.generation_best_Y

flag='D'
bestResult=-1e10
bestResultRecords=[]
maxiter=max_iter
size_pop = size_pop
st=time.time()
start_time = datetime.now()
best_x,best_y,Y_history,generation_best_Y=GAmethod(max_iter=maxiter)
end_time=datetime.now()
print('Cost ',(end_time-start_time).total_seconds())
en=time.time()
costtime=en-st
bestindata,besteffindata=maxindata(weight=weight)
if flag=='D':
    modeFlag=flag.replace('D','Discrete Mode ')
else:
    modeFlag = flag.replace('C', 'Continue Mode ')
GAconfigInfo=modeFlag+'Maxiter=%d '%maxiter+'size_pop=%d '%size_pop+'cost time is '+str(costtime)
bestInfo=' weight is '+str(weight)+' best current is '+\
         str(bestPredictCircuit)+' best structure is '+str(best_x)+' best efficiency is '+str(best_y)
bestindataInfo=' best structure and current in data is '+str(bestindata)\
               +' best Efficiency in data is '+str(besteffindata)
localtimeInfo='-'.join(str(time.localtime()[0:6])[1:-1].replace(' ','').split(','))+' '
Info=localtimeInfo+GAconfigInfo+bestInfo+bestindataInfo+'\n'
with open('GArecord.txt','a') as f:
    f.write(Info)
