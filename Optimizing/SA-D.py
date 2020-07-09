from config.SAConfig import *

import warnings,time
warnings.filterwarnings('ignore')
from sko.SA import SA
from keras.models import load_model
import numpy as np
from bestInData import maxindata
import pandas as pd
from datetime import datetime

model=load_model('5to3.h5')
weight = weight

def calculEffFunc(thickness):
    global bestResult,bestPredictCircuit,bestResultRecords
    hglass, hito, hmapbi3, hpcbm, hpedot =thickness
    thickness=[hglass,hito,hmapbi3,hpcbm,hpedot]
    pen=0

    if abs(thickness[0]*10-round(thickness[0]*10))>0.1 \
        or abs(thickness[1]*10-round(thickness[1]*10))>0.1 \
           or abs(thickness[2]*10-round(thickness[2]*10))>0.1 \
            or abs(thickness[3]*10-round(thickness[3]*10))>0.1 \
            or abs(thickness[4]*10-round(thickness[4]*10))>0.1:
        Eff=-5
        pen=-5
        bestResult = Eff
        bestPredictCircuit = [0, 0, 0, 0]
        #return -Eff

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
    return -(Eff+pen)

def SAmethod(max_iter=300,L=50,T=100,T_min=1e-5,q=0.95):
    lb = lb
    ub = ub
    x0=np.random.random(size=np.array(lb).shape) * (np.array(ub) - np.array(lb)) + np.array(lb)
    sa = SA(func=calculEffFunc,lb=lb, ub=ub,x0=x0,L=L,T=T,T_min=T_min,
            quench=q,debug=True,max_iter=max_iter) #slove lowest

    best_x, best_y = sa.run()
    Y_history = sa.best_y_history # each temperature best solutions
    return best_x,best_y,Y_history,Y_history

flag='D'
bestResult=-1e10
bestResultRecords=[]
max_iter=max_iter
L=L
T= T
T_min=T_min
q=q

for ite in range(1):
    st=time.time()
    start_time = datetime.now()
    best_x,best_y,Y_history,generation_best_Y=SAmethod(L=L,max_iter=max_iter,T=T,T_min=T_min,q=q)
    end_time = datetime.now()
    print('Itreation numbers is {}, weight is '.format(max_iter)
          + str(weight) + ' using time is {}'.format((end_time - start_time).total_seconds()))
    en=time.time()
    costtime=en-st
    bestindata,besteffindata=maxindata(weight=weight)
    if flag=='D':
        modeFlag=flag.replace('D','Discrete Mode ')
    else:
        modeFlag = flag.replace('C', 'Continue Mode ')
    GAconfigInfo=modeFlag+'Maxiter=%d '%max_iter+'initial temperature='+str(T)+' min T is '+str(T_min)+' q is '\
                 +str(q)+' cost time is '+str(costtime)
    bestInfo=' weight is '+str(weight)+' best current is '+\
             str(bestPredictCircuit)+' best structure is '+str(best_x)+' best efficiency is '+str(best_y)
    bestindataInfo=' best structure and current in data is '+str(bestindata)\
                   +' best Efficiency in data is '+str(besteffindata)
    localtimeInfo='-'.join(str(time.localtime()[0:6])[1:-1].replace(' ','').split(','))+' '
    Info=localtimeInfo+GAconfigInfo+bestInfo+bestindataInfo+'\n'
    with open('SArecord.txt','a') as f:
        f.write(Info)

