from config.SAConfig import *

import warnings,time
warnings.filterwarnings('ignore')
from sko.SA import SA
from keras import models
import numpy as np
from bestInData import maxindata
import matplotlib.pyplot as plt
import time
from datetime import datetime


FlagMax=False
size_pop=size_pop
lb=lb
ub=ub

model=models.load_model('5to3.h5')

def predictData(model,input):

    return model.predict(np.array([input]))[0]

def finallFunction(input):
    global bestResult,bestPredictCircuit

    hglass, hito, hmapbi3, hpcbm, hpedot = input
    current=predictData(model,input=[hglass,hito,hmapbi3,hpcbm,hpedot])
    iscbtm,iscr,isctop = current[0],current[1],current[2]
    ka = isctop - iscbtm
    result = weight[0] * iscbtm + weight[2] * isctop - weight[1] * iscr - weight[3] * np.abs(ka)

    if result>bestResult:
        bestResult=result
        bestPredictCircuit=[iscbtm, iscr, isctop, ka]

    if FlagMax:
        return result
    else:
        return -result

def SAmethod(max_iter=50):
    T,T_min,q=100,1e-250,0.99
    x0 = np.array(lb)+0.1
    start_time=datetime.now()
    sa = SA(func=finallFunction,lb=lb, ub=ub,x0=x0,L=max_iter,T=T,T_min=T_min,q=q,debug=True,max_iter=max_iter) #只求解最小值
    end_time=datetime.now()
    print('Itreation numbers is {}, weight is '.format(max_iter)
          +str(weight)+' using time is {}'.format((end_time-start_time).total_seconds()))
    best_x, best_y = sa.run()
    Y_history = sa.best_y_history
    tm=time.localtime(time.time())
    tm_str='{}-{}-{} {}:{}:{} '.format(tm[0],tm[1],tm[2],tm[3],tm[4],tm[5])

    return best_x,best_y,Y_history,sa

weights=[weight]
for i in range(weights.__len__()):
    weight=weights[i]
    bestResult = -1e10
    max_iter=max_iter
    bestPredictCircuit = []
    start_time = datetime.now()
    best_x,best_y,Y_history,sa=SAmethod(max_iter=max_iter)
    end_time = datetime.now()
    print('Itreation numbers is {}, weight is '.format(max_iter)
          + str(weight) + ' using time is {}'.format((end_time - start_time).total_seconds()))

