from config.PSOConfig import *

import warnings,time
warnings.filterwarnings('ignore')
from pso import pso
from keras import models
import numpy as np
from bestInData import maxindata
import matplotlib.pyplot as plt
model=models.load_model('5to3.h5')

weight = weight
#weight = [1, 3, 1, 1000]
def calculEffFunc(thickness):
    global bestResult,bestPredictCircuit,bestResultRecords
    hglass, hito, hmapbi3, hpcbm, hpedot =thickness
    thickness=[hglass,hito,hmapbi3,hpcbm,hpedot]
    pen=0
    if flag=='D':
        if abs(thickness[0] * 10 - round(thickness[0] * 10)) > 0.1 \
            or abs(thickness[1] * 10 - round(thickness[1] * 10)) > 0.1 \
            or abs(thickness[2] * 10 - round(thickness[2] * 10)) > 0.1 \
            or abs(thickness[3] * 10 - round(thickness[3] * 10)) > 0.1 \
            or abs(thickness[4] * 10 - round(thickness[4] * 10)) > 0.1:
            #Eff=-5
            pen=-5
            bestResult = Eff
            bestPredictCircuit = [0, 0, 0, 0]



    currentPro=model.predict(np.array([thickness]))
    Isctop=currentPro[0][0]
    Iscr=currentPro[0][1]
    Iscbtm=currentPro[0][2]
    Ka=Isctop-Iscbtm
    Eff=weight[0]*Isctop-weight[1]*Iscr+weight[2]*Iscbtm-weight[3]*np.abs(Ka)
    Eff=Eff+pen
    if Eff>bestResult:
        bestResultRecords.append(Eff)
        bestResult=Eff
        bestPredictCircuit = [Iscbtm, Iscr, Isctop, Ka]
    return -Eff

def PSOmethod(max_iter=50,swamsize=500):
    lb = lb
    ub = ub
    #omega=0.8, phip=0.8, phig=0.8
    g, fg, p, fp = pso(func=calculEffFunc,lb=lb,ub=ub,swarmsize=swamsize, omega=omega, phip=phip, phig=phig,
                       maxiter=max_iter,debug=True,particle_output=True)

    return g, fg, p, fp

flag='D'
bestResult=-1e10
bestResultRecords=[]
maxiter=max_iter
size_pop = swarmsize
st=time.time()
g, fg, p, fp=PSOmethod(max_iter=maxiter,swamsize=size_pop)
best_x,best_y=g,fg
en=time.time()
costtime=en-st
bestindata,besteffindata=maxindata(weight=weight)
if flag=='D':
    modeFlag=flag.replace('D','Discrete Mode ')
else:
    modeFlag = flag.replace('C', 'Continue Mode ')
PSOconfigInfo=modeFlag+'Maxiter=%d '%maxiter+'swam size=%d '%size_pop+'cost time is '+str(costtime)

bestCu=model.predict(np.array([best_x]))
eff=weight[0]*bestCu[0][0]-weight[1]*bestCu[0][1]+weight[2]*bestCu[0][2]-weight[3]*np.abs(bestCu[0][0]-bestCu[0][2])
bestInfo=' weight is '+str(weight)+' best current is '+\
         str(bestCu)+' best structure is '+str(best_x)+' best Efficiency is '+str(eff)
# bestInfo=' weight is '+str(weight)+' best current is '+\
#          str(bestPredictCircuit)+' best structure is '+str(best_x)+' best Efficiency is '+str(-best_y)
bestindataInfo=' best structure and current in data is '+str(bestindata)\
               +' best Efficiency in data is '+str(besteffindata)
localtimeInfo='-'.join(str(time.localtime()[0:6])[1:-1].replace(' ','').split(','))+' '
Info=localtimeInfo+PSOconfigInfo+bestInfo+bestindataInfo+'\n'
with open('PSOrecord.txt','a') as f:
    f.write(Info)


