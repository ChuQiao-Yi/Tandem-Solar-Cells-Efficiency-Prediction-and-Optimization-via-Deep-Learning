from config.PSOConfig import *

import numpy as np
from keras.models import load_model
from pso import pso
import time
from datetime import datetime


FlagMax=False
size_pop=swarmsize
lb=lb
ub=ub

model=load_model('5to3.h5')

def predictData(model,input):

    return model.predict(np.array([input]))[0]

def finallFunction(input):
    global bestResult,bestPredictCircuit,printbestResult

    hglass, hito, hmapbi3, hpcbm, hpedot = input
    current = predictData(model, input=[hglass, hito, hmapbi3, hpcbm, hpedot])
    iscbtm, iscr, isctop = current[0], current[1], current[2]
    ka = isctop - iscbtm
    result = weight[0] * iscbtm + weight[2] * isctop - weight[1] * iscr - weight[3] * np.abs(ka)

    if result>bestResult: # result performs more well, if result is higher (result>0)
        bestResult=result
        bestPredictCircuit=[iscbtm, iscr, isctop, ka]
        printbestResult=iscbtm + isctop -  iscr - np.abs(ka)

    if FlagMax:
        return result
    else:
        return -result

def PSOmethod(max_iter=50):
    start_time=datetime.now()
    xopt1, fopt1,p, fp =pso(func=finallFunction,lb=lb,ub=ub,swarmsize=swarmsize, omega=omega,
                      phip=phip, phig=phig, maxiter=max_iter,debug=True,particle_output=True)
    end_time = datetime.now()
    print('Itreation numbers is {}, weight is '.format(max_iter)
          + str(weight) + ' using time is {}'.format((end_time - start_time).total_seconds()))
    print('The optimum is at:')
    print('    {}'.format(xopt1))
    print('Optimal function value:')
    print('    myfunc: {}'.format(fopt1))
    tm=time.localtime(time.time())
    tm_str='{}-{}-{} {}:{}:{} '.format(tm[0],tm[1],tm[2],tm[3],tm[4],tm[5])

    return xopt1, fopt1,p, fp

iteration=[500]
weights=[weight]
for i in range(weights.__len__()):
    weight=weights[i]
    for it in iteration:
        bestResult = -1e10
        bestPredictCircuit = []
        start_time = datetime.now()
        g, fg, p, fp=PSOmethod(max_iter=it)
        end_time = datetime.now()
        print('Itreation numbers is {}, weight is '.format(it)
              + str(weight) + ' using time is {}'.format((end_time - start_time).total_seconds()))