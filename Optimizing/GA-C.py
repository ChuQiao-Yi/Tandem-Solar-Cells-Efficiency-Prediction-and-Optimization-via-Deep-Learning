from config.GAConfig import *

from sko.GA import GA
import pandas as pd
import numpy as np
from keras import models
import time
from datetime import datetime



FlagMax=False
size_pop=size_pop
lb=lb
ub=ub

model=models.load_model('5to3.h5')

def predictData(model,input):
    '''
    输入一个list，例如input=[10，20，30，5，5]
    :param model: 选择需要使用的预测模型
    :param input: 输入的结构序列
    :return:
    '''
    return model.predict(np.array([input]))[0]

def finallFunction(input):
    global bestResult,bestPredictCircuit
    '''
    对输入的结构进行预测
    :param input: 输入的结构数据
    :param weight: 每个结果的权重,默认权重相同
    :return:
    '''
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

def GAmethod(max_iter=50):
    start_time=datetime.now()
    ga = GA(func=finallFunction, n_dim=5, size_pop=size_pop, max_iter=max_iter, lb=lb, ub=ub, prob_mut=prob_mut) #只求解最小值
    end_time=datetime.now()
    print('Itreation numbers is {}, weight is '.format(max_iter)
          +str(weight)+' using time is {}'.format((end_time-start_time).total_seconds()))
    best_x, best_y = ga.run()
    Y_history = ga.all_history_Y
    tm=time.localtime(time.time())
    tm_str='{}-{}-{} {}:{}:{} '.format(tm[0],tm[1],tm[2],tm[3],tm[4],tm[5])

    with open(r'GArecordHistory_Y.txt','a') as f:
        for yh in ga.all_history_Y:
            f.write(str(yh)[1:-1]+'\n\n')
    with open(r'GAbestPredictCircuit.txt','a') as f:
        f.write(tm_str+'GA algorithm iteration number is {} best Predict Circuit is '
                .format(max_iter)+str(bestPredictCircuit)[1:-1]+' using time is {}'.format((end_time-start_time).total_seconds())+'\n')
    with open(r'GAbestStructure.txt','a') as f:
        f.write(tm_str+'GA algorithm iteration number is {}, weight is {} best Structure is '
                .format(max_iter,weight)+str(best_x)[1:-1]+' '+'Best result is '
                +str(bestPredictCircuit[0]-bestPredictCircuit[1]+bestPredictCircuit[2]-bestPredictCircuit[3])+'\n')

    return best_x,best_y,Y_history,ga


weights=[weight]
for i in range(weights.__len__()):
    weight=weights[i]
    bestResult = -1e10
    max_iter=max_iter
    bestPredictCircuit = []
    start_time = datetime.now()
    best_x,best_y,Y_history,ga=GAmethod(max_iter=max_iter)
    end_time = datetime.now()
    print('Itreation numbers is {}, weight is '.format(max_iter)
          + str(weight) + ' using time is {}'.format((end_time - start_time).total_seconds()))
    f = lambda x: -1*np.min(x)
    z = list(map(f, Y_history))
    #np.save('GA_best_plot.npy',np.array(z))


