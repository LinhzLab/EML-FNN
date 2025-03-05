import numpy as np
from sklearn.model_selection import train_test_split
import random

manualSeed = random.randint(1, 10000)  
random.seed(manualSeed)
np.random.seed(manualSeed)

def X_generation(Sample_num, Xdim):
    X = np.random.uniform(0, 1, (Sample_num, Xdim))
    return X

def betaj_generation(X, fdim, j):
    xdim = X.shape[1]
    beta_j = np.zeros(shape=(xdim,))
    realdim=xdim//fdim//20
    beta_real = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]*realdim)
    beta_real_norm=np.linalg.norm(beta_real, ord=1, axis=None, keepdims=False)
    beta_real = beta_real/beta_real_norm
    beta_j[j*20*realdim:j*20*realdim+20*realdim] = beta_real

    return beta_j


def fun_dim5(X,beta):
    y= np.power(np.matmul(X, beta[:,0]), 3)+np.power(np.matmul(X, beta[:,1]), 2)+np.matmul(X, beta[:,2])+np.abs(np.matmul(X, beta)[:,3])\
      +np.cos(np.matmul(X, beta[:,4]))
    return y


def fun_dim10(X,beta):
    y=np.power(np.matmul(X, beta[:,0]), 3)+np.power(np.matmul(X, beta[:,1]), 2)+np.matmul(X, beta[:,2])+ np.abs(np.matmul(X, beta[:,3]))\
      +np.cos(np.matmul(X, beta[:,4]))+np.sin(np.matmul(X, beta[:,5]))+np.exp(np.matmul(X, beta[:,6]))+np.log(1+np.matmul(X, beta[:,7]))\
      +np.power(np.matmul(X,beta[:,8]), 1/2)+np.power(np.matmul(X, beta[:,9]), 1/3)
    return y

def fun_dim20(X,beta):
    y= np.power(np.matmul(X, beta[:,0]), 5)+np.power(np.matmul(X, beta[:,1]), 4)+ np.power(np.matmul(X, beta[:,2]), 3)+np.power(np.matmul(X, beta[:,3]), 2)\
       +np.matmul(X, beta[:,4])+ np.abs(np.matmul(X, beta[:,5])) +np.abs(np.power(np.matmul(X, beta[:,6]), 3))\
       +np.power(np.matmul(X, beta[:,7]), 1 / 2) + np.power(np.matmul(X, beta[:,8]), 1 / 3)+np.power(np.matmul(X, beta[:,9]), 1 / 4)+np.power(np.matmul(X, beta[:,10]), 1 / 5)-\
       +np.cos(np.matmul(X, beta[:,11]))+np.sin(np.matmul(X, beta[:,12]))+np.sin(np.power(np.matmul(X, beta[:,13]), 2))+np.sin(np.power(np.matmul(X, beta[:,14]), 2))\
       +np.exp(np.matmul(X, beta[:,15]))+np.log(np.matmul(X, beta[:,16])+1)+np.exp(np.power(np.matmul(X,beta[:,17]), 2))+np.log(np.power(np.matmul(X, beta[:,18]), 2)+1)+np.log(np.power(np.matmul(X, beta[:,19]), 1/2)+1)
    return y


def E_normal(Sample_num):
    E = np.random.normal(0, 1, (Sample_num, 1))
    return E

def E_mixgauss(Sample_num):
    E = np.zeros(shape=(Sample_num, 1))
    E[0:int(Sample_num * 0.7)] = np.random.normal(0, 1, (int(Sample_num * 0.7), 1))
    E[int(Sample_num * 0.7):] = np.random.normal(0, 5, (Sample_num - int(Sample_num * 0.7), 1))
    return E

def E_student(Sample_num):
    E = np.random.standard_t(2, (Sample_num, 1))
    return E

def E_heter(Sample_num, X):
    E = np.zeros((Sample_num, 1))
    for i in range(Sample_num):
        E[i]=np.random.normal(0,3*X[i,0]+4*X[i,1])
    return E



def Data_Generation(X, beta, E, fun_dim, train_samples, test_samples):
    
    if fun_dim==20:
        g_X = fun_dim20(X=X, beta=beta)
    elif fun_dim==10:
        g_X = fun_dim10(X=X, beta=beta)
    elif fun_dim==5:
        g_X = fun_dim5(X=X, beta=beta)
        
    g_X=g_X.reshape((g_X.shape[0],1))

    X_train, X_test, g_X_train, g_X_test = train_test_split(X,g_X, train_size=train_samples,test_size=test_samples)
    E_train, E_test = train_test_split(E, train_size=train_samples,test_size=test_samples)
     
    y_train=g_X_train+E_train
    y_test=g_X_test+E_test

    return X_train, X_test, y_train, y_test
