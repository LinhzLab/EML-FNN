import torch.nn as nn
import torch
import Network
import numpy as np






def Test(X, y, intercept, save_path, dropout_p, width):
    input_dim = X.shape[1]
    EML_FNN = Network.FNN(input_dim=input_dim, init_weights=True,dropout_p=dropout_p,width=width)
    EML_FNN.load_state_dict(torch.load(save_path))
    loss_function = nn.MSELoss(reduction='mean')
    EML_FNN.eval()
    with torch.no_grad():
        y_pred = EML_FNN(X) + intercept
        PE = loss_function(y_pred , y).cpu().numpy()

    return y_pred, PE


def Validate(X, intercept, save_path, dropout_p, width):
    input_dim = X.shape[1]
    EML_FNN = Network.FNN(input_dim=input_dim, init_weights=True,dropout_p=dropout_p,width=width)
    EML_FNN.load_state_dict(torch.load(save_path))
    EML_FNN.eval()
    with torch.no_grad():
        g_X_hat = EML_FNN(X) + intercept

    return g_X_hat.cpu().numpy().flatten()


def Rmse(g_X,g_X_hat):
    g_X_hat_mean = np.mean(g_X_hat, axis=0)
    
    bias = np.mean(np.abs(g_X-g_X_hat_mean.T))
    sd = np.mean(np.std(g_X_hat,axis=0))
    Rmse = np.sqrt(np.power(bias, 2) + np.power(sd, 2))

    return bias, sd, Rmse