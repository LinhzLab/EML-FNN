import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm

import Network
import Loss_function



import random




def Train(X, y, bandwidth, transfer_path, base_path, save_path, dropout_p, width, transfer, learning_rate):


    totalmanualSeed = random.randint(1, 10000) 
    random.seed(totalmanualSeed)

    sample_num = X.shape[0]
    input_dim = X.shape[1]


    EML_FNN = Network.FNN(input_dim=input_dim, init_weights=True,dropout_p=dropout_p,width=width)
    Base_FNN = Network.FNN(input_dim=input_dim, init_weights=True,dropout_p=dropout_p,width=width)
    

    if sample_num >= 10000 :
        batch_size =  sample_num // 4
    else:
        batch_size =  sample_num


    EML = Loss_function.EML()

    Base_FNN.load_state_dict(torch.load(base_path))
    Base_FNN.eval()
    with torch.no_grad():
        y_hat = Base_FNN(X)
    epsilon = y - y_hat
    epsilon = epsilon - torch.mean(epsilon)
    Var_bw=Loss_function.Variable_bandwidth(x=epsilon,bandwidth=bandwidth)


    train = torch.utils.data.TensorDataset(X, y, Var_bw)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,shuffle=False, num_workers=0)

    
    epochs = 1000

    best_loss = 1e10

    params = [p for p in EML_FNN.parameters() if p.requires_grad]
    if transfer == True: 
        EML_FNN.load_state_dict(torch.load(transfer_path))
        optimizer = optim.Adam(params, lr=learning_rate[0], betas=(0.9, 0.99))  
    else: 
        optimizer = optim.Adam(params, lr=learning_rate[1], betas=(0.9, 0.99)) 

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)

    for epoch in range(epochs):
        EML_FNN.train()
        train_bar = tqdm(train_loader)
        running_losskde = 0.0
        for step, data in enumerate(train_bar):
            train_x, train_y, train_bw= data
            reg = EML_FNN(train_x)+torch.mean(train_y-EML_FNN(train_x))
            train_loss = EML(reg, train_y, train_bw)
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(train_loss)
        if train_loss <= best_loss:
            best_loss = train_loss
            torch.save(EML_FNN.state_dict(),save_path)
                

    EML_FNN.eval()
    with torch.no_grad():
        intercept = torch.mean(y - EML_FNN(X))


    return save_path, intercept


