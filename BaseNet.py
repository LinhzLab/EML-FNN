import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm

import Network




import random


class Cauchy(nn.Module):
    def __init__(self,zeta):
        super(Cauchy, self).__init__()
        self.zeta=zeta


    def forward(self,x,y):
        loss=torch.log(1+self.zeta**2*torch.pow(y-x,2))
        loss = torch.mean(loss)
       
        return loss


def BaseNet(X, y, base_path, dropout_p, width, learning_rate):


    totalmanualSeed = random.randint(1, 10000)
    random.seed(totalmanualSeed)

    sample_num = X.shape[0]
    input_dim = X.shape[1]


    Base_FNN = Network.FNN(input_dim=input_dim, init_weights=True,dropout_p=dropout_p,width=width)
    


    batch_size =  sample_num


    
    train = torch.utils.data.TensorDataset(X, y)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,shuffle=False, num_workers=0)




    
    epochs = 1000

    best_loss = 1e10


    params = [p for p in Base_FNN.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=learning_rate[0], betas=(0.9, 0.99))  
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)
    Loss_function = Cauchy(zeta=1)



    for epoch in range(epochs):
        Base_FNN.train()
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            train_x, train_y= data
            reg = Base_FNN(train_x)
            train_loss = Loss_function(reg, train_y)
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(train_loss)
        if train_loss <= best_loss:
            best_loss = train_loss
            torch.save(Base_FNN.state_dict(),base_path)
                



    return base_path


