# Read Data
data_path = 'test_load_data/MERRA2_400.statD_2d_slv_Nx.'
selected_date = '20191003'
tt_path = data_path + selected_date + '.nc4'
longitude = int(120.982024)
latitude = int(23.973875) 

# Import package
# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For data preprocess
import numpy as np
from netCDF4 import Dataset as nDS
import csv
import os
from datetime import date

myseed = 42069  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

# Utilities
def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def is_leap_year(year):
    if (year % 4) == 0:
        if (year % 100) == 0:
            if (year % 400) == 0:
                return True  
            else:
                return False
        else:
            return True
    else:
        return False

# Dataset
class heatwaveDataset(Dataset):
    def __init__(self, path, mode='test', target_only=False):
        self.mode = mode

        # Init data
        data = torch.zeros([1, 361, 576, 1, features_num], dtype = torch.float64)
        # Recursively read and preprocess daily air temperature data
        year = path[-12: -8]
        month = path[-8: -6]
        day = path[-6: -4]
        if int((year + month + day)) > 20210930:
            year = '2021'
            month = '09'
            day = '30'

        year = int(year)
        month = int(month)
        day = int(day)

        for i in range(1, 5):
            t_year = year
            t_month = month
            t_day = day
            
            t_day = t_day - i
            if t_day <= 0:
                t_month = 12 if t_month-1 == 0 else t_month-1
                if t_month == 12:
                    t_year -= 1
                    t_day = 31 + day - i
                elif t_month == 2 and is_leap_year(t_year):
                    t_day = 29 + day - i
                elif t_month == 2:
                    t_day = 28 + day - i
                elif t_month in [1,3,5,7,8,10]:
                    t_day = 31 + day - i
                elif t_month in [4,6,9,11]:
                    t_day = 30 + day - i

            print([t_year, t_month, t_day])
            nasa_data = nDS(path[:-12] + str(t_year) + str(t_month).rjust(2,'0') + str(t_day).rjust(2,'0') +'.nc4', mode='r')
        
            T2MMAX = torch.tensor(nasa_data.variables['T2MMAX'][:,:,:].astype(float))
            TPRECMAX = torch.tensor(nasa_data.variables['TPRECMAX'][:,:,:].astype(float))

            # Extend to five dimensions
            T2MMAX = torch.unsqueeze((T2MMAX), 3)
            TPRECMAX = torch.unsqueeze((TPRECMAX), 3)
            T2MMAX = torch.unsqueeze((T2MMAX), 4)
            TPRECMAX = torch.unsqueeze((TPRECMAX), 4)

            # Store all data in fifth dimension
            temp_data = torch.cat((TPRECMAX, T2MMAX), 4)
            data = torch.cat((data, temp_data), 3)

        y_axis = 180 - latitude
        x_axis = 288 + longitude
        data = data[0, y_axis, x_axis, 1:, :]
        
        
        # Testing data
        self.data = data

        # Normalize features
        self.data[:, :] = \
                    (self.data[:, :] - self.data[:, :].mean(dim=0, keepdim=True)) \
                    / self.data[:, :].std(dim=0, keepdim=True) 
        
        self.dim = self.data.shape[1]

        print('Finished reading the {} set of heatwave Dataset ({} samples found, each dim = {})'.format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        # Returns one sample at a time
        # For testing (no target)
        return self.data[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)

# Dataloader
def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset = heatwaveDataset(path, mode=mode, target_only=target_only) # Construct dataset
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True) # Construct dataloader
    return dataloader

# Deep Neural Network
class NeuralNet(nn.Module):
    ''' A simple fully-connected deep neural network '''
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64).float(),
            nn.ReLU().float(),
            nn.Linear(64, 1).float()
        )

        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        # TODO: improve model
        return self.net(x.float()).squeeze(1)

    def cal_loss(self, pred, target):
        ''' Calculate loss '''
        # TODO: you may implement L1/L2 regularization here
        return self.criterion(pred, target)

# Testing
def test(load_set, model, device):
    model.eval()                                # set model to evalutation mode
    preds = []
    for x in load_set:                          # iterate through the dataloader
        x = x.to(device)                        # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            preds.append(pred.detach().cpu())   # collect prediction
    preds = torch.cat(preds, dim=0).numpy()     # concatenate all predictions and convert to a numpy array
    return preds

# Setup Hyperparameter
device = get_device()                 # get the current available device ('cpu' or 'cuda')
os.makedirs('models', exist_ok=True)  # The trained model will be saved to ./models/
target_only = False                   # False: Using Whole features
features_num = 2

config = {
    'batch_size': 24,               # mini-batch size for dataloader
    'model_path': 'models/model.pth'  # your model will be saved here
}

# Load Data & Model
load_set = prep_dataloader(tt_path, 'test', config['batch_size'], target_only=target_only)

model = NeuralNet(load_set.dataset.dim).to(device)
ckpt = torch.load(config['model_path'], map_location='cpu')  # Load your best model
model.load_state_dict(ckpt)

# Testing & Saving Prediction
def save_pred(preds, file):
    ''' Save predictions to specified file '''
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

preds = test(load_set, model, device)  # predict heatwave cases with your model
save_pred(preds, 'pred.csv')         # save prediction file to pred.csv