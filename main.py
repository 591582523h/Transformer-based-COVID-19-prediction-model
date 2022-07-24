import torch
import numpy as np
from model import Transformer_prediction
from train_trick import train_epochs
from plot_module import model_plot
from data_progress import get_data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_sequence_length = 14
output_sequence_length = 1

data, label = get_data(input_sequence_length, output_sequence_length, file='data/china_covid.xlsx')

model = Transformer_prediction()
model = model.to(device)
data, label = data.to(device), label.to(device)

criterion = torch.nn.MSELoss().to(device)

lr = 0.0000001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = 1010

model = train_epochs(model, optimizer, criterion, data, label, device, lr, epochs)

model_plot(model, data, label)


