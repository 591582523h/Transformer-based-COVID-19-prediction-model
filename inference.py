import torch
import numpy as np
from model import Transformer_prediction
from plot_module import model_plot
from data_progress import get_data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_sequence_length = 14
output_sequence_length = 1
file = 'data/china_pred.xlsx'

data, label = get_data(input_sequence_length, output_sequence_length, file)
# print(np.shape(data))
data, label = data.to(device), label.to(device)

model = Transformer_prediction()
filename = r'china_model/net_961_0.002793417545.pt'
checkpoint = torch.load(filename, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['net'])
model.to(device)

# model.eval()

with torch.no_grad():
    # print(np.shape(data))
    output = model(data.unsqueeze(1), device).squeeze()
    output = output.to(device)
    pred_seq = output.cpu().numpy().tolist()
    

model_plot(data, label, pred_seq, device, file)