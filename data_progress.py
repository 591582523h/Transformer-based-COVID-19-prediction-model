import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



def get_data(input_sequence_length, output_sequence_length, file = 'data/us_covid.xlsx'):
    seq = input_sequence_length + output_sequence_length

    data = pd.read_excel(file)
    confirm = np.array(list(data['ConfirmedChange']))
    if 'china' in file:
      confirm = confirm[500:]
    
    scaler = MinMaxScaler(feature_range=(0, 5)) # china: (0, 5) usa: (0, 1)
    confirm = scaler.fit_transform(confirm.reshape((-1, 1))).tolist()
    # confirm = confirm.tolist()
    data = []
    label = []

    for i in range(len(confirm) - seq):
        data.append(confirm[i:i + input_sequence_length])
        label.append(confirm[i + input_sequence_length: i + seq])
        
    return torch.Tensor(data).to(torch.float32).squeeze(), torch.Tensor(label).to(torch.float32).squeeze()

















