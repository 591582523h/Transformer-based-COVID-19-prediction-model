import torch
import matplotlib.pyplot as plt
import numpy as np


def model_plot(data, label, pred, device, file):

    # print(pred)
    x = np.arange(0, len(pred))
    label = label.squeeze().cpu().numpy().tolist()
    
    time_china = 214
    time_usa = 757
    
    
    plt.ylabel('Newly Confirmed cases')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    plt.plot(x, label, 'b', label='real')
    
    if 'china' in file:
      plt.plot(x[:time_china], pred[:time_china], 'r', label='estimation')
      plt.plot(x[time_china:], pred[time_china:], 'g', label='prediction')
      plt.title('Transformer Prediction In China')
    elif 'us' in file:
      plt.plot(x[:time_usa], pred[:time_usa], 'r', label='estimation')
      plt.plot(x[time_usa:], pred[time_usa:], 'g', label='prediction')
      plt.title('Transformer Prediction In Usa')
    
    plt.legend()
    plt.show()


