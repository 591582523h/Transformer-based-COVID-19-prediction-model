import torch
import numpy as np


class Transformer_prediction(torch.nn.Module):
    def __init__(self, feature_size=14, num_layers=3, drop_out=0):
        super(Transformer_prediction, self).__init__()

        self.decoder_layer = torch.nn.TransformerEncoderLayer(d_model=feature_size, nhead=7, dropout=drop_out)
        self.transformer_decoder = torch.nn.TransformerEncoder(self.decoder_layer, num_layers=num_layers)
        self.fc1 = torch.nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.fc1.bias.data.zero_()
        self.fc1.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, device):

        mask = self._generate_square_subsequent_mask(len(x)).to(device)

        output = self.transformer_decoder(x, mask)
        # print(np.shape(output))
        output = self.fc1(output)
        # print(np.shape(output))
        return output


