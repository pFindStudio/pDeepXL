import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BiLSTMLinearPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers = n_layers, bidirectional = bidirectional, dropout = dropout, batch_first = True)
        self.fc1 = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.fc2 = nn.ReLU(inplace=True)

    # https://blog.csdn.net/Mr_green_bean/article/details/104713382
    # 肽段类型peptide type，0 for linear, 1 for non-clv, 2 for clv
    def mask_mae_loss(self, padded_y_truth, y_pred, y_length, pep_types):

        if pep_types[0]!=2: # linear or non-clv，只取前4维计算loss、更新梯度，不影响clv的transfer，非常重要！
            padded_y_truth=padded_y_truth[:,:,:4]
            y_pred=y_pred[:,:,:4]

        mask = torch.zeros(padded_y_truth.shape)
        for i,l in enumerate(y_length):
            mask[i,:l,:]=1
        mask = mask.to(device)
        masked_y_pred = y_pred.mul(mask)
        return F.l1_loss(masked_y_pred, padded_y_truth)

    def work(self, packed_peptides):
        outputs, (hidden, cell) = self.lstm(packed_peptides)
        out_unpack, out_len = rnn_utils.pad_packed_sequence(outputs, batch_first=True)
        predictions = self.fc2(self.fc1(out_unpack))
        
        mask = torch.zeros(predictions.shape)
        for i,l in enumerate(out_len):
            mask[i,:l,:]=1
        mask = mask.to(device)
        masked_y_pred = predictions.mul(mask)

        return masked_y_pred,out_len
    
    def forward(self, packed_peptides, padded_y_truth, pep_types):

        # padded_peptides shape: [batch size, peptide len, input dim]
        outputs, (hidden, cell) = self.lstm(packed_peptides)

        out_unpack, out_len = rnn_utils.pad_packed_sequence(outputs, batch_first=True)
        
        # predictions shape: [batch size, peptide len,  output dim]
        predictions = self.fc2(self.fc1(out_unpack))
        loss = self.mask_mae_loss(padded_y_truth,predictions,out_len,pep_types)
        
        return predictions,loss