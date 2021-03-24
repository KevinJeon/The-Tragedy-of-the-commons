import numpy as np
import torch as tr
import torch.nn as nn

class CDC(nn.Module):

    def __init__(self, timestep, batch_size, seq_len, num_channel):
        super(CDC, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timetep = timestep
        self.num_hidden = 128
        # Input Size (N, 88, 88, 3)
        self.encoder = nn.Sequential(
                nn.Conv2d(num_chaanel, 512 , kernel_size=10, stride=5, padding=3, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True))
        self.lstm = nn.LSTM(512, 128, num_layers=1, batch_first=True)
        self.linear = nn.Modulelist([nn.Linear(128, 512) for i in range(timestep)])
    def init_hidden(self, batch_size):
        return tr.zeros(1, batch_size, 128)

    def forward(self, obss, h, acts):
        '''
        x : (seq, batch, c, h, w)
        h : (1, batch, hidden)
        act : (seq, batch, num_action)
        '''
        seq, batch, c, h, w = obss.size()
        # random t
        obs = obss.view(seq * batch, c, h, w)
        t = tr.randint(self.seq_len - self.timestep, size=(1,)).long() # range for 0 ~ seq - timestep
        z = self.encoder(obss)
        z_ts = tr.empty((self.timestep, batch, 512))
        for i in range(self.timestep):
            z_ts[i] = z[: , t+i, :].view(batch, 512)
        z_lstm = []
        for i in range(self.seq_len + 1):
            out, h = self.lstm(z[i], h)
            z_lstm.append(out)
        c_t = z_lstm[t].squeeze(1)
        c = tr.stack(tensors=z_lstm, dim=0)
        pred = tr.empty((self.timestep, batch, 512)).float()
        for i in range(self.timestep):
            linear = self.linear[i]
            pred[i] = linear(c_t)
        nce = 0
        for i in range(self.timestep):
            loss = tr.mm(z_ts[i], pred[i].transpose(1,0))
            nce += tr.sum(tr.diag(F.log_softmax(loss)))
        nce /= -1 * batch * self.timestep

        return nce, hidden

