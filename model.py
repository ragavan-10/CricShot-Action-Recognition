import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn

CNN_CHANNELS = [32, 64, 128, 256]
CNN_PROJ_DIM = 256
LSTM_HIDDEN  = 256
LSTM_LAYERS  = 1
CLF_HIDDEN   = 128
CLF_DROPOUT  = 0.5


class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        ch = CNN_CHANNELS

        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, ch[0], 3, padding=1),
            nn.BatchNorm2d(ch[0]), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 112 -> 56
            nn.Conv2d(ch[0], ch[1], 3, padding=1),
            nn.BatchNorm2d(ch[1]), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Dropout2d(p=0.2),

            # Block 3: 56 -> 28
            nn.Conv2d(ch[1], ch[2], 3, padding=1),
            nn.BatchNorm2d(ch[2]), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 28 -> 14
            nn.Conv2d(ch[2], ch[3], 3, padding=1),
            nn.BatchNorm2d(ch[3]), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Dropout2d(p=0.2)
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.project = nn.Linear(ch[3] * 4*4, CNN_PROJ_DIM)

    def forward(self, x):
        x = self.features(x)

        x = self.pool(x).flatten(1)
        return self.project(x)

class LSTMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=CNN_PROJ_DIM,
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(p=0.3)
        self._init_weights()
        self.attn_score = nn.Linear(LSTM_HIDDEN * 2, 1)

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        scores = self.attn_score(out)   # (B, T, 1)
        weights = torch.softmax(scores, dim=1)  # (B, T, 1)
        context = (out * weights).sum(dim=1)    # (B, H*2)
        return context

class ClassifierHead(nn.Module):
    """
    Input  : (B, 256)
    Output : (B, num_classes)  logits
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LSTM_HIDDEN*2, CLF_HIDDEN),
            nn.ReLU(),
            nn.Dropout(CLF_DROPOUT),
            nn.Linear(CLF_HIDDEN, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


class CricketActionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn  = CNNEncoder()
        self.lstm = LSTMEncoder()
        self.head = ClassifierHead()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, T, C, H, W = x.shape
        feat = self.cnn(x.view(B * T, C, H, W)).view(B, T, -1)
        context = self.lstm(feat)
        return self.head(context)

