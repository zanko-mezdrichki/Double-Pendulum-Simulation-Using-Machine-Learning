import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, n_hidden=256): #256 neurons for every hidden level
        super(PINN, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(1, n_hidden),  # 1 input (t)
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(), #4 hidden level
            nn.Linear(n_hidden, 4)  # 4 outputs (theta1, dtheta1, theta2, dtheta2)
        )
        
         #Xavier initialization for stability
        for m in self.net.modules():
           if isinstance(m, nn.Linear):
               nn.init.xavier_normal_(m.weight)
               nn.init.zeros_(m.bias)

    def forward(self, t):
        return self.net(t)