import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
print(torch.cuda.is_available())

class DQN(nn.Module):
    """Some Information about MyModule"""
    def __init__(self, lr, n_actions, name, input_dims, dir):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride = 4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride = 2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride = 1)


        self.fc1 = nn.Linear(self.calc_input(input_dims), 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.name = name
        
        self.dir = os.path.join(dir, name)



    
    def calc_input(self, input_dims):
        state = torch.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)

        

        return int(torch.prod(torch.tensor(dims.size())))





    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #conv 3 shape: bathch_size * filters * height * width
        x = x.view(x.size()[0], -1)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x
    
    def save(self):
        print('\n \t saving model \n')
        torch.save(self.state_dict(), self.dir)
    
    def load(self):
        print('\n \t loading model \n')
        self.load_state_dict(torch.load(self.dir))


        