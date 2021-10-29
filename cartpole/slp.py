import torch
print(torch.cuda.is_available())

class SLP(torch.nn.Module):
    def __init__(self, input_shape, output_shape, device = torch.device("cuda"), n_neurons = 40):
        super(SLP, self).__init__()
        self.device = device
        self.input_shape = input_shape[0]
        
        self.l1 = torch.nn.Linear(self.input_shape, n_neurons)
        self.output = torch.nn.Linear(n_neurons, output_shape)

    def forward(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        x = torch.nn.functional.relu(self.l1(x))
        x = self.output(x)
        return x
    



