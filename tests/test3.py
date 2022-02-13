class A:
    def __init__(self) -> None:
        self.att = 0
        self.batt = 1
        self.li = []
        
    def fun(self):
        self.other(2)
    
    def other(self, num):
        for i in range(num):
            self.li.append(i)
            
class B(A):
    def __init__(self) -> None:
        super().__init__()
        
    # @Override
    def other(self, num):
        for i in range(num):
            self.li.append(-i)

import torch

w= torch.zeros(3,2)
print(torch.nn.init.uniform_(w))

w= torch.zeros(3,2)
print(torch.nn.init.uniform_(w))