'''
main.ph
'''

import torch
from model.began import *

def main():
    print('Hello, world!')
    # Testing below
    a = torch.Tensor(5,3)
    opt = {'h': 3, 'n':2, 'ngpu':0, 'alpha':1}
    g = BeganGenerator(opt)
    print(g.forward(a))
    d = BeganDiscriminator(opt)
    i = torch.Tensor(5,3,128,128)
    print(d.forward(i))

if __name__ == '__main__':
    main()
