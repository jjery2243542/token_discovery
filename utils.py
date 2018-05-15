import torch 

def cc(net):
    if torch.cuda.is_available():
        return net.cuda()
    else:
        return net
