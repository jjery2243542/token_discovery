import torch 
import torch.nn.functional as F

def cc(net):
    if torch.cuda.is_available():
        return net.cuda()
    else:
        return net

def onehot(input_x, encode_dim=514):
    input_x = input_x.type(torch.LongTensor).unsqueeze(2)
    return cc(torch.zeros(input_x.size(0), input_x.size(1), encode_dim).scatter_(-1, input_x, 1))

def sample_gumbel(size):
    eps = 1e-20
    noise = torch.rand(size)
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    return noise

def gumbel_softmax_sample(logits, temperature=1.):
    y = logits + cc(sample_gumbel(logits.size()))
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=1., hard=False):
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        _, max_ind = torch.max(y, dim=-1)
        y_hard = torch.zeros_like(y).scatter_(-1, max_ind.unsqueeze(-1), 1.0)
        y = (y_hard - y).detach() + y
    return y
