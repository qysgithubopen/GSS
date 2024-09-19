import torch
import random

def randn_sampling(maxint, sample_rate):
    B, C, D, H, W = maxint
    sample_size = int(D*H*W*sample_rate)
    ramdom_index = torch.LongTensor([[],[],[],[],[]])
    # ramdom_index = torch.tensor(ramdom_index, dtype=torch.int)
    for i in range(B):
        b = torch.ones((sample_size,1)).int() - 1 + i
        d = torch.randint(D, size=(sample_size, 1))
        h = torch.randint(H, size=(sample_size, 1))
        w = torch.randint(W, size=(sample_size, 1))
        for j in range(C):
            c = torch.ones((sample_size,1)).int() - 1 + j
            tmpindex = torch.cat([b.long(), c.long(), d.long(), h.long(), w.long()], dim=1).t()
            # tmpindex = torch.tensor(tmpindex, dtype=torch.long)
            ramdom_index = torch.cat([ramdom_index,tmpindex], dim=1)
    return ramdom_index.long()


def random_mask(soft_target, pre, sample_rate):
    B, C, D, H, W = soft_target.size()
    ramdom_index = randn_sampling((B, C, D, H, W), sample_rate)
    ramdom_index_t = torch.split(ramdom_index, 1, dim=0)
    soft_target_mask = soft_target.index_put(ramdom_index_t, values=torch.tensor(0.))
    pre_mask = pre.index_put(ramdom_index_t, values=torch.tensor(0.))

    return soft_target_mask, pre_mask

