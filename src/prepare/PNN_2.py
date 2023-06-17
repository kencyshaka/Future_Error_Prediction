import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper function for the PNN layer

def pnn1(inputs, embed_size, hidden_dim, keep_prob,device):
    inputs = [input_tensor.to(device) for input_tensor in inputs]
    num_inputs = len(inputs)
    num_pairs = int(num_inputs * (num_inputs-1) / 2)

    xw = torch.cat(inputs, 1)
    xw3d = xw.view(-1, num_inputs, embed_size)  # [batch_size, 3, embedding_size]

    row = []
    col = []
    for i in range(num_inputs-1):
        for j in range(i+1, num_inputs):
            row.append(i)
            col.append(j)      # row = [0 0 1]  col = [1 2 2]

    p = xw3d.transpose(0, 1)[row].transpose(0, 1)  # batch * pair * k
    q = xw3d.transpose(0, 1)[col].transpose(0, 1)  # batch * pair * k
    p = p.view(-1, num_pairs, embed_size)
    q = q.view(-1, num_pairs, embed_size)
    ip = (p * q).sum(-1).view(-1, num_pairs)
    l = torch.cat([xw, ip], 1)
    l = l.to(device)

    h = nn.Linear(l.size(1), hidden_dim).to(device)
    h = h(l)
    h = F.relu(h)
    h = F.dropout(h, p=keep_prob)

    p = nn.Linear(h.size(1), 1).to(device)
    p = p(h).view(-1)

    return h, p