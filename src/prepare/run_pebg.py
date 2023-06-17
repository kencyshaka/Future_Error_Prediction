"""
    bipartite graph node embedding --> item embedding, skill embedding
       (item is question.)

    plus: item difficutly features

    three different feature use PNN to fuse, and with the help of auxilary target
"""

import os
import torch
import torch.optim as optim
import numpy as np
import math
from scipy import sparse
from src.prepare.pebg_model import PEBGModel
import wandb

# load data
wandb.login()


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_dataset(data_folder):
    pro_skill_coo = sparse.load_npz(os.path.join(data_folder, 'pro_skill_sparse.npz'))
    skill_skill_coo = sparse.load_npz(os.path.join(data_folder, 'skill_skill_sparse.npz'))
    pro_pro_coo = sparse.load_npz(os.path.join(data_folder, 'pro_pro_sparse.npz'))
    pro_num, skill_num = pro_skill_coo.shape
    print('problem number %d, skill number %d' % (pro_num, skill_num))
    print('pro-skill edge %d, pro-pro edge %d, skill-skill edge %d' % (pro_skill_coo.nnz, pro_pro_coo.nnz, skill_skill_coo.nnz))

    pro_skill_dense = pro_skill_coo.toarray()
    pro_pro_dense = pro_pro_coo.toarray()
    skill_skill_dense = skill_skill_coo.toarray()

    pro_feat = np.load(os.path.join(data_folder, 'pro_feat.npz'))['pro_feat']
    print('problem feature shape', pro_feat.shape)
    print(pro_feat[:, 0].min(), pro_feat[:, 0].max())
    print(pro_feat[:, 1].min(), pro_feat[:, 1].max())

    diff_feat_dim = pro_feat.shape[1] - 1

    return pro_num, skill_num, diff_feat_dim, pro_skill_dense, pro_pro_dense, skill_skill_dense, pro_feat


def main(data_folder, saved_model_folder, data_folder_output):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model configuration
    embed_dim = 64
    hidden_dim = 128
    keep_prob = 0.5
    lr = 0.001
    bs = 10    # 256
    epochs = 60  # 200
    model_flag = 0
    con_sym = '_'

    run = wandb.init(
        # Set the project where this run will be logged
        project="PEBG",
        #     # Track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "epochs": epochs,
            "hidden_dim": hidden_dim,
            "batch": bs,
            "embed_dim": embed_dim,
            "framework": "pytorch",
            "device": device,
        })

    # Load the dataset
    pro_num, skill_num, diff_feat_dim, pro_skill_dense, pro_pro_dense, skill_skill_dense, pro_feat = load_dataset(
        data_folder)

    # Create PyTorch tensors
    pro_skill_dense = torch.Tensor(pro_skill_dense).to(device)
    pro_pro_dense = torch.Tensor(pro_pro_dense).to(device)
    skill_skill_dense = torch.Tensor(skill_skill_dense).to(device)
    pro_feat = torch.Tensor(pro_feat).to(device)
    skill_skill_dense = skill_skill_dense.to(device)

    # Initialize the model
    model = PEBGModel(pro_num, skill_num, diff_feat_dim, embed_dim, hidden_dim, keep_prob)
    model.to(device)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model
    train_steps = math.ceil(pro_num / bs)

    for epoch in range(model_flag, epochs):
        train_loss = 0

        for m in range(train_steps):
            b, e = m * bs, min((m + 1) * bs, pro_num)

            batch_pro = torch.arange(b, e).to(device)
            batch_pro_skill_targets = pro_skill_dense[b:e, :].to(device)
            batch_pro_pro_targets = pro_pro_dense[b:e, :].to(device)
            batch_diff_feat = pro_feat[b:e, :-1].to(device)
            batch_auxiliary_targets = pro_feat[b:e, -1].to(device)

            optimizer.zero_grad()

            loss, pro_final_embed = model(batch_pro, batch_diff_feat, batch_pro_skill_targets, batch_pro_pro_targets,
                                          skill_skill_dense, batch_auxiliary_targets, device)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss /= train_steps

        wandb.log({"epoch": epoch, "loss": train_loss})

        print("epoch %d, loss %.4f" % (epoch, train_loss))

        if epoch + 1 in [5, 10, 20, 50, 100, 200, 500]:
            torch.save(model.state_dict(), os.path.join(saved_model_folder, 'pebg_p%d.ckpt' % (epoch + 1)))
            print("the path is ", saved_model_folder)

    print('finish training')

    # Store pretrained pro skill embedding
    pro_repre, skill_repre = model.pro_embedding_matrix.weight.detach().cpu().numpy(), model.skill_embedding_matrix.weight.detach().cpu().numpy()

    batch_pro = torch.arange(pro_num).to(device)
    batch_diff_feat = pro_feat[:, :-1].to(device)
    batch_auxiliary_targets = pro_feat[:, -1].to(device)
    batch_pro_skill_targets = pro_skill_dense.to(device)
    batch_pro_pro_targets = pro_pro_dense.to(device)

    # tf_keep_prob: [1.]

    with torch.no_grad():
        _, pro_final_repre = model(batch_pro, batch_diff_feat, batch_pro_skill_targets, batch_pro_pro_targets,
                                   skill_skill_dense, batch_auxiliary_targets, device)
    # pro_final_repre = model.pro_final_embed(feed_dict)
    pro_final_repre = pro_final_repre.cpu().numpy()

    print("the shape of problem embedding and skills emebedings")
    print(pro_repre.shape, skill_repre.shape)

    print("the shape of problem embedding and skills emebedings after final")
    print(pro_final_repre.shape)

    with open(os.path.join(data_folder, 'skill_id_dict.txt'), 'r') as f:
        skill_id_dict = eval(f.read())

    join_skill_num = len(skill_id_dict)
    print('original skill number %d, joint skill number %d' % (skill_num, join_skill_num))

    skill_repre_new = np.zeros([join_skill_num, skill_repre.shape[1]])
    skill_repre_new[:skill_num, :] = skill_repre

    for s in skill_id_dict.keys():
        if con_sym in str(s):
            tmp_skill_id = skill_id_dict[s]
            tmp_skills = [skill_id_dict[int(ele)] for ele in s.split(con_sym)]
            skill_repre_new[tmp_skill_id, :] = np.mean(skill_repre[tmp_skills], axis=0)

    np.savez(os.path.join(data_folder_output, 'embedding_p%d.npz' % epochs), pro_repre=pro_repre,
             skill_repre=skill_repre_new,
             pro_final_repre=pro_final_repre)


if __name__ == '__main__':
    # Set device
    data_folder = '../../data/prepared/question'
    data_folder_output = '../../data/prepared/question/embedding'

    saved_model_folder = os.path.join(data_folder, 'pebg_model')
    if not os.path.exists(saved_model_folder):
        os.mkdir(saved_model_folder)

    if not os.path.exists(data_folder_output):
        os.mkdir(data_folder_output)

    main(data_folder, saved_model_folder, data_folder_output)
