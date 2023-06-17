# Define the PEBG model

import torch
import torch.nn as nn

from PNN_2 import pnn1

class PEBGModel(nn.Module):
    def __init__(self, pro_num, skill_num, diff_feat_dim, embed_dim, hidden_dim, keep_prob):
        super(PEBGModel, self).__init__()
        self.pro_embedding_matrix = nn.Embedding(pro_num, embed_dim)
        self.skill_embedding_matrix = nn.Embedding(skill_num, embed_dim)
        self.diff_embedding_matrix = nn.Linear(diff_feat_dim, embed_dim)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.keep_prob = keep_prob
        # self.pnn = PNN1(embed_dim, hidden_dim)

        # nn.init.xavier_uniform_(self.pro_embedding_matrix)
        # nn.init.xavier_uniform_(self.skill_embedding_matrix)
        # nn.init.xavier_uniform_(self.pro_pro_targets)

    def forward(self, pro, diff_feat, pro_skill_targets, pro_pro_targets, skill_skill_targets, tf_auxiliary_targets,
                device):
        pro_embed = self.pro_embedding_matrix(pro)
        diff_feat_embed = self.diff_embedding_matrix(diff_feat)

        # Pro-Skill
        pro_skill_logits = torch.flatten(torch.matmul(pro_embed, self.skill_embedding_matrix.weight.t()))
        pro_skill_logits = torch.reshape(pro_skill_logits, pro_skill_targets.shape)
        cross_entropy_pro_skill = nn.BCEWithLogitsLoss()(pro_skill_logits, pro_skill_targets)

        # Pro-Pro
        pro_pro_logits = torch.flatten(torch.matmul(pro_embed, self.pro_embedding_matrix.weight.t()))
        pro_pro_logits = torch.reshape(pro_pro_logits, pro_pro_targets.shape)
        cross_entropy_pro_pro = nn.BCEWithLogitsLoss()(pro_pro_logits, pro_pro_targets)

        # Skill-Skill
        skill_skill_logits = torch.flatten(
            torch.matmul(self.skill_embedding_matrix.weight, self.skill_embedding_matrix.weight.t()))
        skill_skill_logits = torch.reshape(skill_skill_logits, skill_skill_targets.shape)
        cross_entropy_skill_skill = nn.BCEWithLogitsLoss()(skill_skill_logits, skill_skill_targets)

        # Feature fusion
        skill_embed = torch.matmul(pro_skill_targets, self.skill_embedding_matrix.weight) / torch.sum(pro_skill_targets,
                                                                                                      dim=1,
                                                                                                      keepdim=True)
        # pro_final_embed, p = self.pnn([pro_embed, skill_embed, diff_feat_embed])
        pro_final_embed, p = pnn1([pro_embed, skill_embed, diff_feat_embed], self.embed_dim, self.hidden_dim,
                                  self.keep_prob, device)

        # Auxiliary target
        mse = nn.MSELoss()(p, tf_auxiliary_targets)

        loss = mse + cross_entropy_pro_skill + cross_entropy_pro_pro + cross_entropy_skill_skill

        return loss, pro_final_embed
