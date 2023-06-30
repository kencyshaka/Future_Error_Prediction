# -*- coding: utf-8 -*-
# Inspired by jarvis.zhang

import torch
import torch.nn as nn


class c2vRNNModel(nn.Module):
    def __init__(self,config,model_type, input_dim, hidden_dim, layer_dim, output_dim, node_count, path_count, device):
        super(c2vRNNModel, self).__init__()

        self.embed_nodes = nn.Embedding(node_count+2, 100) # adding unk and end
        self.embed_paths = nn.Embedding(path_count+2, 100) # adding unk and end
        self.embed_dropout = nn.Dropout(0.2)
        self.path_transformation_layer = nn.Linear(input_dim+300,input_dim+300)
        self.attention_layer = nn.Linear(input_dim+300,1)
#         self.feature_layer = nn.Linear(300,10)
        self.prediction_layer = nn.Linear(input_dim+300,1)
        self.attention_softmax = nn.Softmax(dim=1)

        if model_type == "Code-DKT":
            input_dimensions = 2*input_dim + 3*config.MAX_CODE_LEN
        elif model_type == "DKT":
            input_dimensions = input_dim
        elif model_type == "P-Code-DKT":
            input_dimensions = 2 * input_dim + 3*config.MAX_CODE_LEN + config.MAX_QUESTION_LEN_partI +config.MAX_QUESTION_LEN_partII  #( score, code, question,reference_soln)
        elif model_type == "R-Code-DKT":    
            input_dimensions = 2 * input_dim + 3*config.MAX_CODE_LEN \
                               + config.MAX_QUESTION_LEN_partI \
                               + config.MAX_QUESTION_LEN_partII \
                               + config.Reference_LEN  #( score, code, question,reference_soln)
        elif model_type == "E-Code-DKT":
            input_dimensions = 2 * input_dim + 3*config.MAX_CODE_LEN \
                               + config.MAX_QUESTION_LEN_partI \
                               + config.MAX_QUESTION_LEN_partII \
                               + config.Reference_LEN  \
                               + output_dim #ErrorID_LEN
                                                    #( score, code, question,reference_soln,errorID)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim

        self.rnn = nn.LSTM(input_dimensions,
                          hidden_dim,
                          layer_dim,
                          batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.sig = nn.Sigmoid()
        self.device = device
        self.model_type = model_type
        self.config = config

    def forward(self, x, evaluating=False):  # shape of input: [batch_size, length, input_dimensions]
#         h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device)  # shape: [num_layers * num_directions, batch_size, hidden_size]
        # correctness vector
        rnn_first_part = x[:, :, :self.input_dim] # (b,l,2q)

        # code vector
        rnn_attention_part = torch.stack([rnn_first_part]*self.config.MAX_CODE_LEN,dim=-2) # (b,l,c,2q)

        c2v_input = x[:, :, self.input_dim:self.config.MAX_CODE_LEN*3+ self.input_dim].reshape(x.size(0), x.size(1), self.config.MAX_CODE_LEN, 3).long() # (b,l,c,3)

        starting_node_index = c2v_input[:,:,:,0] # (b,l,c,1)
        ending_node_index = c2v_input[:,:,:,2] # (b,l,c,1)
        path_index = c2v_input[:,:,:,1] # (b,l,c,1)
        # print("starting_node_index",starting_node_index.shape)
        starting_node_embed = self.embed_nodes(starting_node_index) # (b,l,c,1) -> (b,l,c,ne)
        ending_node_embed = self.embed_nodes(ending_node_index) # (b,l,c,1) -> (b,l,c,ne)
        path_embed = self.embed_paths(path_index) # (b,l,c,1) -> (b,l,c,pe)
        
        full_embed = torch.cat((starting_node_embed, ending_node_embed, path_embed, rnn_attention_part), dim=3) # (b,l,c,2ne+pe+q)
        if not evaluating:
            full_embed = self.embed_dropout(full_embed) # (b,l,c,2ne+pe+2q)
        
        full_embed_transformed = torch.tanh(self.path_transformation_layer(full_embed)) # (b,l,c,2ne+pe+2q)
        context_weights = self.attention_layer(full_embed_transformed) # (b,l,c,1)
        attention_weights = self.attention_softmax(context_weights) # (b,l,c,1)
        code_vectors = torch.sum(torch.mul(full_embed,attention_weights),dim=2) # (b,l,2ne+pe+2q)

        # question vector
        question_vectors = x[:, :, self.config.MAX_CODE_LEN*3 + self.input_dim : self.config.MAX_CODE_LEN*3
                                                                                 + self.input_dim
                                                                                 + self.config.MAX_QUESTION_LEN_partI
                                                                                 + self.config.MAX_QUESTION_LEN_partII] # (b,l,q_embedds)

        # reference vector 
        reference_vectors = x[:, :, self.config.MAX_CODE_LEN*3
                                    + self.input_dim + self.config.MAX_QUESTION_LEN_partI
                                    + self.config.MAX_QUESTION_LEN_partII : self.config.MAX_CODE_LEN*3
                                                                            + self.input_dim
                                                                            + self.config.MAX_QUESTION_LEN_partI
                                                                            + self.config.MAX_QUESTION_LEN_partII
                                                                            + self.config.Reference_LEN ] # (b,l,reference_embedds)
        # errorID vector
        errorID_vectors = x[:, :,  self.config.MAX_CODE_LEN*3
                                   + self.input_dim
                                   + self.config.MAX_QUESTION_LEN_partI
                                   + self.config.MAX_QUESTION_LEN_partII
                                   + self.config.Reference_LEN : ] # (b,l,errorID_embedds)

        if self.model_type == "Code-DKT":
            rnn_input = torch.cat((rnn_first_part, code_vectors), dim=2)
        elif self.model_type == "DKT":
            rnn_input = rnn_first_part
        elif self.model_type == "P-Code-DKT":
            rnn_input = torch.cat((rnn_first_part, code_vectors,question_vectors), dim=2)
        elif self.model_type == "R-Code-DKT":
            rnn_input = torch.cat((rnn_first_part, code_vectors,question_vectors, reference_vectors), dim=2)
        elif self.model_type == "E-Code-DKT":
            rnn_input = torch.cat((rnn_first_part, code_vectors,question_vectors, reference_vectors,errorID_vectors), dim=2)
        
#         print(rnn_input.shape)
        out, hn = self.rnn(rnn_input)  # shape of out: [batch_size, length, hidden_size]
#         out, hn = self.rnn(x, h0)  # shape of out: [batch_size, length, hidden_size]
#         res = self.sig(self.fc(self.dropout(out)))  # shape of res: [batch_size, length, question]
        res = self.sig(self.fc(out))  # shape of res: [batch_size, length, question]
        # print(res)
        # print(res.shape)
        return res
