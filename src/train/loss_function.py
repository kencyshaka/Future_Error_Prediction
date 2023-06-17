import tqdm
import torch
import logging
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import torch.nn as nn
from sklearn import metrics
from evaluation import performance_granular
import wandb


def plot_heatmap(batch, pred, fold, batch_n, config):
    
    # TODO: No hardcoding problem dict but what about other assignments?
    problem_dict = {"000000010":"1",
                    "000000001":"3",
                    "000010000":"5",
                    "010000000":"13",
                    "001000000":"232",
                    "000100000":"233",
                    "100000000":"234",
                    "000001000":"235",
                    "000000100":"236"
                   }
    problems = []
    for s in range(pred.shape[0]):
        
        delta = (batch[s][1:, 0:config.questions] + batch[s][1:, config.questions:config.questions*2]).detach().cpu().numpy()
        
        a = (((batch[s][:, 0:config.questions] - batch[s][:, config.questions:config.questions*2]) + 1) // 2)[1:].detach().cpu().numpy()
        p = pred[s].detach().cpu().numpy()

        for i in range(len(delta)):
            if np.sum(delta, axis=1)[i] > 0:
                p = p[i:]
                a = a[i:]
                delta = delta[i:]
                break
        
        problems = [problem_dict["".join([str(int(i)) for i in sk])] for sk in delta]
        
        plt.figure(figsize=(15, 6), dpi=80)
    
        ax = sns.heatmap(p.T, annot=a.T, linewidth=0.5, vmin=0, vmax=1, cmap="Blues")

        plt.xticks(np.arange(len(problems))+0.5, problems, rotation=45)
        plt.yticks(np.arange(config.questions)+0.5, ['234', '13', '232', '233', '5', '235', '236', '1', '3'], rotation=45)
        plt.xlabel("Attempting Problem")
        plt.ylabel("Problem")

        
        plt.title("Heatmap for student "+str(s)+" fold "+str(fold))
        plt.tight_layout()
        plt.savefig("heatmaps/b"+str(batch_n)+"_s"+str(s)+"_f"+str(fold)+".png")
        


class lossFunc(nn.Module):
    def __init__(self, num_of_questions, max_step, device, prediction_type,loss_type,):
        super(lossFunc, self).__init__()
        self.crossEntropy = nn.BCELoss()
        self.num_of_questions = num_of_questions
        self.max_step = max_step
        self.device = device
        self.prediction = prediction_type
        self.loss_type = loss_type


    def forward(self, pred, batch):
        loss = 0
        prediction = torch.tensor([])
        ground_truth = torch.tensor([])
        pred = pred.to('cpu')


        for student in range(pred.shape[0]):

            #Find rows in ground truth that contain at least a 1 in any of their column i.e. student attmpted a question

            if self.prediction == "ErrorIDs":
                selected_p, selected_gt = self.get_attempted_question_ErrorID(pred,batch,student)
            else:
                selected_p, selected_gt = self.get_attempted_question_performance(pred,batch,student)


            if self.loss_type == "BCE":
                loss += self.crossEntropy(selected_p, selected_gt)
            else:
                loss += self.crossEntropy(selected_p, selected_gt) # implement a different loss function



            prediction = torch.cat([prediction, selected_p])
            ground_truth = torch.cat([ground_truth, selected_gt])

        return loss, prediction, ground_truth

    def get_attempted_question_ErrorID(self,pred,batch,student):
        gt = batch[student][1:, ]
        rows_with_ones = torch.any(gt == 1, dim=1)

        # Select the rows of a that contain at least a 1
        a = gt[rows_with_ones]

        # Select the corresponding rows of p
        p = pred[student][rows_with_ones]

        return p,a


    def get_attempted_question_performance(self,pred,batch,student):
        delta = batch[student][:, 0:self.num_of_questions] + batch[student][:,
                                                             self.num_of_questions:self.num_of_questions * 2]  # shape: [length, questions]
        x = pred[student][:self.max_step - 1]
        y = delta[1:].t()
        temp = x.mm(y)
        index = torch.tensor([[i for i in range(self.max_step - 1)]],
                             dtype=torch.long)
        p = temp.gather(0, index)[0]  # which prediction, the next q+1 or all of the questions?
        a = (((batch[student][:, 0:self.num_of_questions] -
               batch[student][:, self.num_of_questions:self.num_of_questions * 2]).sum(1) + 1) //
             2)[1:]

        for i in range(len(p)):
            if p[i] > 0:
                p = p[i:]
                a = a[i:]
                break

        return p,a
    

def train_epoch(model, trainLoader, optimizer, loss_func, config, device,epoch):
    model.to(device)
    for batch in tqdm.tqdm(trainLoader, desc='Training:    ', mininterval=2):
        batch_new = batch[:,:-1,:].to(device)    # pass except the last one.
        pred = model(batch_new)                  # trains the model and get the results [batch,49,85 (size of errorIDs)]
        starting_point = config.questions*2 + config.MAX_CODE_LEN*3 + config.MAX_QUESTION_LEN_partI + config.MAX_QUESTION_LEN_partII + config.Reference_LEN
        print("the shape of the batch is", batch[:,:,starting_point:].shape)
        print("the shape of the model output",pred.shape)
        loss, prediction, ground_truth = loss_func(pred, batch[:,:,starting_point:])


        print("the prediciton is ", prediction.shape)
        print("the ground_truth is ", ground_truth.shape)

        # log the wandb loss value here for training per epoch
        wandb.log({"loss": loss, "epoch": epoch})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    wandb.log({"Avg_loss": loss, "epoch": epoch})

    return model, optimizer


def test_epoch(model, testLoader, loss_func, device, config, fold):
    model.to(device)
    ground_truth = torch.tensor([])
    prediction = torch.tensor([])
    full_data = torch.tensor([])
    preds = torch.tensor([])
    batch_n = 0
    for batch in tqdm.tqdm(testLoader, desc='Testing:     ', mininterval=2):
        batch_new = batch[:,:-1,:].to(device)
        pred = model(batch_new)    #should i not turn off with grad is 0?
        starting_point = config.questions * 2 + config.MAX_CODE_LEN * 3 + config.MAX_QUESTION_LEN_partI + config.MAX_QUESTION_LEN_partII + config.Reference_LEN
        loss, p, a = loss_func(pred, batch[:, :, starting_point:])
        
        prediction = torch.cat([prediction, p])
        ground_truth = torch.cat([ground_truth, a])
        full_data = torch.cat([full_data, batch])    #includes all 50
        preds = torch.cat([preds, pred.cpu()])
        # plot_heatmap(batch, pred, fold, batch_n)
        batch_n += 1

    return performance_granular(full_data, preds, ground_truth, prediction, config)


