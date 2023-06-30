import os
import random
import wandb
import torch
import sys
import torch.optim as optim
import numpy as np
import pandas as pd
from dataloader import get_data_loader
import loss_function
import warnings
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', '..')  # Adjust the number of '..' as per your file structure
sys.path.append(src_dir)
warnings.filterwarnings("ignore")

from c2vRNNModel import c2vRNNModel
from evaluation import logging_evaluation_metrics

wandb.login()


def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    wandb.init(project="Sweep Future Error prediction")
    config = wandb.config

    setup_seed(0)

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    save_model = os.path.join(current_dir, '../../model/')
    if not os.path.isdir(save_model):
        os.mkdir(save_model)

    save_result = os.path.join(current_dir, '../../result/')
    if not os.path.isdir(save_result):
        os.mkdir(save_result)

    df_errors = pd.read_csv(os.path.join(current_dir, '../../data/prepared/errors/' + str(
        config.assignment) + '/occured_errors_updated_' + str(config.frequency) + '.csv'))
    ErrorID_LEN = len(df_errors)

    performance_list = []
    scores_list = []
    first_scores_list = []
    first_total_scores_list = []
    confusion_matrix_list = []

    for fold in range(2):
        print("----", fold, "-th run----")

        train_loader, test_loader = get_data_loader(config, config.questions, config.length, fold, ErrorID_LEN)
        if config.assignment == 487:
            node_count, path_count = np.load(os.path.join(current_dir, "../../data/prepared/DKTFeatures_"
                                                          + str(config.assignment) + "_" + str(
                config.frequency) + "/np_counts_"
                                                          + str(config.assignment) + "_" + str(fold) + ".npy"))
        else:
            node_count, path_count = np.load(os.path.join(current_dir, "../../data/prepared/DKTFeatures_"
                                                          + str(config.assignment) + "_" + str(
                config.frequency) + "/np_counts.npy"))

        model = c2vRNNModel(config, config.model_type, config.questions * 2,
                            config.hidden,
                            config.layers,
                            ErrorID_LEN,
                            node_count, path_count, device)

        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        loss_func = loss_function.lossFunc(config.questions, config.length, device, config.Prediction, config.loss_type)

        # training
        model.train()
        for epoch in range(config.epochs):
            print('epoch: ' + str(epoch))
            model, optimizer = loss_function.train_epoch(model, train_loader, optimizer,
                                                         loss_func, config, device, epoch)

        # Save the trained model:
        torch.save(model.state_dict(), os.path.join(save_model, 'model_' + str(fold) + '.pth'))

        # Inference time
        model.eval()
        first_total_scores, first_scores, scores, performance, confusion_matrix = loss_function.test_epoch(
            model, test_loader, loss_func, device, config, fold)

        first_total_scores_list.append(first_total_scores)
        scores_list.append(scores)
        first_scores_list.append(first_scores)
        performance_list.append(performance)
        confusion_matrix_list.append(confusion_matrix)

    logging_evaluation_metrics(first_total_scores_list, scores_list, first_scores_list, performance_list,
                               confusion_matrix_list, config, ErrorID_LEN)


if __name__ == '__main__':
    sweep_config = {
        'method': 'bayes',  # grid, random, bayes
        'metric': {
            'name': 'Overall_F1_score',
            'goal': 'maximize'
        },
        'parameters': {
            'epochs': {
                'values': [5, 10, 15, 40, 60]
            },
            'bs': {
                'values': [32, 64, 128]
            },
            'lr': {
               # 'distribution': 'uniform',
               # 'min': 0.000001,
               # 'max': 0.01
                'values': [ 0.009,0.0001,0.00005,0.00001]
            },
            'hidden': {
                'values': [64, 128, 260, 512]
            },
            'layers': {
                'values': [1, 4, 6, 8]
            },
            'loss_type': {
                'values': ["BCE"]
            },
            'length': {
                'values': [50]
            },
            'questions': {
                'values': [10]
            },
            'assignment': {
                'values': [439]
            },
            'frequency': {
                'values': [10, 50, 100]
            },
            'code_path_length': {
                'values': [8]
            },
            'code_path_width': {
                'values': [2]
            },
            'MAX_CODE_LEN': {
                'values': [100]
            },
            'Reference_LEN': {
                'values': [200]
            },
            'embedds_type': {
                'values': ["p10"]
            },
            'Prediction': {
                'values': ["ErrorIDs"]
            },
            'ErrorID_LEN': {
                'values': [85]
            },
            'model_type': {
                'values': ['E-Code-DKT', 'R-Code-DKT', 'P-Code-DKT', 'DKT','Code-DKT']
            },
            'MAX_QUESTION_LEN_partI': {
                'values': [0, 768]
            },
            'MAX_QUESTION_LEN_partII': {
                'values': [0, 128]
            },
        }
    }

    # 3: Start the sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="Sweep Future Error prediction"
    )
    print("the sweep_id is", sweep_id)

    wandb.agent(sweep_id, function=main, count=200)






