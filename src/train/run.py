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
from src.config import Config
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
    config = Config()

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

    run = wandb.init(
        # Set the project where this run will be logged
        project="Future Error prediction",
        # Track hyperparameters and run metadata
        config={
            "mode": 'Train',
            "learning_rate": config.lr,
            "epochs": config.epochs,
            "attempts": config.length,
            "batch": config.bs,
            "questions": config.questions,
            "assignment": config.assignment,
            "device": device,
            "model_type": config.model_type,
            "code": config.MAX_CODE_LEN * 3,
            "question": config.MAX_QUESTION_LEN_partI + config.MAX_QUESTION_LEN_partII,
            "q_embedds_type": config.embedds_type,
            "error_vector": ErrorID_LEN,
            "prediction": config.Prediction,
            "loss_type": config.loss_type,
            "frequency": config.frequency,
        })

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
                                                          + str(config.assignment) + "_"+str(config.frequency) +"/np_counts_"
                                                          + str(config.assignment) + "_" + str(fold) + ".npy"))
        else:
            node_count, path_count = np.load(os.path.join(current_dir, "../../data/prepared/DKTFeatures_"
                                                          + str(config.assignment) + "_"+str(config.frequency) + "/np_counts.npy"))

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
        torch.save(model.state_dict(), os.path.join(save_model, 'model_'+str(fold)+'.pth'))

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
                               confusion_matrix_list, config,ErrorID_LEN)


if __name__ == '__main__':
    main()
