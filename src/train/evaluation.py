import torch
import os
import numpy as np
import random
import wandb
import loss_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.config import Config
from c2vRNNModel import c2vRNNModel
from dataloader import get_test_loader
from sklearn.metrics import hamming_loss, accuracy_score, precision_score, recall_score, f1_score, \
    multilabel_confusion_matrix, roc_auc_score
from matplotlib.backends.backend_pdf import PdfPages


def main():
    config = Config()

    def setup_seed(seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    run = wandb.init(
        # Set the project where this run will be logged
        project="TRIAL-DKT",
        # Track hyperparameters and run metadata
        config={
            "mode": 'Eval',
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
            "error_vector": config.ErrorID_LEN,
            "prediction": config.Prediction,
            "loss_type": config.loss_type,
        })

    performance_list = []
    scores_list = []
    first_scores_list = []
    first_total_scores_list = []
    confusion_matrix_list = []

    for fold in range(2):
        print("----", fold, "-th run----")
        test_loader = get_test_loader(config, config.questions, config.length, fold)
        # load the model
        if config.assignment == 487:
            node_count, path_count = np.load(
                "../../data/prepared/DKTFeatures_" + str(config.assignment) + "/np_counts_" + str(
                    config.assignment) + "_" + str(
                    fold) + ".npy")
        else:
            node_count, path_count = np.load(
                "../../data/prepared/DKTFeatures_" + str(config.assignment) + "/np_counts.npy")

        model = c2vRNNModel(config, config.model_type, config.questions * 2,
                            config.hidden,
                            config.layers,
                            config.ErrorID_LEN,
                            node_count, path_count, device)

        # Load the saved model weights
        model.load_state_dict(torch.load('../../model/model_' + str(fold) + '.pth'))

        # Set the model in evaluation mode
        model.eval()

        # perform inference loop

        loss_func = loss_function.lossFunc(config.questions, config.length, device, config.Prediction, config.loss_type)

        # Inference time
        first_total_scores, first_scores, scores, performance, confusion_matrix = loss_function.test_epoch(
            model, test_loader, loss_func, device, config, fold)
        first_total_scores_list.append(first_total_scores)
        scores_list.append(scores)
        first_scores_list.append(first_scores)
        performance_list.append(performance)
        confusion_matrix_list.append(confusion_matrix)

    logging_evaluation_metrics(first_total_scores_list, scores_list, first_scores_list, performance_list,
                               confusion_matrix_list, config)


def performance_granular(batch, pred, ground_truth, prediction, config):
    scores = {}
    first_scores = {}

    preds, gts, first_preds, first_gts = get_attempted_questions(pred, batch, config)

    ## calculate per question
    first_total_gts = []
    first_total_preds = []

    for j in range(config.questions):
        metrics_values = evaluate_multilabel_classification(gts[j], np.array(preds[j]))

        first_metrics_values = evaluate_multilabel_classification(first_gts[j],
                                                                  np.array(first_preds[j]))

        scores[j] = [metrics_values['Hamming_Loss'], metrics_values['F1_Score_Weighted'],
                     metrics_values['Precision_Weighted'], metrics_values['Recall_Weighted'],
                     metrics_values['Accuracy']]

        first_scores[j] = [first_metrics_values['Hamming_Loss'], first_metrics_values['F1_Score_Weighted'],
                           first_metrics_values['Precision_Weighted'], first_metrics_values['Recall_Weighted'],
                           first_metrics_values['Accuracy']]

        print('problem ' + str(j) + ' hamming: ' + str(metrics_values['Hamming_Loss']) +
              ' f1: ' + str(metrics_values['F1_Score_Weighted']) +
              ' precision: ' + str(metrics_values['Precision_Weighted']) +
              ' recall: ' + str(metrics_values['Recall_Weighted']) +
              ' f1_macro: ' + str(metrics_values['F1_Score_Macro']) +
              ' precision_macro: ' + str(metrics_values['Precision_Macro']) +
              ' recall_macro: ' + str(metrics_values['Recall_Macro']) +
              ' sub_acc: ' + str(metrics_values['Accuracy']) +
              ' f1_micro: ' + str(metrics_values['F1_Score_Micro']))

        print('First prob ' + str(j) + ' hamming: ' + str(first_metrics_values['Hamming_Loss']) +
              ' f1: ' + str(first_metrics_values['F1_Score_Weighted']) +
              ' precision: ' + str(first_metrics_values['Precision_Weighted']) +
              ' recall: ' + str(first_metrics_values['Recall_Weighted']) +
              ' f1_macro: ' + str(first_metrics_values['F1_Score_Macro']) +
              ' precision_macro: ' + str(first_metrics_values['Precision_Macro']) +
              ' recall_macro: ' + str(first_metrics_values['Recall_Macro']) +
              ' sub_acc: ' + str(first_metrics_values['Accuracy']) +
              ' f1_micro: ' + str(first_metrics_values['F1_Score_Micro']))

        first_total_gts.extend(first_gts[j])
        first_total_preds.extend(np.array(first_preds[j]))

        # metrics_values = evaluate_multilabel_classification(gts[j], np.array(preds[j]))
        #
        # first_metrics_values = evaluate_multilabel_classification(first_gts[j], np.array(first_preds[j]))
        #
        # scores[j] = [metrics_values['auc'], metrics_values['F1_Score_Macro'],
        #              metrics_values['Recall_Macro'], metrics_values['Precision_Macro'],
        #              metrics_values['Accuracy']
        #              ]
        #
        # first_scores[j] = [first_metrics_values['auc'], first_metrics_values['F1_Score_Macro'],
        #                    first_metrics_values['Recall_Macro'], first_metrics_values['Precision_Macro'],
        #                    first_metrics_values['Accuracy']]
        #
        # print('problem ' + str(j) + ' auc: ' + str(metrics_values['auc']) +
        #       ' f1: : ' + str(metrics_values['F1_Score_Macro']) +
        #       ' recall: ' + str(metrics_values['Recall_Macro']) +
        #       ' precision: ' + str(metrics_values['Precision_Macro']) +
        #       ' acc: ' + str(metrics_values['Accuracy']))
        #
        # print(' first prediction problem ' + str(j) + ' auc: ' + str(first_metrics_values['auc']) +
        #       ' f1: : ' + str(first_metrics_values['F1_Score_Macro']) +
        #       ' recall: ' + str(first_metrics_values['Recall_Macro']) +
        #       ' precision: ' + str(first_metrics_values['Precision_Macro']) +
        #       ' acc: ' + str(first_metrics_values['Accuracy']))
        #
        # first_total_gts.extend(first_gts[j])
        # first_total_preds.extend(first_preds[j])

    ## calculate overall

    overall_metrics_values = evaluate_multilabel_classification(ground_truth.detach().numpy(),
                                                                prediction.detach().numpy())

    overall_first_metrics_values = evaluate_multilabel_classification(np.array(first_total_gts),
                                                                      np.array(first_total_preds))

    first_total_scores = [overall_first_metrics_values['Hamming_Loss'],
                          overall_first_metrics_values['F1_Score_Weighted'],
                          overall_first_metrics_values['Precision_Weighted'],
                          overall_first_metrics_values['Recall_Weighted'],
                          overall_first_metrics_values['Accuracy']]

    overall_total_scores = [overall_metrics_values['Hamming_Loss'], overall_metrics_values['F1_Score_Weighted'],
                            overall_metrics_values['Precision_Weighted'], overall_metrics_values['Recall_Weighted'],
                            overall_metrics_values['Accuracy']]

    print(' hamming: ' + str(overall_metrics_values['Hamming_Loss']) +
          ' f1: ' + str(overall_metrics_values['F1_Score_Weighted']) +
          ' precision: ' + str(overall_metrics_values['Precision_Weighted']) +
          ' recall: ' + str(overall_metrics_values['Recall_Weighted']) +
          ' f1_macro: ' + str(overall_metrics_values['F1_Score_Macro']) +
          ' precision_macro: ' + str(overall_metrics_values['Precision_Macro']) +
          ' recall_macro: ' + str(overall_metrics_values['Recall_Macro']) +
          ' sub_acc: ' + str(overall_metrics_values['Accuracy']) +
          ' f1_micro: ' + str(overall_metrics_values['F1_Score_Micro']))

    return first_total_scores, first_scores, scores, overall_total_scores, overall_metrics_values[
        'confusion_matrix']

    # overall_metrics_values = evaluate_multilabel_classification(ground_truth.detach().numpy(),
    #                                                             prediction.detach().numpy())
    #
    # overall_first_metrics_values = evaluate_multilabel_classification(first_total_gts, first_total_preds)
    #
    # first_total_scores = [overall_first_metrics_values['auc'],
    #                       overall_first_metrics_values['F1_Score_Macro'],
    #                       overall_first_metrics_values['Recall_Macro'],
    #                       overall_first_metrics_values['Precision_Macro'],
    #                       overall_first_metrics_values['Accuracy']]
    #
    # overall_total_scores = [overall_metrics_values['auc'],
    #                         overall_metrics_values['F1_Score_Macro'],
    #                         overall_metrics_values['Recall_Macro'],
    #                         overall_metrics_values['Precision_Macro'],
    #                         overall_metrics_values['Accuracy']]
    #
    # print('overall auc: ' + str(overall_metrics_values['auc']) +
    #       ' f1: : ' + str(overall_metrics_values['F1_Score_Macro']) +
    #       ' recall: ' + str(overall_metrics_values['Recall_Macro']) +
    #       ' precision: ' + str(overall_metrics_values['Precision_Macro']) +
    #       ' acc: ' + str(overall_metrics_values['Accuracy']))
    #
    # return first_total_scores, first_scores, scores, overall_total_scores, overall_metrics_values[
    #     'confusion_matrix']


def evaluate_multilabel_classification(y_true, y_pred):
    y_pred_thresholded = (y_pred >= 0.5).astype(int)
    # Calculate subset accuracy
    accuracy = accuracy_score(y_true, y_pred_thresholded)

    # Calculate Hamming Loss
    hamming = hamming_loss(y_true, y_pred_thresholded)

    # Calculate micro-averaged precision, recall, and F1 score
    micro_precision = precision_score(y_true, y_pred_thresholded, average='micro', zero_division=1)
    micro_recall = recall_score(y_true, y_pred_thresholded, average='micro', zero_division=1)
    micro_f1 = f1_score(y_true, y_pred_thresholded, average='micro', zero_division=1)

    # Calculate weigthed-averaged precision, recall, and F1 score
    weighted_precision = precision_score(y_true, y_pred_thresholded, average='weighted', zero_division=1)
    weighted_recall = recall_score(y_true, y_pred_thresholded, average='weighted', zero_division=1)
    weighted_f1 = f1_score(y_true, y_pred_thresholded, average='weighted', zero_division=1)

    # Calculate macro-averaged precision, recall, and F1 score
    macro_precision = precision_score(y_true, y_pred_thresholded, average='macro', zero_division=1)
    macro_recall = recall_score(y_true, y_pred_thresholded, average='macro', zero_division=1)
    macro_f1 = f1_score(y_true, y_pred_thresholded, average='macro', zero_division=1)

    # calculate multilabel_confusion_matrix
    cm = multilabel_confusion_matrix(y_true, y_pred_thresholded)

    # Create a dictionary to store the results
    metrics = {
        'Hamming_Loss': hamming,
        'Precision_Micro': micro_precision,
        'Precision_Macro': macro_precision,
        'Precision_Weighted': weighted_precision,
        'Recall_Micro': micro_recall,
        'Recall_Macro': macro_recall,
        'Recall_Weighted': weighted_recall,
        'F1_Score_Micro': micro_f1,
        'F1_Score_Macro': macro_f1,
        'F1_Score_Weighted': weighted_f1,
        'Accuracy': accuracy,
        'confusion_matrix': cm,
    }

    return metrics


def get_attempted_questions(pred, batch, config):
    preds = {k: [] for k in range(config.questions)}
    gts = {k: [] for k in range(config.questions)}
    first_preds = {k: [] for k in range(config.questions)}
    first_gts = {k: [] for k in range(config.questions)}

    for s in range(pred.shape[0]):
        delta = (batch[s][1:, 0:config.questions] + batch[s][1:, config.questions:config.questions * 2])

        # Find rows in ground truth that contain at least a 1 in any of their column i.e. student attempted the question
        gt = batch[s][1:,
             config.questions * 2 + config.MAX_CODE_LEN * 3 + config.MAX_QUESTION_LEN_partII + config.MAX_QUESTION_LEN_partI + config.Reference_LEN:]
        rows_with_ones = torch.any(gt == 1, dim=1)

        # Select the rows of a that contain at least a 1
        selected_gt = gt[rows_with_ones].detach().cpu().numpy()

        # Select the corresponding rows of p
        selected_p = pred[s][rows_with_ones].detach().cpu().numpy()

        delta = delta.detach().cpu().numpy()[rows_with_ones]

        for i in range(len(selected_p)):
            for j in range(config.questions):
                if delta[i, j] == 1:  # student attempted this question and add their prediction and ground thruth
                    preds[j].append(selected_p[i])
                    gts[j].append(selected_gt[i])
                    if i == 0 or delta[
                        i - 1, j] != 1:  # for that attempt check if the previous is not 1, i.e its a first attempt
                        first_preds[j].append(selected_p[i])
                        first_gts[j].append(selected_gt[i])
                    break

    return preds, gts, first_preds, first_gts


def get_attempted_questions_performance(pred, batch, config):
    preds = {k: [] for k in range(config.questions)}
    gts = {k: [] for k in range(config.questions)}
    first_preds = {k: [] for k in range(config.questions)}
    first_gts = {k: [] for k in range(config.questions)}

    for s in range(pred.shape[0]):
        delta = (batch[s][1:, 0:config.questions] + batch[s][1:, config.questions:config.questions * 2])
        temp = pred[s][:config.length - 1].mm(delta.T)
        index = torch.tensor([[i for i in range(config.length - 1)]],
                             dtype=torch.long)
        p = temp.gather(0, index)[0].detach().cpu().numpy()
        a = (((batch[s][:, 0:config.questions] - batch[s][:, config.questions:config.questions * 2]).sum(1) + 1) // 2)[
            1:].detach().cpu().numpy()

        for i in range(len(p)):
            if p[i] > 0:
                p = p[i:]
                a = a[i:]
                delta = delta.detach().cpu().numpy()[i:]
                break

        for i in range(len(p)):
            for j in range(config.questions):
                if delta[i, j] == 1:  # student attempted this question and add their prediction and ground thruth
                    preds[j].append(p[i])
                    gts[j].append(a[i])
                    if i == 0 or delta[
                        i - 1, j] != 1:  # for that attempt check if the previous is not 1, i.e its a first attempt
                        first_preds[j].append(p[i])
                        first_gts[j].append(a[i])
                    break

    return preds, gts, first_preds, first_gts


def logging_evaluation_metrics(first_total_scores_list, scores_list, first_scores_list, performance_list,
                               confusion_matrix_list, config):
    print("***************************************** done ******************************************")
    print("Average scores of the first attempts:   ", np.mean(first_total_scores_list, axis=0))
    print("Average scores of all overall attempts: ", np.mean(performance_list, axis=0))

    wandb.log(
            {"First_F1_score": np.mean(first_total_scores_list, axis=0)[1],
             "Overall_F1_score": np.mean(performance_list, axis=0)[1]})

    # calculating overall performance for all folds
    metrics = ['hamming', 'f1', 'precision', 'recall', 'accuracy']
    columns = ["category", "auc", "f1", "recall", "precision", "acc"]

    table_data = []
    data = np.mean(first_total_scores_list, axis=0)
    table_data.append(["first", data[0], data[1], data[2], data[3], data[4]])

    data = np.mean(performance_list, axis=0)
    table_data.append(["overall", data[0], data[1], data[2], data[3], data[4]])

    table = wandb.Table(data=table_data, columns=columns)
    wandb.log({"Overall performance": table})

    # calculating overall performance of the first attempt per problem for all folds
    avg_metrics = []
    for problem in range(len(first_scores_list[0])):
        avg_problem_metrics = []

        for metric_idx, metric in enumerate(metrics):
            metric_sum = 0

            for run_scores in first_scores_list:
                metric_sum += run_scores[problem][metric_idx]

            avg_problem_metrics.append(metric_sum / len(first_scores_list))

        avg_metrics.append(avg_problem_metrics)

    avg_metrics = np.array(avg_metrics)

    table_data = []
    for problem_id, metrics in enumerate(avg_metrics):
        problem_data = ["Problem" + str(problem_id + 1)] + metrics.tolist()
        table_data.append(problem_data)

    table = wandb.Table(data=table_data, columns=columns)
    wandb.log({"Average first attempt performance per problem": table})

    # calculating overall performance per problem for all folds
    avg_metrics = []
    for problem in range(len(scores_list[0])):
        avg_problem_metrics = []

        for metric_idx, metric in enumerate(metrics):
            metric_sum = 0

            for run_scores in scores_list:
                metric_sum += run_scores[problem][metric_idx]

            avg_problem_metrics.append(metric_sum / len(scores_list))

        avg_metrics.append(avg_problem_metrics)

    avg_metrics = np.array(avg_metrics)

    table_data = []
    for problem_id, metrics in enumerate(avg_metrics):
        problem_data = ["Problem" + str(problem_id + 1)] + metrics.tolist()
        table_data.append(problem_data)

    table = wandb.Table(data=table_data, columns=columns)
    wandb.log({"Average overall performance per problem": table})

    # plotting the confusion matrix

    # get the best confusion matrix i.e. the matrix with higher f1score_weighted
    performance_array = np.array(performance_list)
    max_index = np.argmax(performance_array[:, 1])
    confusion_matrices = confusion_matrix_list[max_index]
    # Define the labels for the first 10 errorIDs subset
    labels_first = ['ErrorID ' + str(i) for i in range(10)]
    plot_confusion_matrix(confusion_matrices, labels_first, "confusion_matrices_first.pdf")

    if config.ErrorID_LEN == 85:
        labels_last = ['ErrorID ' + str(i) for i in range(74, 84)]
        plot_confusion_matrix(confusion_matrices, labels_last, "confusion_matrices_last.pdf")


def plot_confusion_matrix(confusion_matrices, labels_first,filename):
    # Create subplots for the first subset
    fig, axes = plt.subplots(2, 5, figsize=(16, 8))
    axes = axes.ravel()

    # Create a directory to save the results
    save_dir = '../../result'
    os.makedirs(save_dir, exist_ok=True)

    # Create a PDF file to save the confusion matrices
    pdf_path = os.path.join(save_dir, filename)
    pdf_pages = PdfPages(pdf_path)

    # Plot and save the confusion matrices for the first subset
    for i in range(len(labels_first)):
        # Plot the confusion matrix
        axes[i].imshow(confusion_matrices[i], interpolation='nearest', cmap=plt.cm.Blues)
        axes[i].set_title('Confusion Matrix - ' + labels_first[i])
        axes[i].set_xticks([0, 1])
        axes[i].set_yticks([0, 1])
        axes[i].set_xticklabels(['Negative', 'Positive'])
        axes[i].set_yticklabels(['Negative', 'Positive'])
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')

        # Add text annotations to indicate correct/incorrect predictions
        for row in range(2):
            for col in range(2):
                # Determine the text color based on the cell value
                text_color = 'brown' if confusion_matrices[i][row, col] > confusion_matrices[i][col, col]/2 else 'black'

                # Add the text annotation to the cell
                axes[i].text(col, row, confusion_matrices[i][row, col], ha='center', va='center', color=text_color)

    # Display the confusion matrix
    plt.tight_layout()
    # plt.show()

    # Save the figure to the PDF file
    pdf_pages.savefig(fig)

    pdf_pages.close()
    plt.close()

    # Log the confusion matrix to Weights & Biases
    wandb.log({labels_first[i]: wandb.Image(fig)})

    # Clear the current figure
    plt.clf()


# create a main function and load the test case and perform evaluation
if __name__ == '__main__':
    main()
