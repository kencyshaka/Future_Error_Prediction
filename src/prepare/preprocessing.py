import os
import sys
import pandas as pd
import numpy as np
import csv
import itertools
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', '..')  # Adjust the number of '..' as per your file structure
sys.path.append(src_dir)
from src.config import Config


def plot_test_train_distribution(percentages,filename):
    print("percentages", percentages)
    # Extract the data from the percentage list
    errors = [item[0] for item in percentages]
    train_percentages = [item[1] for item in percentages]
    test_percentages = [item[2] for item in percentages]

    # Set the width of each bar
    bar_width = 0.35
    # Set the positions of the bars on the x-axis
    x_train = [i for i in range(len(errors))]
    x_test = [i + bar_width for i in x_train]

    # Set the figure size
    plt.figure(figsize=(12, 6))
    # Plot the bars for training percentages
    plt.bar(x_train, train_percentages, width=bar_width, align='center', label='Training')
    plt.bar(x_test, test_percentages, width=bar_width, align='center', label='Test')
    plt.xticks(x_train, errors, rotation=90)
    plt.ylabel('Percentage')

    # Set the title of the plot
    plt.title('Error Distribution')
    plt.legend()
    plt.savefig(filename + '.png', dpi=300)

def plot_error_distribution(error_list, save_file_name):
    #all_labels = [label for label  in error_list]
    print("totol number of errors",len(error_list))

    # Count the occurrence of each error label and calculate the percentages
    label_counts = pd.Series(error_list).value_counts()
    percentage = label_counts / len(error_list) * 100

    # Clear the plot
    plt.clf()
    # Plot the bar graph
    plt.bar(range(len(percentage)), percentage.values)
    plt.ylabel('Percentage')
    plt.title('Occurrence of Error IDs')

    # Remove x-axis labels
    plt.xticks([])

    plt.tight_layout()
    # Save the plot as a PNG file
    plt.savefig(save_file_name + '.png', dpi=300)

def get_percentages(df_errors, y_train, y_test):
    percentages = []
    # get percentage distribution for each errors
    for error in df_errors['ErrorID']:  #
        print("error", error)
        train_error_count = y_train.count(int(error))
        test_error_count = y_test.count(int(error))

        print("count", train_error_count, test_error_count)

        frequency = df_errors[df_errors['ErrorID'] == error]['Frequency'].values[0]
        train_error_percentage = train_error_count / frequency
        test_error_percentage = test_error_count / frequency

        percentages.append([error, train_error_percentage, test_error_percentage, frequency])

    return percentages
def get_error_count(list_error, count_num,filename):
    '''Count the frequency of errors'''
    # Flatten the nested list
    flat_list = [item for sublist in list_error for item in sublist]
    counts = Counter(flat_list)
    sorted_numbers = sorted(counts.items(), key=lambda x: int(x[0]), reverse=False)

    # Print the counts
    standard_errors = []
    index = 0
    for number, count in sorted_numbers:
        if (count > count_num):
            standard_errors.append([index, number, count])
            index = index + 1
            print(f"{number}: {count}")
    print(filename)
    with open(filename+".csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Index', 'ErrorID', 'Frequency'])
        writer.writerows(standard_errors)

    return standard_errors


def get_indices_to_remove(df, error_ids):
    ''' filter the df to hold only records where the list in the unique_values are in the the df_errors
    '''
    filtered_rows = []
    indices_to_remove = []
    for i, row in enumerate(df.itertuples()):
        unique_values = [value for value in row.unique_values]  # Convert strings to integers
        if any(values not in error_ids for values in unique_values):
            filtered_rows.append(row)
            indices_to_remove.append(row.Index)

    return filtered_rows, indices_to_remove



def get_filtered_df(config, save_filepath):
    main_df = pd.read_csv(os.path.join(current_dir, "../../data/prepared/errors/error_MainTable.csv"))
    main_df = main_df[main_df["assignment_ID"] == config.assignment]
    main_df['error_labels'] = main_df['error_ID'].str.split('_')
    main_df['unique_values'] = main_df['error_labels'].apply(lambda x: sorted(set(x)))
    error_list = main_df['error_labels'].tolist()
    occured_errors = get_error_count(error_list,config.frequency,save_filepath+"/occured_errors_"+str(config.frequency))

    df_occured_errors = pd.DataFrame(occured_errors, columns=['Index', 'ErrorID','Frequency'])

    # Filter the DataFrame based on frequency of error occurance
    error_ids = df_occured_errors['ErrorID'].tolist()

    filtered_rows,rows_to_remove = get_indices_to_remove(main_df,error_ids)
    filtered_df = main_df.drop(rows_to_remove)

    #After filtering get the updated count of the errors
    updated_error_list = filtered_df['unique_values'].tolist()
    updated_occured_errors = get_error_count(updated_error_list,0, save_filepath+"/occured_errors_updated_"+str(config.frequency))

    df_errors = pd.DataFrame(updated_occured_errors, columns=['Index', 'ErrorID','Frequency'])

    # Create a dictionary mapping error IDs to their indices and save as the df
    error_indices = {error[1]: idx for idx, error in enumerate(updated_occured_errors)}
    print(error_indices)
    # Reshape the dictionary to list of tuples (key, value)
    error_indices = [(k, v) for k, v in error_indices.items()]

    # Create the DataFrame from the resized dictionary
    error_indices_df = pd.DataFrame(error_indices, columns=['key', 'value'])
    error_indices_df.to_csv(os.path.join(save_filepath, "error_indices_"+str(config.frequency)+".csv"), index=False)

    return main_df,filtered_df, error_indices_df,df_errors




def create_data_folds(config,save_filepath):
    main_df,filtered_df, error_indices_df, df_errors = get_filtered_df(config,save_filepath)
    grouped_df = filtered_df.groupby('subject_ID')['unique_values'].apply(list).reset_index()

    error_indices = error_indices_df.set_index('key')['value'].to_dict()
    # Extract unique values from each row's nested list
    grouped_df['unique_errors'] = grouped_df['unique_values'].apply(lambda x: list(set([num for sublist in x for num in sublist])))
    grouped_df['unique_errors_values'] = grouped_df['unique_errors'].apply(lambda x: [error_indices.get(str(val), None) for val in x])

    # Convert the 'unique_errors' column into an array of 1s and 0s
    grouped_df['error_array'] = grouped_df['unique_errors_values'].apply(lambda values: np.array([1 if i in values else 0 for i in range(len(error_indices))]))

    # get the student as features and all commited errors as labels
    features = [num for num in range(grouped_df.shape[0])]
    labels = np.array(grouped_df['error_array'].tolist())

    print(labels[0:10])
    print(len(features),labels.shape)

    # folders for storing the data features
    if not os.path.isdir(os.path.join(current_dir, "../../data/prepared/DKTFeatures_"+str(config.assignment)+"_"+ str(config.frequency))):
        os.mkdir(os.path.join(current_dir, "../../data/prepared/DKTFeatures_"+str(config.assignment)+"_"+ str(config.frequency)))

    # create 10 folds of test and trainging set
    msss = MultilabelStratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=42)
    # Split the training into validation and Training set
    msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)

    split_num=0
    print("features",features)
    print("labels",labels)
    for train_index, test_index in msss.split(features, labels):
        # print("TRAIN:", train_index.tolist(), "TEST:", test_index)
        print("TRAIN:", len(train_index), "TEST:", len(test_index))
        train, test = grouped_df.drop(test_index.tolist()), grouped_df.drop(train_index.tolist())

        test_error_list = [int(item) for sublist1 in test['unique_values'] for sublist2 in sublist1 for item in sublist2]
        train_error_list = [int(item) for sublist1 in train['unique_values'] for sublist2 in sublist1 for item in sublist2]

        #get the percentage distirbution
        percentages = get_percentages(df_errors, train_error_list, test_error_list)
        plot_test_train_distribution(percentages, os.path.join(save_filepath,'test_train_distribution_msss_'+ str(config.frequency) +'_'+ str(split_num)))
        plot_error_distribution(train_error_list,os.path.join(save_filepath,'train_error_distribution_msss_' + str(config.frequency) +'_'+ str(split_num)))
        plot_error_distribution(test_error_list,os.path.join(save_filepath,'test_error_distribution_msss_' + str(config.frequency) +'_'+ str(split_num)))

        d = {}
        problems = sorted(pd.unique(filtered_df["problem_ID"]))
        problems_d = {k: v for (v, k) in enumerate(problems)}

        np.save(os.path.join(current_dir,"../../data/prepared/DKTFeatures_" + str(config.assignment) +"_"+ str(config.frequency) + "/training_students.npy"),train['subject_ID'].tolist())
        np.save(os.path.join(current_dir,"../../data/prepared/DKTFeatures_" + str(config.assignment) +"_"+ str(config.frequency) + "/testing_students.npy"),test['subject_ID'].tolist())

        file_test = open(os.path.join(current_dir, "../../data/prepared/DKTFeatures_" + str(config.assignment) +"_"+ str(config.frequency) + "/test_firstatt_" + str(split_num) + ".csv"), "w")

        for s in test['subject_ID']:

            df = filtered_df[filtered_df["subject_ID"] == s]

            if len(df) > 0:
                file_test.write(str(len(df))) # number of attempts
                file_test.write(",\n")
                file_test.write(",".join(list(df["codestate_ID"]))) #code solutions id
                file_test.write(",\n")
                file_test.write(",".join([str(problems_d[i]) for i in df["problem_ID"]])) #attempted questions
                file_test.write(",\n")
                file_test.write(",".join(list((df["isError"] == 0).astype(int).astype(str)))) #result
                file_test.write(",\n")
                file_test.write(",".join(list(df["error_ID"]))) #commited errors IDs
                file_test.write(",\n")


        print("train_index",train_index)
        train.reset_index(drop=True, inplace=True)
        train['error_array'] = train['error_array'].apply(lambda x: np.array(x))
        array_list = np.array(train['error_array'].tolist())
        print("train['error_array']",array_list)

        for train_sub_index, test_sub_index in msss_val.split(train['subject_ID'], array_list):
            print("TRAIN:", len(train_sub_index), "VAL:", len(test_sub_index))
            print("TRAIN:", train_sub_index, "VAL:", test_sub_index)
            train_sub, val = train.drop(test_sub_index.tolist()), train.drop(train_sub_index.tolist())



            file_train = open(os.path.join(current_dir,
                                          "../../data/prepared/DKTFeatures_" + str(config.assignment) + "_"
                                           + str(config.frequency) + "/train_firstatt_" + str(split_num) + ".csv"), "w")

            for s in train_sub['subject_ID']:
                df = filtered_df[filtered_df["subject_ID"] == s]

                if len(df) > 0:
                    file_train.write(str(len(df)))  # number of attempts
                    file_train.write(",\n")
                    file_train.write(",".join(list(df["codestate_ID"])))  # code solutions id
                    file_train.write(",\n")
                    file_train.write(",".join([str(problems_d[i]) for i in df["problem_ID"]]))  # attempted questions
                    file_train.write(",\n")
                    file_train.write(",".join(list((df["isError"] == 0).astype(int).astype(str))))  # result
                    file_train.write(",\n")
                    file_train.write(",".join(list(df["error_ID"])))  # commited errors IDs
                    file_train.write(",\n")

            file_val =  open(os.path.join(current_dir,
                                          "../../data/prepared/DKTFeatures_" + str(config.assignment) + "_"
                                           + str(config.frequency) + "/val_firstatt_" + str(split_num) + ".csv"), "w")

            for s in val['subject_ID']:
                df = filtered_df[filtered_df["subject_ID"] == s]

                if len(df) > 0:
                    file_val.write(str(len(df)))  # number of attempts
                    file_val.write(",\n")
                    file_val.write(",".join(list(df["codestate_ID"])))  # code solutions id
                    file_val.write(",\n")
                    file_val.write(",".join([str(problems_d[i]) for i in df["problem_ID"]]))  # attempted questions
                    file_val.write(",\n")
                    file_val.write(",".join(list((df["isError"] == 0).astype(int).astype(str))))  # result
                    file_val.write(",\n")
                    file_val.write(",".join(list(df["error_ID"])))  # commited errors IDs
                    file_val.write(",\n")

        split_num = split_num + 1


if __name__ == '__main__':
    config = Config()
    if not os.path.isdir(os.path.join(current_dir, "../../data/prepared/errors/" + str(config.assignment))):
        os.mkdir(os.path.join(current_dir, "../../data/prepared/errors/" + str(config.assignment)))

    save_filepath = os.path.join(current_dir, "../../data/prepared/errors/"+ str(config.assignment))

    create_data_folds(config,save_filepath)