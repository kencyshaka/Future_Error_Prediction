import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', '..')  # Adjust the number of '..' as per your file structure
sys.path.append(src_dir)

from src.config import Config

config = Config()

main_df = pd.read_csv(os.path.join(current_dir, "../../data/prepared/errors/error_MainTable.csv"))

main_df = main_df[main_df["assignment_ID"] == config.assignment]

students = pd.unique(main_df["subject_ID"])

problems = sorted(pd.unique(main_df["problem_ID"]))

problems_d = {k:v for (v,k) in enumerate(problems) }

d = {}
for s in students:
    d[s] = {}
    df = main_df[main_df["subject_ID"] == s]
    d[s]["length"] = len(df)
    d[s]["Problems"] = [str(problems_d[i]) for i in df["problem_ID"]]
    d[s]["Result"] = list((df["isError"] == 0).astype(int).astype(str))
    d[s]["CodeStates"] = list(df["codestate_ID"])
    d[s]["ErrorIDs"] = list(df["error_ID"])

    
train_val_s, test_s = train_test_split(students, test_size=0.2, random_state=1)

if not os.path.isdir(os.path.join(current_dir, "../../data/prepared/DKTFeatures_"+str(config.assignment))):
    os.mkdir(os.path.join(current_dir, "../../data/prepared/DKTFeatures_"+str(config.assignment)))

np.save(os.path.join(current_dir, "../../data/prepared/DKTFeatures_"+str(config.assignment)+"/training_students.npy"), train_val_s)
np.save(os.path.join(current_dir, "../../data/prepared/DKTFeatures_"+str(config.assignment)+"/testing_students.npy"), test_s)

file_test = open(os.path.join(current_dir, "../../data/prepared/DKTFeatures_"+str(config.assignment)+"/test_data.csv"),"w")

for s in test_s:
    if d[s]['length']>0:
        file_test.write(str(d[s]['length']))
        file_test.write(",\n")
        file_test.write(",".join(d[s]['CodeStates']))
        file_test.write(",\n")
        file_test.write(",".join(d[s]['Problems']))
        file_test.write(",\n")
        file_test.write(",".join(d[s]['Result']))
        file_test.write(",\n")
        file_test.write(",".join(d[s]['ErrorIDs']))
        file_test.write(",\n")
        
for fold in range(10):
    train_s, val_s = train_test_split(train_val_s, test_size=0.25, random_state=fold)

    file_train = open(os.path.join(current_dir, "../../data/prepared/DKTFeatures_"+str(config.assignment)+"/train_firstatt_"+str(fold)+".csv"),"w")
    for s in train_s:
        if d[s]['length']>0:
            file_train.write(str(d[s]['length']))
            file_train.write(",\n")
            file_train.write(",".join(d[s]['CodeStates']))
            file_train.write(",\n")
            file_train.write(",".join(d[s]['Problems']))
            file_train.write(",\n")
            file_train.write(",".join(d[s]['Result']))
            file_train.write(",\n")
            file_train.write(",".join(d[s]['ErrorIDs']))
            file_train.write(",\n")

    file_val = open(os.path.join(current_dir, "../../data/prepared/DKTFeatures_"+str(config.assignment)+"/val_firstatt_"+str(fold)+".csv"),"w")
    for s in val_s:
        if d[s]['length']>0:
            file_val.write(str(d[s]['length']))
            file_val.write(",\n")
            file_val.write(",".join(d[s]['CodeStates']))
            file_val.write(",\n")
            file_val.write(",".join(d[s]['Problems']))
            file_val.write(",\n")
            file_val.write(",".join(d[s]['Result']))
            file_val.write(",\n")
            file_val.write(",".join(d[s]['ErrorIDs']))
            file_val.write(",\n")