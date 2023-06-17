import re
import os
import pandas as pd
from src.ProgSnap2 import ProgSnap2Dataset


def get_event_list(main_table):
    submissions = []
    grouped = main_table.groupby(['SubjectID', 'AssignmentID', 'ProblemID'])

    for _, group in grouped:
        errors_found = False
        submission_list = []
        first_correct = True
        for _, row in group.iterrows():

            if row['EventType'] == 'Compile':
                if row['Compile.Result'] == 'Error':
                    errors_found = True
                    first_correct = False
                    submission_list.append(row['EventID'])
                elif errors_found and not first_correct and row['Compile.Result'] == 'Success':
                    submission_list.append(row['EventID'])
                    first_correct = True
                elif not errors_found:
                    submission_list.append(row['EventID'])
                    break
        #         print(submissions)

        submissions.extend(submission_list)
    # print(submissions)
    new_df = main_table[main_table['EventID'].isin(submissions)][['SubjectID', 'AssignmentID', 'ProblemID', 'EventID']]
    event_list = new_df.groupby(['SubjectID', 'AssignmentID', 'ProblemID']).apply(
        lambda x: list(x.EventID)).reset_index(name='submission_event_list')

    print("event list shape before cleaning wrong pairs ---", event_list.shape)
    event_list = event_list[~event_list['submission_event_list'].astype(str).str.contains(
        '212703|463379')]  # delete rows that contain wrong solutions

    print("event list shape after cleaning ---", event_list.shape)
    print("event list---", event_list.head())

    if not os.path.isdir('../../data/prepared/errors'):
        os.mkdir('../../data/prepared/errors')

    event_list.to_csv('../../data/prepared/errors/submission_list.csv', index=False)

    return event_list


def get_error_list(main_table):
    compile_messages = main_table[(main_table['EventType'] == 'Compile.Error')]  # get the compiler message for each record with errors
    compile_messages = compile_messages[['ParentEventID', 'CompileMessageData']]
    # if isDataset == CF.F19:
    #     compile_messages['ParentEventID'] = compile_messages['ParentEventID'].astype(
    #         int)  # convert the float column to a string column

    # group the dataframe by ParentEventID and concatenate the CompileMessageData column as a list of integers
    error_list = compile_messages.reset_index().groupby('ParentEventID', as_index=False, sort=False).agg(
        {'CompileMessageData': lambda x: '@'.join(list(x))}).rename(
        columns={'CompileMessageData': 'submission_error_list'})

    print("error list shape ---", error_list.shape)
    print("error list---", error_list.head())

    error_list.to_csv('../../data/prepared/errors/submission_error_list.csv', index=False)

    return error_list


def get_line_number(error_message):
    pattern = r"line (\d+):"
    match = re.search(pattern, error_message)
    if match:
        return int(match.group(1))
    else:
        return None


def remove_last_char_if_dot(s):
    if s.endswith(' '):
        s = s[:-2]
    return s


def get_errorID(df_error, error):
    clean_error = replace_msg_identifiers(error)
    row = df_error[df_error['message'] == clean_error].iloc[0]
    errorID = row['ID']

    return errorID


def get_error_message(error):
    # some are just 3 and some just 2
    s = error.split(':')
    if (len(s) == 4) and any(word in s[3] for word in ["variable", "method", "class"]):
        last_two = ":".join(s[-2:])  # joins the last two parts using ":"
        msg = last_two
    elif ((len(s) == 4) and (len(s[2]) == 1)):
        last_two = "something ".join(s[-2:])  # joins the last two parts using ":"
        msg = last_two
    elif ((len(s) >= 3) and (len(s[2]) != 1)):
        msg = s[2]
    elif (len(s) == 2 and s[1] != "error"):
        msg = s[1]

    return msg


def replace_msg_identifiers(msg):
    # Define a regular expression pattern to match all special characters
    pattern1 = r"(?<=')[^a-zA-Z0-9\s]+(?=')"
    # Define a regular expression pattern to match the phrase for variable, method, class, package names
    pattern2 = r"variable\s+(.*?)\s+might not have been initialized"
    pattern3 = r"variable\s+(.*?)\s+already defined in method\s+(.*?)$"
    pattern4 = r"no suitable method found for\s+(.*?)$"
    pattern5 = r"method\s+(.*?)\s+already defined in class\s+(.*?)$"
    pattern6 = r"method\s+(.*?)\s+in class\s+(.*?)\s+cannot be applied to given types;"
    pattern7 = r"non-static method\s+(.*?)\s+cannot be referenced from a static context"
    pattern8 = r"package\s+(.*?)\s+does not exist"
    pattern9 = r"variable\s+(.*?)\s+already defined in class\s+(.*?)$"
    pattern10 = r"bad operand type\s+(.*?)\s+for unary operator 'ID'"
    pattern11 = r"no suitable constructor found for\s+(.*?)$"
    pattern12 = r"class\s+(.*?)\s+already defined in package unnamed\s+(.*?)$"
    pattern13 = "cannot assign a value to final variable\s+(.*?)$"
    pattern14 = "> expected"
    pattern15 = "< expected"
    pattern16 = "-> expected"
    pattern17 = "'ID' or 'ID' expected"
    pattern18 = "-'ID' expected"
    pattern19 = "Illegal static declaration in inner class\s+(.*?)$"
    pattern20 = ": expected"
    pattern21 = "cannot find symbol: variable\s+(.*?)$"
    pattern22 = "cannot find symbol: method\s+(.*?)$"
    pattern23 = "cannot find symbol: class\s+(.*?)$"

    # Define the replacement text
    replace_text2 = "variable ID might not have been initialized"
    replace_text3 = "variable ID already defined"
    replace_text4 = "no suitable method found for method ID"
    replace_text5 = "method ID already defined in class ID"
    replace_text6 = "method ID in class ID cannot be applied to given types"
    replace_text7 = "non-static method ID cannot be referenced from a static context"
    replace_text8 = "package ID does not exist"
    replace_text9 = "bad operand type for unary operator 'ID'"
    replace_text10 = "no suitable constructor found"
    replace_text11 = "class ID already defined in package unnamed"
    replace_text12 = "cannot assign a value to final variable ID"
    replace_text13 = "'ID' expected"
    replace_text14 = "Illegal static declaration in inner class ID"
    replace_text15 = "cannot find symbol: variable ID"
    replace_text16 = "cannot find symbol: method ID"
    replace_text17 = "cannot find symbol: class ID"

    # Replace all special characters in the 'text' column with a token ID of your choice (in this case, 999)
    msg = re.sub(pattern1, 'ID', msg)

    # Replace the value the variable names with ID
    msg = re.sub(pattern2, replace_text2, msg)
    msg = re.sub(pattern3, replace_text3, msg)
    msg = re.sub(pattern4, replace_text4, msg)
    msg = re.sub(pattern5, replace_text5, msg)
    msg = re.sub(pattern6, replace_text6, msg)
    msg = re.sub(pattern7, replace_text7, msg)
    msg = re.sub(pattern8, replace_text8, msg)
    msg = re.sub(pattern9, replace_text3, msg)
    msg = re.sub(pattern10, replace_text9, msg)
    msg = re.sub(pattern11, replace_text10, msg)
    msg = re.sub(pattern12, replace_text11, msg)
    msg = re.sub(pattern13, replace_text12, msg)
    msg = re.sub(pattern14, replace_text13, msg)
    msg = re.sub(pattern15, replace_text13, msg)
    msg = re.sub(pattern16, replace_text13, msg)
    msg = re.sub(pattern17, replace_text13, msg)
    msg = re.sub(pattern18, replace_text13, msg)
    msg = re.sub(pattern20, replace_text13, msg)
    msg = re.sub(pattern19, replace_text14, msg)
    msg = re.sub(pattern21, replace_text15, msg)
    msg = re.sub(pattern22, replace_text16, msg)
    msg = re.sub(pattern23, replace_text17, msg)

    return msg


def get_all_errors_per_line(error_line, error_list, df_error_ID):
    ID = []
    for error in error_list:
        #         print("the error is ",error)
        line = get_line_number(error)

        if line == error_line:
            message = remove_last_char_if_dot(get_error_message(error))
            ID.append(get_errorID(df_error_ID, message))

    IDs = '_'.join(str(id) for id in ID)


    return IDs


def formatDataset_Error(df_submission, error_list, df_error_ID, main_table):
    data = []
    # Loop through each row in df
    print("number of attempts per problem with errors", df_submission.shape)
    for _, row in df_submission.iterrows():
        submissions = row["submission_event_list"]
        submission_len = len(submissions)

        if submission_len == 1:

            # add the errorID column for this attempt submmision event and the score
            data.append({
                'subject_ID': row['SubjectID'],
                'assignment_ID': row['AssignmentID'],
                'problem_ID': row['ProblemID'],
                'submission_ID': submissions[submission_len-1],
                'codestate_ID': main_table.loc[main_table['EventID'] == submissions[submission_len-1], 'CodeStateID'].values[0],
                'isError': 0,
                'error_ID': "0"
            })
        else:
            for i in range(submission_len):
                error_msgs = error_list.loc[
                    error_list['ParentEventID'] == submissions[i], 'submission_error_list'].values
                if error_msgs.size > 0: #the submission has errors
                    # error_msgs = error_list.loc[error_list['ParentEventID'] == submissions[i], 'submission_error_list'].values
                    error_msgs = error_msgs[0].split("@")
                    error_lines = [get_line_number(msg) for msg in error_msgs]
                    error_lines_unique = sorted(list(set(error_lines)))
                    error_IDs = []
                    for error_line_num in error_lines_unique:
                        IDs = get_all_errors_per_line(error_line_num, error_msgs, df_error_ID)
                        error_IDs.append(IDs)

                    # add the errorID column for this attempt submmision event and the score update base on the submission[i] eventID
                    data.append({
                        'subject_ID': row['SubjectID'],
                        'assignment_ID': row['AssignmentID'],
                        'problem_ID': row['ProblemID'],
                        'submission_ID': submissions[i],
                        'codestate_ID': main_table.loc[main_table['EventID'] == submissions[i], 'CodeStateID'].values[0],
                        'isError': 1,
                        'error_ID': '_'.join(str(id) for id in error_IDs)
                    })
                else:  #student has fixed esxisitng errors
                    data.append({
                        'subject_ID': row['SubjectID'],
                        'assignment_ID': row['AssignmentID'],
                        'problem_ID': row['ProblemID'],
                        'submission_ID': submissions[i],
                        'codestate_ID':
                            main_table.loc[main_table['EventID'] == submissions[i], 'CodeStateID'].values[0],
                        'isError': 0,
                        'error_ID': "0"
                    })

    df = pd.DataFrame(data)
    print("number of attempts with errors and first compiled solutions", df.shape)

    return df


if __name__ == '__main__':
    PATH = "../../data/raw"
    data = ProgSnap2Dataset(PATH)

    df_error_ID = pd.read_csv('../../data/raw/errors/S19errors.csv')
    main_table = data.get_main_table()


    event_list = get_event_list(main_table)
    error_list = get_error_list(main_table)

    main_table = main_table[main_table['EventType'] == "Compile"]

    formated_data = formatDataset_Error(event_list, error_list, df_error_ID, main_table)
    print("the shape of the formated dataset", formated_data.shape)

    formated_data.to_csv('../../data/prepared/errors/error_MainTable.csv', index=False)

    # don't forget to add the student attempts that hard no error while it was their first attempt
