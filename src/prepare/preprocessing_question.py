import os
import pandas as pd
import numpy as np
from scipy import sparse
current_dir = os.path.dirname(os.path.abspath(__file__))

class DataProcess():
    def __init__(self, data_folder_main='../../data/raw', file_name='MainTable.csv'):
        print("Process Dataset %s" % os.path.join(data_folder_main, file_name))
        self.data_folder_input = os.path.join(current_dir, '../../data/raw/question')
        self.data_folder_output = os.path.join(current_dir, '../../data/prepared/question')
        self.data_folder_main = os.path.join(current_dir, data_folder_main)
        self.file_name = file_name

        if not os.path.isdir(self.data_folder_output):
            os.mkdir(self.data_folder_output)


    def pro_skill_graph(self):
        df = pd.read_csv(os.path.join(self.data_folder_main, self.file_name))
        df_question_concept = pd.read_csv(os.path.join(self.data_folder_input, "prompt_concept.csv"))
        df_concept = pd.read_csv(os.path.join(self.data_folder_input, "concepts.csv"))

        for index, row in df_question_concept.iterrows():
            tmp_skills = row[(row == 1) & (row.index != 'ProblemID')].index.tolist()
            combine_skills = [df_concept.loc[df_concept['name'] == skill, 'id'].iloc[0] for skill in tmp_skills]
            combine_skills = '_'.join(str(num) for num in combine_skills)
            df_question_concept.at[index, 'combined_concepts'] = combine_skills

        problems = df_question_concept['ProblemID']
        pro_id_dict_original = dict(zip(problems, range(len(problems))))
        pro_id_dict = dict(zip( range(len(problems)),problems))

        print('problem number %d' % len(problems))

        # pro_type = df['answer_type'].unique()
        # pro_type_dict = dict(zip(pro_type, range(len(pro_type))))
        # print('problem type: ', pro_type_dict)

        pro_feat = []
        pro_skill_adj = []
        skill_id_dict, skill_cnt = {}, 0
        for pro_id in range(len(problems)):
            tmp_df = df[df['ProblemID']==problems[pro_id]]
            tmp_df_0 = tmp_df.iloc[0]

            # pro_feature: [no_of_words, num_attempts, mean_correct_num]
            question = df_question_concept[df_question_concept['ProblemID']==pro_id_dict[pro_id]]
            text = question['Requirement'].values[0]
            no_of_words = len(text.split())
            num_attempts = len(tmp_df[tmp_df["EventType"] == "Run.Program"])
            num_correct_attempts = len(tmp_df[(tmp_df["EventType"] == "Run.Program") & (tmp_df["Score"] == 1)])

            tmp_pro_feat = [0.] * 3
            tmp_pro_feat[0] = no_of_words
            tmp_pro_feat[1] = num_attempts
            tmp_pro_feat[2] = num_correct_attempts/num_attempts
            pro_feat.append(tmp_pro_feat)

            # build problem-skill bipartite
            tmp_skills = question.columns[(question.eq(1).any()) & (question.columns != 'ProblemID')].tolist()
            for s in tmp_skills:
                s = df_concept[df_concept['name'] == s]['id'].values[0]
                if s not in skill_id_dict:
                    skill_id_dict[s] = skill_cnt
                    skill_cnt += 1
                pro_skill_adj.append([pro_id, skill_id_dict[s], 1])

        pro_skill_adj = np.array(pro_skill_adj).astype(np.int32)
        pro_feat = np.array(pro_feat).astype(np.float32)
        
        # normalise the [no_of_words, num_attempts] to range from 0 to 1
        pro_feat[:, 0] = (pro_feat[:, 0] - np.min(pro_feat[:, 0])) / (np.max(pro_feat[:, 0])-np.min(pro_feat[:, 0]))
        pro_feat[:, 1] = (pro_feat[:, 1] - np.min(pro_feat[:, 1])) / (np.max(pro_feat[:, 1]) - np.min(pro_feat[:, 1]))
        pro_num = np.max(pro_skill_adj[:, 0]) + 1
        skill_num = np.max(pro_skill_adj[:, 1]) + 1
        print('problem number %d, skill number %d' % (pro_num, skill_num))

        # save pro-skill-graph in sparse matrix form
        pro_skill_sparse = sparse.coo_matrix((pro_skill_adj[:, 2].astype(np.float32), (pro_skill_adj[:, 0], pro_skill_adj[:, 1])), shape=(pro_num, skill_num))
        sparse.save_npz(os.path.join(self.data_folder_output, 'pro_skill_sparse.npz'), pro_skill_sparse)

        # take joint skill as a new skill
        skills = df_question_concept['combined_concepts'].unique()
        for s in skills:
            if '_' in s:
                skill_id_dict[s] = skill_cnt
                skill_cnt += 1

        # save the update problem_concept dataframe
        df_question_concept.to_csv(os.path.join(self.data_folder_output,'combined_concepts.csv'), index=False)
        # save pro-id-dict, skill-id-dict
        self.save_dict(pro_id_dict_original, os.path.join(self.data_folder_output, 'pro_id_dict.txt'))
        self.save_dict(pro_id_dict, os.path.join(self.data_folder_output, 'id_pro_dict.txt'))
        self.save_dict(skill_id_dict, os.path.join(self.data_folder_output, 'skill_id_dict.txt'))

        # save pro_feat_arr
        np.savez(os.path.join(self.data_folder_output, 'pro_feat.npz'), pro_feat=pro_feat)

        return pro_skill_sparse

    # to extract the implicit similarrity between questions and skills
    def extract_pro_pro_sim(self,pro_skill_csr,pro_num):
        # extract pro-pro similarity sparse matrix
        pro_pro_adj = []
        for p in range(pro_num):
            tmp_skills = pro_skill_csr.getrow(p).indices
            similar_pros = pro_skill_csc[:, tmp_skills].indices
            zipped = zip([p] * similar_pros.shape[0], similar_pros)
            pro_pro_adj += list(zipped)

        pro_pro_adj = list(set(pro_pro_adj))
        pro_pro_adj = np.array(pro_pro_adj).astype(np.int32)
        data = np.ones(pro_pro_adj.shape[0]).astype(np.float32)
        pro_pro_sparse = sparse.coo_matrix((data, (pro_pro_adj[:, 0], pro_pro_adj[:, 1])), shape=(pro_num, pro_num))

        sparse.save_npz(os.path.join(self.data_folder_output, 'pro_pro_sparse.npz'), pro_pro_sparse)

    def extract_skill_skill_sim(self,pro_skill_csc,skill_num):
        # extract skill-skill similarity sparse matrix
        skill_skill_adj = []
        for s in range(skill_num):
            tmp_pros = pro_skill_csc.getcol(s).indices
            similar_skills = pro_skill_csr[tmp_pros, :].indices
            zipped = zip([s] * similar_skills.shape[0], similar_skills)
            skill_skill_adj += list(zipped)

        skill_skill_adj = list(set(skill_skill_adj))
        skill_skill_adj = np.array(skill_skill_adj).astype(np.int32)
        data = np.ones(skill_skill_adj.shape[0]).astype(np.float32)
        skill_skill_sparse = sparse.coo_matrix((data, (skill_skill_adj[:, 0], skill_skill_adj[:, 1])),
                                               shape=(skill_num, skill_num))
        sparse.save_npz(os.path.join(self.data_folder_output, 'skill_skill_sparse.npz'), skill_skill_sparse)



    def save_dict(self, dict_name, file_name):
        f = open(file_name, 'w')
        f.write(str(dict_name))
        f.close


    def write_txt(self, file, data):
        with open(file, 'w') as f:
            for dd in data:
                for d in dd:
                    f.write(str(d)+'\n')


if __name__ == '__main__':
    DP = DataProcess()
    DP.pro_skill_graph()

    pro_skill_coo = sparse.load_npz(os.path.join(current_dir, "../../data/prepared/question/pro_skill_sparse.npz"))
    [pro_num, skill_num] = pro_skill_coo.toarray().shape
    print('problem number %d, skill number %d' % (pro_num, skill_num))
    pro_skill_csc = pro_skill_coo.tocsc()
    pro_skill_csr = pro_skill_coo.tocsr()

    DP.extract_pro_pro_sim(pro_skill_csr, pro_num)
    DP.extract_skill_skill_sim(pro_skill_csc, skill_num)

