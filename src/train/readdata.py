import numpy as np
import itertools
import pandas as pd
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', '..')  # Adjust the number of '..' as per your file structure
sys.path.append(src_dir)
from src.config import Config


def create_word_index_table(vocab):
    """
    Creating word to index table
    Input:
    vocab: list. The list of the node vocabulary

    """
    ixtoword = {}
    # period at the end of the sentence. make first dimension be end token
    ixtoword[0] = 'END'
    ixtoword[1] = 'UNK'
    wordtoix = {}
    wordtoix['END'] = 0
    wordtoix['UNK'] = 1
    ix = 2
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1
    return wordtoix, ixtoword

def convert_to_idx(sample, node_word_index, path_word_index):
    """
    Converting to the index 
    Input:
    sample: list. One single training sample, which is a code, represented as a list of neighborhoods.
    node_word_index: dict. The node to word index dictionary.
    path_word_index: dict. The path to word index dictionary.

    """
    sample_index = []
    for line in sample:
        components = line.split(",")
        if len(components) > 2:
            if components[0] in node_word_index:
                starting_node = node_word_index[components[0]]
            else:
                starting_node = node_word_index['UNK']
            if components[1] in path_word_index:
                path = path_word_index[components[1]]
            else:
                path = path_word_index['UNK']
            if components[2] in node_word_index:
                ending_node = node_word_index[components[2]]
            else:
                ending_node = node_word_index['UNK']

        sample_index.append([starting_node,path,ending_node])
    return sample_index


class data_reader():
    def __init__(self, config,fold,train_path, val_path, test_path, maxstep, numofques):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.maxstep = maxstep
        self.numofques = numofques
        self.config = config
        self.fold = fold

    def get_data(self, file_path):
        config = Config()
        data = []
        code_df = pd.read_csv(os.path.join(current_dir,"../../data/prepared/DKTFeatures_"+str(config.assignment)+"/labeled_paths.tsv"),sep="\t")
        training_students = np.load(os.path.join(current_dir,"../../data/prepared/DKTFeatures_"+str(config.assignment)+"/training_students.npy"),allow_pickle=True)
        all_training_code = code_df[code_df['subject_ID'].isin(training_students)]['RawASTPath']
        separated_code = []
        for code in all_training_code:
            if type(code) == str:
                separated_code.append(code.split("@"))

        node_hist = {}
        path_hist = {}
        starting_nodes = []
        path = []
        ending_nodes = []

        for paths in separated_code:
            if config.assignment == 487 :
                for p in paths:
                    if len(p.split(",")) > 2:
                        starting_nodes.append(p.split(",")[0])
                        path.append(p.split(",")[1])
                        ending_nodes.append(p.split(",")[2])
            else:
                starting_nodes = [p.split(",")[0] for p in paths]
                path = [p.split(",")[1] for p in paths]
                ending_nodes = [p.split(",")[2] for p in paths]


            nodes = starting_nodes + ending_nodes
            for n in nodes:
                if not n in node_hist:
                    node_hist[n] = 1
                else:
                    node_hist[n] += 1
            for p in path:
                if not p in path_hist:
                    path_hist[p] = 1
                else:
                    path_hist[p] += 1

        node_count = len(node_hist)
        path_count = len(path_hist)

        if config.assignment == 487:
            np.save(os.path.join(current_dir,"../../data/prepared/DKTFeatures_"+str(config.assignment)+"/np_counts_"+str(self.fold)+".npy"), [node_count, path_count])
        else:
            np.save(os.path.join(current_dir,"../../data/prepared/DKTFeatures_"+str(config.assignment)+"/np_counts.npy"), [node_count, path_count])

        # small frequency then abandon, for node and path
        valid_node = [node for node, count in node_hist.items()]
        valid_path = [path for path, count in path_hist.items()]

        # create ixtoword and wordtoix lists
        node_word_index, node_index_word = create_word_index_table(valid_node)
        path_word_index, path_index_word = create_word_index_table(valid_path)

        # get the question embeddings

        #part I question embeddings from the GPT-2
        question_embeddings = pd.read_csv(os.path.join(current_dir, '../../data/raw/question/question_embeddings.csv'))
        q_embeddings = question_embeddings[question_embeddings['AssignmentID'] == config.assignment]

        # part II question embeddings from the bipartite graph embeddings
        embed_data = np.load(os.path.join(current_dir, '../../data/prepared/question/embedding/embedding_'+config.embedds_type+'.npz'))
        _, _, pre_pro_embed = embed_data['pro_repre'], embed_data['skill_repre'], embed_data['pro_final_repre']
        #print(pre_pro_embed.shape, pre_pro_embed.dtype)




        with open(file_path, 'r') as file:
            for lent, css, ques, ans, err in itertools.zip_longest(*[file] * 5):
                lent = int(lent.strip().strip(','))
                ques = [int(q) for q in ques.strip().strip(',').split(',')]
                ans = [int(a) for a in ans.strip().strip(',').split(',')]
                css = [cs for cs in css.strip().strip(',').split(',')]
                err = [er for er in err.strip().strip(',').split(',')]

                #temp = np.zeros(shape=[self.maxstep, 2 * self.numofques+MAX_CODE_LEN*3]) # Skill DKT #1, original

                temp = np.zeros(shape=[self.maxstep, 2 * self.numofques
                                       + self.config.MAX_CODE_LEN*3
                                       + self.config.MAX_QUESTION_LEN_partI
                                       + self.config.MAX_QUESTION_LEN_partII
                                       + self.config.Reference_LEN
                                       + self.config.ErrorID_LEN])

                if lent >= self.maxstep:
                    steps = self.maxstep
                    extra = 0
                    ques = ques[-steps:]
                    ans = ans[-steps:]
                    css = css[-steps:]
                    err = err[-steps:]

                else:
                    steps = lent
                    extra = self.maxstep-steps

                for j in range(steps):

                    #Adding the correctness vector
                    if ans[j] == 1:
                        temp[j+extra][ques[j]] = 1
                    else:
                        temp[j+extra][ques[j] + self.numofques] = 1

                    #extract the code vector
                    code = code_df[code_df['CodeStateID']==css[j]]['RawASTPath'].iloc[0]


                    if type(code) == str:
                        code_paths = code.split("@")
                        raw_features = convert_to_idx(code_paths, node_word_index, path_word_index)
                        if len(raw_features) < self.config.MAX_CODE_LEN:
                            raw_features += [[0,0,0]]*(self.config.MAX_CODE_LEN - len(raw_features))
                        else:
                            raw_features = raw_features[:self.config.MAX_CODE_LEN]


                        features = np.array(raw_features).reshape(-1, self.config.MAX_CODE_LEN*3)


                        temp[j+extra][2*self.numofques: self.config.MAX_CODE_LEN*3 + 2*self.numofques] = features  #[20:320]

                    #extract the question vector
                    if self.config.MAX_QUESTION_LEN_partI > 0 and self.config.MAX_QUESTION_LEN_partII > 0:
                        # extract the question embeddings for Part I GPT_2
                        question_embedding = q_embeddings.loc[q_embeddings['ProblemID'] == ques[j], 'prompt-embedding']
                        question_embedding = question_embedding[ques[j]]
                        values_str = question_embedding[question_embedding.index('[') + 1:question_embedding.index(']')]


                        question_embeds_partI = [float(val.strip()) for val in values_str.split(',')] # Split the values by comma and remove any leading/trailing whitespace
                        question_embeds_partI = np.array(question_embeds_partI).reshape(1,self.config.MAX_QUESTION_LEN_partI)

                        # extract the question embeddings for Part II bipartite
                        question_embeds_partII = pre_pro_embed[ques[j]]
                        question_embeds_partII = question_embeds_partII.reshape(1,self.config.MAX_QUESTION_LEN_partII)

                        question_embeds = np.concatenate((question_embeds_partI, question_embeds_partII), axis=1)

                    elif self.config.MAX_QUESTION_LEN_partII == 0 and self.config.MAX_QUESTION_LEN_partI > 0:
                        # extract the question embeddings for Part I GPT_2
                        question_embedding = q_embeddings.loc[q_embeddings['ProblemID'] == ques[j], 'prompt-embedding']
                        question_embedding = question_embedding[ques[j]]
                        values_str = question_embedding[question_embedding.index('[') + 1:question_embedding.index(']')]


                        question_embeds_partI = [float(val.strip()) for val in values_str.split(',')] # Split the values by comma and remove any leading/trailing whitespace
                        question_embeds = np.array(question_embeds_partI).reshape(1,self.config.MAX_QUESTION_LEN_partI)


                    elif self.config.MAX_QUESTION_LEN_partII > 0 and self.config.MAX_QUESTION_LEN_partI == 0:

                        # extract the question embeddings for Part II bipartite
                        question_embeds_partII = pre_pro_embed[ques[j]]
                        question_embeds = question_embeds_partII.reshape(1,self.config.MAX_QUESTION_LEN_partII)

                    section_two_question = self.config.MAX_QUESTION_LEN_partI + self.config.MAX_QUESTION_LEN_partII +self.config.MAX_CODE_LEN*3 + 2*self.numofques
                    temp[j+extra][self.config.MAX_CODE_LEN*3 + 2*self.numofques: section_two_question ] = question_embeds    #[320: 868]


                    # extract the reference embeddings
                    reference_embeds = q_embeddings.loc[q_embeddings['ProblemID'] == ques[j], 'reference-embedding']
                    reference_embeds = reference_embeds[ques[j]]
                    values_str = reference_embeds[reference_embeds.index('[') + 1:reference_embeds.index(']')]
                    reference_embeds = [float(val.strip()) for val in values_str.split(',')] # Split the values by comma and remove any leading/trailing whitespace
                    reference_embeds = np.array(reference_embeds).reshape(1,self.config.Reference_LEN)

                    section_two_reference = self.config.Reference_LEN + section_two_question
                    temp[j+extra][section_two_question: section_two_reference] = reference_embeds   #[320+ 868: 320+ 868+200]

                    #extract error embeddings
                    errorID_embeds = self.get_errorID_embeddings(err[j])
                    temp[j + extra][section_two_reference:] = errorID_embeds


                data.append(temp.tolist())
            print('done: ' + str(np.array(data).shape))
        return data

    def get_errorID_embeddings(self, error):

        if '_' in error:

            error_ids = [int(id) for id in error.split('_')]
        else:
            error_ids = [error]

        # Remove duplicates and convert to integers
        error_ids = list(set(error_ids))
        error_ids = [int(id) for id in error_ids]

        # Create a vector of size 84 with 1s at the specified indices
        vector_size = self.config.ErrorID_LEN
        vector = np.zeros((1, vector_size))
        vector[0, error_ids] = 1

        return vector

    def get_train_data(self):
        print('loading train data...',self.train_path)
        print ('the path is ',self.train_path)
        train_data = self.get_data(self.train_path)
        val_data = self.get_data(self.val_path)
        if self.config.assignment == 487:
            np.save(os.path.join(current_dir,"../../data/prepared/DKTFeatures_"+str(self.config.assignment)
                                 +"/train_data_"+str(self.fold) +".npy"), np.array(train_data+val_data))
        return np.array(train_data+val_data)

    def get_test_data(self):
        print('loading test data...',self.test_path)
        test_data = self.get_data(self.test_path)
        if self.config.assignment == 487:
            np.save(os.path.join(current_dir,"../../data/prepared/DKTFeatures_"+str(self.config.assignment)
                                 +"/test_data_"+str(self.fold) +".npy"), np.array(test_data))
        return np.array(test_data)
