# CSEDM Dataset and A-DKT Model Training

This repository contains code for training a Knowledge Tracing (KT) model using the CSEDM dataset. The CSEDM dataset can be found here.

## Dataset Preparation

To train our KT model, the data needs to be processed. There are two major steps involved in the process.

### Part I: Prepare the dataset 
First is to select students attempts with errors and their first compiled attempt for every question Features 

1. Run src/prepare/preprocessing_errors.py to select student and error associated with their attempts 

### Part II: Prepare the DKT Features
To prepare the DKT features for training, validation, and testing, follow these steps:

1. Run src/prepare/preprocessing.py to generate the DKT features.
This script will process the data and generate the necessary training, validation, and testing features.

2. Run src/prepare/path_extractor.py to extract paths for all submitted codes that are parsable. These paths will be used in code2vec.

### Part III: Generate Bipartite Graph Embedding for the Questions
To generate the embeddings for the questions using the Product Layer and PEBG model, follow these steps:

1. Run src/prepare/preprocessing_question.py to create all features for training a bipartite
2. Run src/prepare/run_pebg.py to generate the embeddings using the implemented models.

This step will generate the necessary bipartite graph embeddings for the questions.
The product later is implemented in PNN.py and PEBG model in Pebg_model.py. For more details, please refer to the cited paper.

## Training the KT Model

After preparing and processing all the necessary input features, you can train the KT model by following these steps:

1. Run src/run.py.
This script will train the KT model based on the configuration settings specified in config.py.

Make sure to set the appropriate configurations in config.py before running the script.


