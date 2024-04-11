#!/usr/bin/env python
# coding: utf-8

# # Tuning an LLM with RLHF

# #### Project environment setup
# 
# The RLHF training process has been implemented in a machine learning pipeline as part of the (Google Cloud Pipeline Components) library. This can be run on any platform that supports KubeFlow Pipelines (an open source framework), and can also run on Google Cloud's Vertex AI Pipelines.
# 
# 


get_ipython().system('pip3 install google-cloud-pipeline-components')
get_ipython().system('pip3 install kfp')


# ### Compile the pipeline

# Import the pipeline & components
from google_cloud_pipeline_components.preview.llm \
import rlhf_pipeline


# Import from KubeFlow pipelines
from kfp import compiler


# Define a path to the yaml file. This file contains the RlHF pipeline definition
RLHF_PIPELINE_PKG_PATH = "rlhf_pipeline.yaml"


# Execute the compile function
compiler.Compiler().compile(
    pipeline_func=rlhf_pipeline,
    package_path=RLHF_PIPELINE_PKG_PATH
)


# To preview the pipeline content, we print the first lines of the YAML file
get_ipython().system('head rlhf_pipeline.yaml')


# ## Define the Vertex AI pipeline job

# ### Calculate the number of reward model training steps
# 
# **reward_model_train_steps** is the number of steps to use when training the reward model.  This depends on the size of your preference dataset. We recommend the model should train over the preference dataset for 20-30 epochs for best results.
# 
# $$ stepsPerEpoch = \left\lceil \frac{datasetSize}{batchSize} \right\rceil$$
# $$ trainSteps = stepsPerEpoch \times numEpochs$$
# 
# The RLHF pipeline parameters are asking for the number of training steps and not number of epochs. Here's an example of how to go from epochs to training steps, given that the batch size for this pipeline is fixed at 64 examples per batch.
# 
# 

# Preference dataset size
PREF_DATASET_SIZE = 3000


# Batch size is fixed at 64
BATCH_SIZE = 64


import math


REWARD_STEPS_PER_EPOCH = math.ceil(PREF_DATASET_SIZE / BATCH_SIZE)
print(REWARD_STEPS_PER_EPOCH)


REWARD_NUM_EPOCHS = 30


# Calculate number of steps in the reward model training
reward_model_train_steps = REWARD_STEPS_PER_EPOCH * REWARD_NUM_EPOCHS


print(reward_model_train_steps)


# ### Calculate the number of reinforcement learning training steps
# The **reinforcement_learning_train_steps** parameter is the number of reinforcement learning steps to perform when tuning the base model. 
# - The number of training steps depends on the size of your prompt dataset. Usually, this model should train over the prompt dataset for roughly 10-20 epochs.
# - Reward hacking: if given too many training steps, the policy model may figure out a way to exploit the reward and exhibit undesired behavior.

# Prompt dataset size
PROMPT_DATASET_SIZE = 2000

# Batch size is fixed at 64
BATCH_SIZE = 64


import math

RL_STEPS_PER_EPOCH = math.ceil(PROMPT_DATASET_SIZE / BATCH_SIZE)
print(RL_STEPS_PER_EPOCH)


RL_NUM_EPOCHS = 10


# Calculate the number of steps in the RL training
reinforcement_learning_train_steps = RL_STEPS_PER_EPOCH * RL_NUM_EPOCHS

print(reinforcement_learning_train_steps)


# ### Define the instruction
# 
# 

# Completed values for the dictionary
parameter_values={
        "preference_dataset": \
    "gs://vertex-ai/generative-ai/rlhf/text/summarize_from_feedback_tfds/comparisons/train/*.jsonl",
        "prompt_dataset": \
    "gs://vertex-ai/generative-ai/rlhf/text/reddit_tfds/train/*.jsonl",
        "eval_dataset": \
    "gs://vertex-ai/generative-ai/rlhf/text/reddit_tfds/val/*.jsonl",
        "large_model_reference": "llama-2-7b",
        "reward_model_train_steps": 10000,
        "reinforcement_learning_train_steps": 2270, 
        "reward_model_learning_rate_multiplier": 1.0,
        "reinforcement_learning_rate_multiplier": 0.2,
        "kl_coeff": 0.1,
        "instruction":\
    "Summarize text in less than 100 words."}


# ### Set up Google Cloud to run the Vertex AI pipeline

# Authenticate in utils
from utils import authenticate
credentials, PROJECT_ID, STAGING_BUCKET = authenticate()

# RLFH pipeline is available in this region
REGION = "europe-west4"


# ## Run the pipeline job on Vertex AI
# 
# Now that we have created our dictionary of values, we can create a PipelineJob. Meaning, that the RLHF pipeline will execute on Vertex AI. So it's not running locally here in the notebook, but on some server on Google Cloud.

import google.cloud.aiplatform as vertexAI


vertexAI.init(project = PROJECT_ID,
                location = REGION,
                credentials = credentials)


# Look at the path for the YAML file
RLHF_PIPELINE_PKG_PATH


# Create a job
job = aiplatform.PipelineJob(
    display_name="tutorial-rlhf-tuning",
    pipeline_root=STAGING_BUCKET,
    template_path=RLHF_PIPELINE_PKG_PATH,
    parameter_values=parameter_values)

#Run the job
job.run()

