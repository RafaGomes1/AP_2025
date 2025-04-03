import os
import numpy as np
import tensorflow as tf
import random

# Reprodutibilidade
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Hiperparâmetros
MAX_WORDS = 20000
MAX_LEN = 500
EMBEDDING_DIM = 128
BATCH_SIZE = 64
EPOCHS = 10

# Paths
TRAIN_INPUT_PATH = "../data/test_input.csv"
TRAIN_OUTPUT_PATH = "../data/test_output.csv"
VAL_INPUT_PATH = "../data/human_ai_input.csv"
VAL_OUTPUT_PATH = "../data/human_ai_output.csv"
TEST_INPUT_PATH = "../data/dataset3_inputs.csv"
OUTPUT_FILE = "results/dnn_s2.csv"
