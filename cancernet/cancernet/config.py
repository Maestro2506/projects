import os

INPUT_DATASET = "/content/drive/My Drive/DeepLearn/datasets/original"

BASE_PATH = "/content/drive/My Drive/DeepLearn/datasets/idc"
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
