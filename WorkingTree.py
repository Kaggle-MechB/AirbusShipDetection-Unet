# coding: utf-8

from pathlib import Path
import os

ROOT_DIR = Path.cwd()
INPUT_DIR = ROOT_DIR.parent/'input'
TRAIN_DATA_PATH = INPUT_DIR/'train'
TEST_DATA_PATH = INPUT_DIR/'test'
SEG_FILE_PATH = INPUT_DIR/'train_ship_segmentations_v2.csv'
MODEL_DIR = ROOT_DIR/'model'
PHOTO_DIR = ROOT_DIR/'photo'
SUBMISSION_DIR = ROOT_DIR/'submission'

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

if not os.path.exists(PHOTO_DIR):
    os.mkdir(PHOTO_DIR)

if not os.path.exists(SUBMISSION_DIR):
    os.mkdir(SUBMISSION_DIR)
