import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(ROOT_DIR, "data")
RAW_PATH = os.path.join(DATA_PATH, "raw")
PROCESSED_PATH = os.path.join(DATA_PATH, "processed")
REVIEW_IMAGES_PATH = os.path.join(DATA_PATH, "review_images")

SRC_PATH = os.path.join(ROOT_DIR, "src")
MODEL_PATH = os.path.join(ROOT_DIR, "model")
SAVE_MODEL_PATH = os.path.join(MODEL_PATH, "save")

for _p in (RAW_PATH, PROCESSED_PATH, REVIEW_IMAGES_PATH, SAVE_MODEL_PATH):
    os.makedirs(_p, exist_ok=True)
