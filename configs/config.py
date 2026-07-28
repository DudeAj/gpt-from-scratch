from pathlib import Path
import torch

# -----------------------
# Paths
# -----------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
TOKENIZER_DIR = PROJECT_ROOT / "tokenizer"

# -----------------------
# Dataset
# -----------------------

DATASET_NAME = "tiny_shakespeare"

TEXT_FILE = DATA_DIR / "tiny_shakespeare.txt"

TRAIN_BIN = DATA_DIR / "train.bin"
VAL_BIN = DATA_DIR / "val.bin"

TRAIN_SPLIT = 0.9

# -----------------------
# Tokenizer
# -----------------------

VOCAB_SIZE = 1000

# -----------------------
# Model
# -----------------------

MAX_SEQ_LEN = 128
BLOCK_SIZE = MAX_SEQ_LEN  # alias used by dataset/training code

D_MODEL = 256
NUM_HEADS = 8
NUM_LAYERS = 6

FFN_EXPANSION = 4

DROPOUT = 0.1

# -----------------------
# Training
# -----------------------

BATCH_SIZE = 32

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 0.01

EPOCHS = 20

# -----------------------
# Generation
# -----------------------

MAX_NEW_TOKENS = 200

# -----------------------
# Misc
# -----------------------

SEED = 42

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)