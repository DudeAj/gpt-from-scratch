from pathlib import Path
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_DIR       = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
TOKENIZER_DIR  = PROJECT_ROOT / "tokenizer"

# ---------------------------------------------------------------------------
# Shakespeare (original small experiment)
# ---------------------------------------------------------------------------
DATASET_NAME = "tiny_shakespeare"
TEXT_FILE    = DATA_DIR / "tiny_shakespeare.txt"
TRAIN_BIN    = DATA_DIR / "train.bin"
VAL_BIN      = DATA_DIR / "val.bin"
TRAIN_SPLIT  = 0.9

VOCAB_SIZE = 1000

MAX_SEQ_LEN = 128
BLOCK_SIZE  = MAX_SEQ_LEN

D_MODEL      = 128
NUM_HEADS    = 4
NUM_LAYERS   = 4
FFN_EXPANSION = 4
DROPOUT      = 0.1

BATCH_SIZE      = 64
LEARNING_RATE   = 3e-4
WEIGHT_DECAY    = 0.01
EPOCHS          = 10
STEPS_PER_EPOCH = 500

MAX_NEW_TOKENS = 200

SEED = 42

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ---------------------------------------------------------------------------
# GPT-2 small — pretraining on WikiText-103
# ---------------------------------------------------------------------------
# Architecture matches the original GPT-2 small (117M params).
# With WikiText-103 (~500MB) this trains in ~6-10 hrs on M3 MPS,
# or ~1-2 hrs on a GPU. Use pretrain.py to run this stage.

PRETRAIN_BLOCK_SIZE = 256          # longer context than shakespeare run

PRETRAIN_D_MODEL      = 768        # GPT-2 small hidden size
PRETRAIN_NUM_HEADS    = 12         # 64-dim per head (768/12)
PRETRAIN_NUM_LAYERS   = 12         # 12 transformer blocks
PRETRAIN_FFN_EXPANSION = 4
PRETRAIN_DROPOUT      = 0.1

PRETRAIN_BATCH_SIZE      = 16      # smaller batch — 768-dim is much heavier
PRETRAIN_LEARNING_RATE   = 3e-4
PRETRAIN_WEIGHT_DECAY    = 0.1
PRETRAIN_EPOCHS          = 3       # 3 passes over WikiText-103
PRETRAIN_STEPS_PER_EPOCH = 2000    # ~2000 steps × 3 epochs = 6000 updates

WIKITEXT_DIR  = DATA_DIR / "wikitext"
PRETRAIN_CKPT = CHECKPOINT_DIR / "pretrain_best.pt"

# ---------------------------------------------------------------------------
# Fine-tuning on DailyDialog (conversation)
# ---------------------------------------------------------------------------
# Fine-tuning starts from the pretrained checkpoint and specialises the
# model on dialogue. Lower LR (10x) to avoid catastrophic forgetting.

FINETUNE_BLOCK_SIZE = 256

FINETUNE_BATCH_SIZE      = 8       # short dialogues → smaller batch fine
FINETUNE_LEARNING_RATE   = 3e-5    # 10x lower than pretraining
FINETUNE_WEIGHT_DECAY    = 0.01
FINETUNE_EPOCHS          = 5
FINETUNE_STEPS_PER_EPOCH = 500

DAILYDIALOG_DIR = DATA_DIR / "dailydialog"
FINETUNE_CKPT   = CHECKPOINT_DIR / "finetune_best.pt"

# Special tokens used to format dialogue turns:
#   <human> user turn </human> <assistant> model reply </assistant>
HUMAN_TOKEN     = "<human>"
ASSISTANT_TOKEN = "<assistant>"
END_TOKEN       = "</s>"
