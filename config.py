from pathlib import Path

_ROOT = Path(__file__).parent

WARP_SIZE          = 900
MNIST_SIZE         = 28
DIGIT_BOX          = 20
CONFIDENCE_THRESHOLD = 0.0

MODEL_PATH     = str(_ROOT / "models" / "chars74_cnn.keras")
DATA_PATH      = str(_ROOT / "data"   / "v2_train" / "v2_train")
TEST_DATA_PATH = str(_ROOT / "data"   / "v2_test"  / "v2_test")
OUT_DIR        = str(_ROOT / "outputs")