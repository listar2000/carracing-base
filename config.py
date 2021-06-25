from typing import Dict

TRAINING_CONFIG: Dict = {
    # how many consecutive frames to stack together
    "frame_stack_num": 3,
    "skip_frame": 2,
    "done_threshold": 500
}