import numpy as np
from tqdm import tqdm

def frame_drop(np_frames, indices):
    selected = []
    for i in tqdm(indices, desc="Dropping", unit="frame", colour="blue"):
        if i < len(np_frames):
            selected.append(np_frames[i])
    return selected