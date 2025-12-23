import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def _optical_flow_pair(args):
    frame1, frame2 = args

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2,
        None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    h, w = gray1.shape
    flow_half = flow * 0.5

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow_half[..., 0]).astype(np.float32)
    map_y = (grid_y + flow_half[..., 1]).astype(np.float32)

    return cv2.remap(frame1, map_x, map_y, cv2.INTER_LINEAR)

def optical_flow(np_frames, indices, threads):
    tasks = []
    for idx in indices:
        if idx + 1 < len(np_frames):
            tasks.append((np_frames[idx], np_frames[idx + 1]))
        else:
            tasks.append((np_frames[idx], np_frames[idx]))

    selected = []
    with ThreadPoolExecutor(max_workers=threads) as executor:
        for result in tqdm(executor.map(_optical_flow_pair, tasks),
                           total=len(tasks),
                           desc="Optical Flow",
                           colour="blue",
                           unit="frame"):
            selected.append(result)

    return selected