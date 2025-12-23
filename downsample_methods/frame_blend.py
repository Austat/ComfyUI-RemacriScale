import cv2
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def _blend_pair(args):
    frame1, frame2 = args
    return cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)

def frame_blend(np_frames, indices, threads):
    tasks = []
    for idx in indices:
        if idx + 1 < len(np_frames):
            tasks.append((np_frames[idx], np_frames[idx + 1]))
        else:
            tasks.append((np_frames[idx], np_frames[idx]))

    selected = []
    with ThreadPoolExecutor(max_workers=threads) as executor:
        for result in tqdm(executor.map(_blend_pair, tasks),
                           total=len(tasks),
                           desc="Blending",
                           colour="blue",
                           unit="frame"):
            selected.append(result)

    return selected