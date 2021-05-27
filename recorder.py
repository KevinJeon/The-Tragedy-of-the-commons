import cv2
import imageio
import os
import numpy as np

from utils.sys import make_dir


class VideoRecorder(object):
    def __init__(self, root_dir, height=608, width=800, camera_id=0, fps=30):
        self.save_dir = make_dir(root_dir, 'video') if root_dir else None
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled

    def record(self, env):
        if self.enabled:
            frame = env.render(coordination=False)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = frame * 255.
            frame = np.array(frame, dtype=np.uint8)
            self.frames.append(frame)

    def record_observation(self, frame):
        if self.enabled:
            frame = cv2.resize(frame, (96, 96))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = frame * 255.
            frame = np.array(frame, dtype=np.uint8)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.save_dir, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)