#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torchvision
from transforms import VideoTransform
from video_process import VideoProcess


class AVSRDataLoader:
    def __init__(
        self,
        speed_rate=1,
        convert_gray=True,
    ):
        self.video_process = VideoProcess(convert_gray=convert_gray)
        self.video_transform = VideoTransform(speed_rate=speed_rate)

    def load_data(self, data_filename, landmarks=None):
        video = self.load_video(data_filename)
        video = self.video_process(video, landmarks)
        video = torch.tensor(video)
        return video

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()
