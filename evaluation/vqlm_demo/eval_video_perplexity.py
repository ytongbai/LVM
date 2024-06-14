
import os
import glob
from functools import partial
from tqdm import tqdm, trange
from multiprocessing import Pool
from PIL import Image
import cv2
import mlxu
from natsort import natsorted
import numpy as np
import einops
import torch

from vqlm_demo.inference import MultiProcessInferenceModel
from vqlm_demo.utils import (
    is_video, random_square_crop,
    read_frames_from_dir, read_frames_from_video
)


FLAGS, _ = mlxu.define_flags_with_default(
    checkpoint='',
    input_files='',
    frame_input=False,
    read_file_list='',
    center_crop=1.0,
    n_context_frames=15,
    n_target_frames=1,
    n_workers=8,
    stride=8,
    batch_size=2,
    torch_devices='',
    shuffle=False,
    random_start=True,
    max_examples=0,
)


class VideoDataset(torch.utils.data.Dataset):

    def __init__(self, videos, frame_input=False, n_context_frames=15,
                 n_target_frames=1, stride=1):
        self.videos = videos
        self.frame_input = frame_input
        self.n_context_frames = n_context_frames
        self.n_target_frames = n_target_frames
        self.stride = stride

    def __getitem__(self, index):
        if self.frame_input:
            frames = read_frames_from_dir(
                self.videos[index],
                self.n_context_frames + self.n_target_frames,
                self.stride,
                center_crop=FLAGS.center_crop,
                random_start=FLAGS.random_start,
            )
        else:
            frames = read_frames_from_video(
                self.videos[index],
                self.n_context_frames + self.n_target_frames,
                self.stride,
                center_crop=FLAGS.center_crop,
                random_start=FLAGS.random_start,
            )
        if frames is None:
            return self[np.random.randint(0, len(self))]
        return frames[:self.n_context_frames], frames[self.n_context_frames:]

    def __len__(self):
        return len(self.videos)



def main(_):
    assert FLAGS.checkpoint != ''
    assert FLAGS.read_file_list != '' or FLAGS.input_files != ''

    model = MultiProcessInferenceModel(
        checkpoint=FLAGS.checkpoint,
        torch_devices=FLAGS.torch_devices,
        perplexity_batch_size=FLAGS.batch_size,
    )

    if FLAGS.read_file_list != '':
        with open(FLAGS.read_file_list, 'r') as f:
            videos = [x.strip() for x in f.readlines()]
    else:
        videos = glob.glob(FLAGS.input_files)

    if FLAGS.frame_input:
        videos = [x for x in videos if os.path.isdir(x)]
    else:
        videos = [x for x in videos if is_video(x)]

    if FLAGS.shuffle:
        np.random.shuffle(videos)

    if FLAGS.max_examples > 0:
        videos = videos[:FLAGS.max_examples]

    dataset = VideoDataset(
        videos,
        frame_input=FLAGS.frame_input,
        n_context_frames=FLAGS.n_context_frames,
        n_target_frames=FLAGS.n_target_frames,
        stride=FLAGS.stride
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size * model.n_processes * 4,
        shuffle=False,
        num_workers=FLAGS.n_workers,
        prefetch_factor=4,
        drop_last=True,
    )

    perplexities = []

    for batch_context_frames, batch_taret_frames in tqdm(dataloader, ncols=0):
        batch_context_frames = batch_context_frames.numpy()
        batch_taret_frames = batch_taret_frames.numpy()
        perplexity = model.compute_perplexity(
            batch_context_frames, batch_taret_frames
        )
        perplexities.append(perplexity)

    perplexities = np.concatenate(perplexities, axis=0)
    print(f'Perplexity: {np.mean(perplexities)}')


if __name__ == '__main__':
    mlxu.run(main)