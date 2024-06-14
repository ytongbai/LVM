
import os
import glob
from functools import partial
from tempfile import NamedTemporaryFile
import random
import json
from base64 import b64encode
from tqdm import tqdm, trange

import numpy as np
np.float = np.float64
np.int = np.int_
import mlxu
from natsort import natsorted

import torch

import jax
import jax.numpy as jnp
import flax
import einops

from PIL import Image
from muse import VQGANModel
from utils import (
    list_dir_with_full_path, is_image, read_image_to_tensor,
    randomly_subsample_frame_indices
)


FLAGS, _ = mlxu.define_flags_with_default(
    input_dirs='',
    output_file='',
    batch_size=1,
    n_frames=16,
    n_shots=2,
    n_epochs=1,
    n_workers=8,
    max_stride=4,
    dtype='fp32',
)


class MultiVideoDataset(torch.utils.data.Dataset):

    def __init__(self, videos, n_frames=8):
        self.videos = videos
        self.n_tasks = len(videos[0])
        self.n_frames = n_frames

    def __getitem__(self, index):
        n_frames = len([x for x in list_dir_with_full_path(self.videos[index][0]) if is_image(x)])
        for i in range(self.n_tasks):
            if len(
                [x for x in list_dir_with_full_path(self.videos[index][i])
                 if is_image(x)]
            ) != n_frames:
                print('Inconsistent number of frames')
                return self[np.random.randint(0, len(self))]
        if n_frames < self.n_frames:
            print(n_frames)
            return self[np.random.randint(0, len(self))]

        indices = randomly_subsample_frame_indices(
            n_frames, self.n_frames, FLAGS.max_stride,
            random_start=True
        )

        all_frames = []
        for task_idx in range(self.n_tasks):
            frames = []
            all_files = [x for x in list_dir_with_full_path(self.videos[index][task_idx]) if is_image(x)]
            all_files = natsorted(all_files)
            for idx in indices:
                frames.append(read_image_to_tensor(all_files[idx]))

            all_frames.append(np.stack(frames, axis=0))

        return np.stack(all_frames, axis=0)

    def __len__(self):
        return len(self.videos)


def main(argv):
    assert FLAGS.input_dirs != ''
    assert FLAGS.output_file != ''

    # Load the pre-trained vq model from the hub
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = VQGANModel.from_pretrained('/home/vqlm/muse/ckpts/laion').to(device)
    net.eval()

    video_dirs = [sorted(glob.glob(x)) for x in FLAGS.input_dirs.split('::')]
    n_tasks = len(video_dirs)

    groups = {}

    for videos in [sorted(glob.glob(x)) for x in FLAGS.input_dirs.split('::')]:
        for video in videos:
            name = video.split('/')[-1]
            # name = video[:-len(video.split('/')[-1].split('_')[-1]) - 1]
            if name not in groups:
                groups[name] = []
            groups[name].append(video)

    video_dirs = []
    for name, videos in groups.items():
        if len(videos) == n_tasks:
            video_dirs.append(videos)


    with open(FLAGS.output_file, 'w') as fout:
        dataset = MultiVideoDataset(video_dirs, n_frames=FLAGS.n_frames)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=FLAGS.batch_size * FLAGS.n_shots,
            shuffle=False,
            num_workers=FLAGS.n_workers,
            prefetch_factor=4,
            drop_last=True,
        )
        with torch.no_grad():
            for _ in trange(FLAGS.n_epochs, ncols=0):

                all_tokens = np.zeros(
                    dtype='i4',
                    shape=(len(dataloader) * FLAGS.batch_size * FLAGS.n_shots, n_tasks, FLAGS.n_frames, 256)
                )
                index = 0

                for batch in tqdm(dataloader, ncols=0):
                    batch_size = batch.shape[0]
                    batch = einops.rearrange(
                        batch.numpy(), 'b t f h w c -> (b t f) c h w'
                    )
                    batch = torch.tensor(batch).to(device)
                    _, tokens = net.encode(batch)
                    tokens = einops.rearrange(
                        tokens.cpu().numpy().astype(np.int32), '(b t f) d -> b t f d', b=batch_size, t=n_tasks, f=FLAGS.n_frames
                    )
                    all_tokens[index:index + batch_size, ...] = tokens
                    index += batch_size


                random_indices = np.random.permutation(all_tokens.shape[0])
                all_tokens = all_tokens[random_indices, ...]


                tokens_sep = einops.rearrange(
                    all_tokens, '(b x) t s d -> b (x t s d)',
                    x=FLAGS.n_shots
                )
                tokens_interleave = einops.rearrange(
                    all_tokens, '(b x) t s d -> b (x s t d)',
                    x=FLAGS.n_shots
                )

                for i in range(tokens_sep.shape[0]):
                    data = {'tokens': b64encode(tokens_sep[i].tobytes()).decode('utf-8')}
                    fout.write(json.dumps(data) + '\n')

                    data = {'tokens': b64encode(tokens_interleave[i].tobytes()).decode('utf-8')}
                    fout.write(json.dumps(data) + '\n')




if __name__ == '__main__':
    mlxu.run(main)
