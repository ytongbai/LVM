
import os
import glob
from functools import partial
from tempfile import NamedTemporaryFile
import random
import json
from base64 import b64encode
from tqdm import tqdm, trange
from muse import VQGANModel

import numpy as np
np.float = np.float64
np.int = np.int_
import mlxu

import torch


import einops

from PIL import Image

from utils import (
    list_dir_with_full_path, is_image, read_image_to_tensor,
    randomly_subsample_frame_indices
)




FLAGS, _ = mlxu.define_flags_with_default(
    input_dirs='',
    output_file='',
    batch_size=1,
    n_frames=16,
    n_epochs=1,
    n_workers=8,
    max_stride=4,
    dtype='fp32',
)


class VideoDataset(torch.utils.data.Dataset):

    def __init__(self, videos, n_frames=8):
        self.videos = videos
        self.n_frames = n_frames

    def __getitem__(self, index):
        frames = []
        for file in sorted(list_dir_with_full_path(self.videos[index])):
            if is_image(file):
                frames.append(read_image_to_tensor(file))
        if len(frames) < self.n_frames:
            return self[np.random.randint(0, len(self))]
        indices = randomly_subsample_frame_indices(
            len(frames), self.n_frames, FLAGS.max_stride,
            random_start=True
        )
        frames = np.stack([frames[i] for i in indices], axis=0)
        return frames

    def __len__(self):
        return len(self.videos)


def main(argv):
    assert FLAGS.input_dirs != ''
    assert FLAGS.output_file != ''

    # Load the pre-trained vq model from the hub
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = VQGANModel.from_pretrained('vqlm/muse/ckpts/laion').to(device)
    net.eval()

    # videos = list_dir_with_full_path(FLAGS.input_dir)
    videos = glob.glob(FLAGS.input_dirs)

    with torch.no_grad():
        with open(FLAGS.output_file, 'w') as fout:
            dataset = VideoDataset(videos, n_frames=FLAGS.n_frames)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=FLAGS.batch_size,
                shuffle=False,
                num_workers=FLAGS.n_workers,
                prefetch_factor=4,
                drop_last=True,
            )
            for _ in range(FLAGS.n_epochs):
                for batch in tqdm(dataloader, ncols=0):
                    batch_size = batch.shape[0]
                    batch = einops.rearrange(
                        batch.numpy(), 'b t h w c -> (b t) c h w'
                    )
                    batch = torch.tensor(batch).to(device)
                    _, tokens = net.encode(batch)
                    tokens = einops.rearrange(
                        tokens.cpu().numpy().astype(np.int32), '(b t) d -> b (t d)', b=batch_size
                    )
                    for i in range(batch_size):
                        data = {'tokens': b64encode(tokens[i].tobytes()).decode('utf-8')}
                        fout.write(json.dumps(data) + '\n')



if __name__ == '__main__':
    mlxu.run(main)
