""" Tokenize multiple related sequences of data """



import os
from copy import deepcopy
from functools import partial
from tempfile import NamedTemporaryFile
import random
import json
from tqdm import tqdm, trange
from muse import VQGANModel
import numpy as np
import mlxu

import torch


import einops

from PIL import Image

from utils import match_mulitple_path

from utils import (
    randomly_subsample_frame_indices, list_dir_with_full_path,
    is_image, b64encode_tokens, read_image_to_tensor
)


FLAGS, _ = mlxu.define_flags_with_default(
    input_dir='',
    output_file='',
    batch_size=4,
    n_frames=4,
    max_stride=4,
    n_shots=4,
    n_epochs=1,
    n_workers=16,
    dtype='fp32',
)



class Co3DDataset(torch.utils.data.Dataset):

    def __init__(self, image_dirs, n_frames):
        self.image_dirs = image_dirs
        self.n_frames = n_frames

    def __getitem__(self, index):
        tasks = self.image_dirs[index]
        frames = []
        length = len(list_dir_with_full_path(tasks[0]))
        indices = randomly_subsample_frame_indices(
            length, self.n_frames, FLAGS.max_stride
        )
        for task in tasks:
            task_frames = []
            files = sorted(list_dir_with_full_path(task))
            for i in indices:
                if is_image(files[i]):
                    task_frames.append(read_image_to_tensor(files[i]))
            frames.append(np.stack(task_frames, axis=0))

        return np.stack(frames, axis=0)

    def __len__(self):
        return len(self.image_dirs)


def main(argv):
    assert FLAGS.input_dir != ''
    assert FLAGS.output_file != ''

    # Load the pre-trained vq model from the hub
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = VQGANModel.from_pretrained('/home/vqlm/muse/ckpts/laion').to(device)
    net.eval()

    dirs = []
    for d in list_dir_with_full_path(FLAGS.input_dir):
        if not os.path.isdir(d):
            continue
        for d2 in list_dir_with_full_path(d):
            if not os.path.isdir(d2):
                continue
            dirs.append(d2)

    image_dirs = []
    for d in dirs:
        image_dirs.append((
            os.path.join(d, 'images'),
            os.path.join(d, 'masks'),
            os.path.join(d, 'depth_masks')
        ))

    with open(FLAGS.output_file, 'w') as fout:
        with torch.no_grad():
            for _ in trange(FLAGS.n_epochs, ncols=0):
                print(image_dirs[0])
                dataset = Co3DDataset(image_dirs, FLAGS.n_frames)
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=FLAGS.batch_size * FLAGS.n_shots,
                    shuffle=False,
                    num_workers=FLAGS.n_workers,
                    drop_last=True
                )

                for batch in tqdm(dataloader, ncols=0):
                    batch_shape = batch.shape[:-3]
                    batch = batch.reshape(-1, *batch.shape[-3:])
                    batch = batch.permute(0,3,1,2)
                    batch = batch.to(device)

                    _, tokens = net.encode(batch)
                    tokens = tokens.reshape(*batch_shape, tokens.shape[-1])
                    # batch x task x frame x token
                    tokens = einops.rearrange(
                        tokens.cpu().numpy().astype(np.int32), '(b s) t f d -> b s t f d',
                        s=FLAGS.n_shots
                    )

                    image_mask_tokens = np.concatenate(
                        (tokens[:, :, 0, :, :], tokens[:, :, 1, :, :]), axis=-2
                    )
                    image_depth_tokens = np.concatenate(
                        (tokens[:, :, 0, :, :], tokens[:, :, 2, :, :]), axis=-2
                    )
                    for i in range(tokens.shape[0]):
                        data = {'tokens': b64encode_tokens(image_mask_tokens[i])}
                        fout.write(json.dumps(data) + '\n')
                        data = {'tokens': b64encode_tokens(image_depth_tokens[i])}
                        fout.write(json.dumps(data) + '\n')







if __name__ == '__main__':
    mlxu.run(main)