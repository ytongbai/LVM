"""
Evaluating the perplexity on few shot tasks. This script accept a jsonl file
as input. Each line of the jsonl file representing a dictionary. Each line
represents one example in the evaluation set. The dictionary should have two key:

    input: a list of paths to the input images as context to the model. This
        list should include the few shot examples.
    target: a list of paths to the target images to evaluate perplexity

Ths script should run the model and compute the average perplexity on the
evaluation set.
"""

import os
import json
from PIL import Image
import numpy as np
import mlxu
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from .inference import MultiProcessInferenceModel


FLAGS, _ = mlxu.define_flags_with_default(
    input_file='',
    checkpoint='',
    input_base_dir='',
    batch_size=2,
    json_input_key='input',
    json_target_key='target',
    dtype='float16',
    torch_devices='',
    n_workers=4,
    max_examples=0,
)


def read_image_to_tensor(path):
    pil_im = Image.open(path).convert('RGB')
    input_img = pil_im.resize((256, 256))
    input_img = np.array(input_img) / 255.0
    input_img = input_img.astype(np.float32)
    return input_img


class MultiFrameDataset(torch.utils.data.Dataset):
    def __init__(self, input_files, target_files):
        assert len(input_files) == len(target_files)
        self.input_files = input_files
        self.target_files = target_files

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_list = np.stack(
            [read_image_to_tensor(f) for f in self.input_files[idx]],
            axis=0
        )
        target_list = np.stack(
            [read_image_to_tensor(f) for f in self.target_files[idx]],
            axis=0
        )
        return input_list, target_list


def main(_):
    assert FLAGS.checkpoint != ''

    print(f'Loading checkpoint from {FLAGS.checkpoint}')
    print(f'Evaluating input file from {FLAGS.input_file}')

    model = MultiProcessInferenceModel(
        checkpoint=FLAGS.checkpoint,
        torch_devices=FLAGS.torch_devices,
        dtype=FLAGS.dtype,
        use_lock=True,
        perplexity_batch_size=FLAGS.batch_size,
    )

    input_files = []
    target_files = []

    with mlxu.open_file(FLAGS.input_file, 'r') as f:
        for line in f:
            record = json.loads(line)
            input_files.append(record[FLAGS.json_input_key])
            target_files.append(record[FLAGS.json_target_key])

    if FLAGS.input_base_dir != '':
        input_files = [
            [os.path.join(FLAGS.input_base_dir, x) for x in y]
            for y in input_files
        ]
        target_files = [
            [os.path.join(FLAGS.input_base_dir, x) for x in y]
            for y in target_files
        ]

    if FLAGS.max_examples > 0:
        input_files = input_files[:FLAGS.max_examples]
        target_files = target_files[:FLAGS.max_examples]

    dataset = MultiFrameDataset(input_files, target_files)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size * model.n_processes,
        shuffle=False,
        num_workers=FLAGS.n_workers
    )

    perplexities = []

    for input_images, target_images in tqdm(data_loader, ncols=0):
        perplexity = model.compute_perplexity(input_images, target_images)
        perplexities.append(perplexity)

    perplexities = np.concatenate(perplexities, axis=0)
    print(f'Perplexity: {np.mean(perplexities)}')


if __name__ == "__main__":
    mlxu.run(main)