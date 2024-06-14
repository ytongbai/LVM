import os
from functools import partial
from tempfile import NamedTemporaryFile
import random
import json
from base64 import b64encode
from tqdm import tqdm, trange

import numpy as np
import mlxu

import torch

import jax
import jax.numpy as jnp
import flax
import einops

from PIL import Image
from utils import read_image_to_tensor

FLAGS, _ = mlxu.define_flags_with_default(
    input_image_dir='/datasets/ilsvrc_2024-01-04_1601/train',
    output_file='/home/yutongbai/vqlm/muse/running_script/tokenized_muse/i1kcate_4.jsonl',
    images_per_shot=4,
    n_shots=4,
    n_epochs=1,
    n_workers=8,
    dtype='fp32',
    batch_size=1
)

# Define the desired fixed token length
fixed_token_length = 4096


class SubfolderImageDataset(torch.utils.data.Dataset):

    def __init__(self, subfolders, images_per_shot):
        self.subfolders = subfolders
        self.images_per_shot = images_per_shot

    def __getitem__(self, index):
        subfolder = self.subfolders[index]
        image_files = [os.path.join(subfolder, f) for f in os.listdir(subfolder) if
                       f.endswith(('.png', '.jpg', '.JPEG'))]
        selected_images = np.random.choice(image_files, self.images_per_shot, replace=False)
        return [read_image_to_tensor(image_file) for image_file in selected_images]

    def __len__(self):
        return len(self.subfolders)


def read_image_to_tensor(image_path):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        img = img.resize((256, 256))
        img = np.array(img) / 255.0
        return torch.tensor(img).float()


def custom_collate(batch):
    return [item for sublist in batch for item in sublist]


def main(argv):
    assert FLAGS.input_image_dir != ''
    assert FLAGS.output_file != ''

    subfolders = [os.path.join(FLAGS.input_image_dir, d) for d in os.listdir(FLAGS.input_image_dir) if
                  os.path.isdir(os.path.join(FLAGS.input_image_dir, d))]

    dataset = SubfolderImageDataset(subfolders, FLAGS.images_per_shot)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.n_workers,
        drop_last=True,
        collate_fn=custom_collate
    )

    total_images = len(dataset) * FLAGS.images_per_shot

    checkpoint_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '/home/vqlm/eval_visualization/torch_ckpts/jax_xh_ckpt.pkl'
    )
    tokenize_fn = get_tokenizer_fn(checkpoint_path, FLAGS.dtype)

    n_devices = jax.device_count()

    with NamedTemporaryFile() as ntf:
        # Adjust the shape of all_tokens to match the fixed token length
        all_tokens = np.memmap(ntf, dtype='i4', mode='w+', shape=(total_images, fixed_token_length))

        index = 0
        for batch in tqdm(dataloader, ncols=0):
            batch_images = np.concatenate(batch, axis=0)

            # Reshape batch_images to (n_devices, batch_size/n_devices, height, width, channels)
            batch_images = batch_images.reshape(n_devices, -1, 256, 256, 3)

            tokens = tokenize_fn(batch_images).flatten()

            # Ensure tokens have a fixed length (fixed_token_length)
            tokens = tokens[:fixed_token_length]

            # Assign tokens to the all_tokens array
            all_tokens[index:index + len(tokens)] = tokens
            index += len(tokens)

        with open(FLAGS.output_file, 'w') as fout:
            for _ in trange(FLAGS.n_epochs, ncols=0):
                indices = np.random.permutation(total_images).reshape(-1, FLAGS.n_shots * FLAGS.images_per_shot)
                for i in trange(indices.shape[0], ncols=0):
                    tokens = all_tokens[indices[i], :].reshape(-1)
                    data = {'tokens': b64encode(tokens.tobytes()).decode('utf-8')}
                    fout.write(json.dumps(data) + '\n')


if __name__ == '__main__':
    mlxu.run(main)
