
import os
from functools import partial
from tempfile import NamedTemporaryFile
import random
import json
from base64 import b64encode
from tqdm import tqdm, trange
import glob
import numpy as np
import mlxu

import torch

import jax
import jax.numpy as jnp
import flax
import einops

from PIL import Image, UnidentifiedImageError

from PIL import Image
from muse import VQGANModel
from utils import read_image_to_tensor, is_image, list_dir_with_full_path


FLAGS, _ = mlxu.define_flags_with_default(
    input_image_dir='/datasets/ilsvrc_2024-01-04_1601/train',
    output_file='./lvm/tokenized_muse/i1k_colorization.jsonl',
    batch_size=1,
    n_shots=8,
    n_epochs=2,
    n_workers=8,
    patch_size=32,
    hole_mask_ratio=0.3,
    dtype='fp32',
    layer = 1
)


class PairedImageDataset(torch.utils.data.Dataset):

    def __init__(self, images):
        self.images = images

    def __getitem__(self, index):
        try:
            original_image = read_image_to_tensor(self.images[index])
        except UnidentifiedImageError:
            original_image = np.zeros((256, 256, 3), dtype=np.float32)

        # make gray images
        grayscale = np.dot(original_image[..., :3], [0.2989, 0.5870, 0.1140])
        gray_image = np.stack((grayscale,) * 3, axis=-1)
        # Stack the grayscale image across the third dimension
        gray_image = np.array(gray_image, dtype=np.float32)
        return gray_image, original_image

    def __len__(self):
        return len(self.images)


def main(argv):
    assert FLAGS.input_image_dir != ''
    assert FLAGS.output_file != ''

    # Load the pre-trained vq model from the hub
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = VQGANModel.from_pretrained('vqlm/muse/ckpts/laion').to(device)
    net.eval()


    input_images = glob.glob('{}{}/*.png'.format(FLAGS.input_image_dir, '/*'*FLAGS.layer))
    input_images += glob.glob('{}{}/*.jpg'.format(FLAGS.input_image_dir, '/*' * FLAGS.layer))
    input_images += glob.glob('{}{}/*.jpeg'.format(FLAGS.input_image_dir, '/*' * FLAGS.layer))
    input_images += glob.glob('{}{}/*.JPEG'.format(FLAGS.input_image_dir, '/*' * FLAGS.layer))

    dataset = PairedImageDataset(input_images)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size * FLAGS.n_shots,
        shuffle=False,
        num_workers=FLAGS.n_workers,
        drop_last=True
    )

    total_images = len(input_images) - len(input_images) % (FLAGS.batch_size * FLAGS.n_shots)
    print(total_images)
    with NamedTemporaryFile() as ntf:
        all_tokens = np.memmap(ntf, dtype='i4', mode='w+', shape=(total_images, 512))
        all_tokens[:] = 0

        index = 0
        # debug_count = 0
        for input_image_batch, output_image_batch in tqdm(dataloader, ncols=0):
            # if debug_count < 5243:
            #     debug_count += 1
            #     continue

            _, input_token_batch = net.encode(input_image_batch.permute(0, 3, 1, 2).to(device))
            _, output_token_batch = net.encode(output_image_batch.permute(0, 3, 1, 2).to(device))

            all_tokens[index:index + input_image_batch.shape[0]] = np.concatenate(
                [input_token_batch.cpu().numpy().astype(np.int32), output_token_batch.cpu().numpy().astype(np.int32)],
                axis=1
            )
            index += input_image_batch.shape[0]

        with open(FLAGS.output_file, 'w') as fout:
            for _ in trange(FLAGS.n_epochs, ncols=0):
                indices = np.random.permutation(total_images).reshape(-1, FLAGS.n_shots)
                for i in trange(indices.shape[0], ncols=0):
                    tokens = all_tokens[indices[i], :].reshape(-1)
                    data = {'tokens': b64encode(tokens.tobytes()).decode('utf-8'),}
                    fout.write(json.dumps(data) + '\n')


if __name__ == '__main__':
    mlxu.run(main)