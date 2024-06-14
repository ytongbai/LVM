
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
from muse import VQGANModel
from utils import read_image_to_tensor


FLAGS, _ = mlxu.define_flags_with_default(
    input_image_dir='./kitti-cot_crop/sementic_seg',
    output_image_dir='./kitti-cot_crop/sementic_seg',
    output_file='./test_kitti_semantic.jsonl',
    input_filter_key='',
    output_filter_key='',
    input_suffix='',
    output_suffix='',
    crop=False,
    batch_size=1,
    n_shots=8,
    n_epochs=5,
    n_workers=8,
    dtype='fp32',
)


class PairedImageDataset(torch.utils.data.Dataset):

    def __init__(self, input_images, output_images):
        self.input_images = input_images
        self.output_images = output_images

    def __getitem__(self, index):
        try:
            return (
                read_image_to_tensor(self.input_images[index], crop=FLAGS.crop),
                read_image_to_tensor(self.output_images[index], crop=FLAGS.crop)
            )
        except Exception as e:
            print(f'Error: {e} for {self.input_images[index]}')
            return self[np.random.randint(0, len(self))]

    def __len__(self):
        return len(self.input_images)


def main(argv):
    assert FLAGS.input_image_dir != ''
    assert FLAGS.output_image_dir != ''
    assert FLAGS.output_file != ''

    # Load the pre-trained vq model from the hub
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = VQGANModel.from_pretrained('vqlm/muse/ckpts/laion').to(device)
    net.eval()


    input_images = os.listdir(FLAGS.input_image_dir)
    input_images = [i for i in input_images if i.endswith('.png') or i.endswith('.jpg') or i.endswith('.jpeg')]
    input_images = [i for i in input_images if FLAGS.input_filter_key in i]
    input_images = sorted(input_images)
    output_images = os.listdir(FLAGS.output_image_dir)
    output_images = [i for i in output_images if i.endswith('.png') or i.endswith('.jpg') or i.endswith('.jpeg')]
    output_images = [i for i in output_images if FLAGS.output_filter_key in i]
    output_images = sorted(output_images)

    assert len(input_images) == len(output_images)


    input_images = [
        os.path.join(FLAGS.input_image_dir, s)
        for s in input_images
    ]
    output_images = [
        os.path.join(FLAGS.output_image_dir, s)
        for s in output_images
    ]

    dataset = PairedImageDataset(input_images, output_images)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size * FLAGS.n_shots,
        shuffle=False,
        num_workers=FLAGS.n_workers,
        drop_last=True
    )

    total_images = len(input_images) - len(input_images) % (FLAGS.batch_size * FLAGS.n_shots)

    with torch.no_grad():
        with NamedTemporaryFile() as ntf:
            all_tokens = np.memmap(ntf, dtype='i4', mode='w+', shape=(total_images, 512))
            all_tokens[:] = 0

            index = 0
            for input_image_batch, output_image_batch in tqdm(dataloader, ncols=0):
                _, input_token_batch = net.encode(input_image_batch.permute(0,3,1,2).to(device))
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