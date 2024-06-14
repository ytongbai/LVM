import os
from copy import deepcopy
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
from muse import VQGANModel
from utils import match_mulitple_path, read_image_to_tensor


# FLAGS, _ = mlxu.define_flags_with_default(
#     input_dir='./lvm/dataset/prismer_i1k/new',
#     input_regex='/datasets/ilsvrc_2024-01-04_1601/train::./lvm/dataset/prismer_i1k/new/depth/train::./lvm/dataset/prismer_i1k/new/edge/train::./lvm/dataset/prismer_i1k/new/normal/train::./lvm/dataset/prismer_i1k/new_mapped_color/seg_coco_colored/train',
#     output_file='./lvm/other_folder/old/i1k_cot_uni-mapped_occupy_gpu.jsonl',
#     shuffle_tasks=True,
#     crop=False,
#     batch_size=16,
#     max_examples=0,
#     n_shots=3,
#     n_epochs=5,
#     n_workers=8,
#     dtype='fp32',
#     layer=2,
# )

FLAGS, _ = mlxu.define_flags_with_default(
    input_dir='./lvm/dataset/prismer_i1k/new',
    input_regex='/datasets/coco2017_2024-01-04_1601/train2017::/shared/yutongbai/labels/normal/helpers/images::/shared/yutongbai/labels/edge/helpers/images::/shared/yutongbai/labels/depth/helpers/images_coco::./lvm/dataset/prismer_coco/seg_coco_colored_mapped_color',
    output_file='./lvm/tokenized_muse/coco_mixed_uni-mapped.jsonl',
    shuffle_tasks=True,
    crop=False,
    batch_size=8,
    max_examples=0,
    n_shots=3,
    n_epochs=5,
    n_workers=8,
    dtype='fp32',
    layer=1,
)

# FLAGS, _ = mlxu.define_flags_with_default(
#     input_dir='./lvm/dataset/prismer_i1k/new',
#     input_regex='./lvm/dataset/kitti-cot_crop/image::./lvm/dataset/kitti-cot_crop/depth::./lvm/dataset/kitti-cot_crop/next_frame::./lvm/dataset/kitti-cot_crop/scene_flow::./lvm/dataset/kitti-cot_crop/sementic_seg::./lvm/dataset/kitti-cot_crop/sementic_seg_rbg::./lvm/dataset/kitti-cot_crop/stereo',
#     output_file='./lvm/tokenized_muse/kitti_new.jsonl',
#     shuffle_tasks=True,
#     crop=False,
#     batch_size=16,
#     max_examples=0,
#     n_shots=2,
#     n_epochs=5,
#     n_workers=8,
#     dtype='fp32',
#     layer=1,
# )


class MultipleImageDataset(torch.utils.data.Dataset):

    def __init__(self, input_images):
        self.input_images = input_images

    def __getitem__(self, index):
        try:
            if FLAGS.crop:
                crop_rng = np.random.default_rng(np.random.randint(0, 2 ** 32))
            else:
                crop_rng = None
            return tuple(
                read_image_to_tensor(x, crop=FLAGS.crop, crop_rng=deepcopy(crop_rng))
                for x in self.input_images[index]
            )
        except UnidentifiedImageError as e:
            print(f'Error: {e} for {self.input_images[index]}')
            return self[np.random.randint(0, len(self))]

    def __len__(self):
        return len(self.input_images)

def match_mulitple_path_v2(root, regex):
    groups = {}
    for modal in regex:
        images = glob.glob(modal + '{}.png'.format(FLAGS.layer*'/*'))
        images += glob.glob(modal + '{}.jpg'.format(FLAGS.layer*'/*'))
        images += glob.glob(modal + '{}.JPEG'.format(FLAGS.layer*'/*'))
        images += glob.glob(modal + '{}.jpeg'.format(FLAGS.layer*'/*'))
        for img in images:
            img_idx = img.split('/')[-1].split('.')[0]
            if not img_idx in groups:
                groups[img_idx] = []
            groups[img_idx].append(img)

    groups = [groups[idx] for idx in groups if len(groups[idx]) == 5]

    return groups


def main(argv):
    assert FLAGS.input_dir != ''
    assert FLAGS.input_regex != ''
    assert FLAGS.output_file != ''

    # Load the pre-trained vq model from the hub
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = VQGANModel.from_pretrained('vqlm/muse/ckpts/laion').to(device)
    net.eval()

    regex = FLAGS.input_regex.split('::')
    input_images = match_mulitple_path_v2(FLAGS.input_dir, regex)

    print(f'Found {len(input_images)} images')
    assert len(input_images) > 0, 'No images found'

    if FLAGS.max_examples > 0:
        input_images = input_images[:FLAGS.max_examples]

    random.shuffle(input_images)

    dataset = MultipleImageDataset(input_images)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size * FLAGS.n_shots,
        shuffle=False,
        num_workers=FLAGS.n_workers,
        drop_last=True
    )

    total_images = len(input_images) - len(input_images) % (FLAGS.batch_size * FLAGS.n_shots)

    with NamedTemporaryFile() as ntf:
        all_tokens = np.memmap(ntf, dtype='i4', mode='w+', shape=(total_images, 256 * len(input_images[0])))
        all_tokens[:] = 0

        index = 0
        for batch in tqdm(dataloader, ncols=0):
            k = 0
            for image in batch:
                batch_size = image.shape[0]
                image  = einops.rearrange(
                    image.numpy(), 'b h w c -> b c h w'
                )
                image  = torch.tensor(image).to(device)
                _, tokens = net.encode(image)
                tokens = einops.rearrange(
                    tokens.cpu().numpy().astype(np.int32), '(b t) d -> b (t d)', b=batch_size
                )
                all_tokens[index:index + image.shape[0], k:k + 256] = tokens
                k += 256
            index += batch[0].shape[0]

        with open(FLAGS.output_file, 'w') as fout:
            for _ in trange(FLAGS.n_epochs, ncols=0):
                indices = np.random.permutation(total_images).reshape(-1, FLAGS.n_shots)
                for i in trange(indices.shape[0], ncols=0):
                    tokens = deepcopy(all_tokens[indices[i], :])
                    tokens = einops.rearrange(tokens, 'b (s t) -> b s t', t=256)
                    if FLAGS.shuffle_tasks:
                        permutations = np.random.permutation(tokens.shape[1])
                        tokens = tokens[:, permutations, :]
                    tokens = tokens.reshape(-1)
                    data = {'tokens': b64encode(tokens.tobytes()).decode('utf-8'),}
                    fout.write(json.dumps(data) + '\n')


if __name__ == '__main__':
    mlxu.run(main)