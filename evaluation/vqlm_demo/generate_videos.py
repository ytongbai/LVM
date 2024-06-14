
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
    output_dir='',
    center_crop=1.0,
    n_context_frames=12,
    n_new_frames=4,
    n_candidates=8,
    temperature=1.0,
    top_p=1.0,
    n_workers=8,
    stride=8,
    batch_size=32,
    torch_devices='',
    shuffle=False,
    max_examples=0,
)


def save_image(args):
    image, filename = args
    base = FLAGS.input_files.split('*')[0]
    filename = filename[len(base):].replace('/', '_') + '.png'
    Image.fromarray(image).save(os.path.join(FLAGS.output_dir, filename))


class VideoDataset(torch.utils.data.Dataset):

    def __init__(self, videos, frame_input=False, n_frames=8, stride=1):
        self.videos = videos
        self.frame_input = frame_input
        self.n_frames = n_frames
        self.stride = stride

    def __getitem__(self, index):
        if self.frame_input:
            frames = read_frames_from_dir(
                self.videos[index], self.n_frames, self.stride,
                center_crop=FLAGS.center_crop,
            )
        else:
            frames = read_frames_from_video(
                self.videos[index], self.n_frames, self.stride,
                center_crop=FLAGS.center_crop,
            )
        if frames is None:
            return self[np.random.randint(0, len(self))]
        return frames, self.videos[index]

    def __len__(self):
        return len(self.videos)



def main(_):
    assert FLAGS.checkpoint != '' and FLAGS.output_dir != ''
    assert FLAGS.read_file_list != '' or FLAGS.input_files != ''
    os.makedirs(FLAGS.output_dir, exist_ok=True)

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
        n_frames=FLAGS.n_context_frames,
        stride=FLAGS.stride
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        num_workers=FLAGS.n_workers,
        prefetch_factor=4,
        drop_last=True,
    )

    if FLAGS.torch_devices == '':
        torch_devices = None
    else:
        torch_devices = [f'cuda:{x}' for x in FLAGS.torch_devices.split(',')]

    model = MultiProcessInferenceModel(
        checkpoint=FLAGS.checkpoint, torch_devices=torch_devices,
    )

    save_img_pool = Pool(FLAGS.n_workers)



    for batch, filenames in tqdm(dataloader, ncols=0):
        
        
        
        batch = batch.numpy()



        generated = model(
            batch,
            n_new_frames=FLAGS.n_new_frames,
            n_candidates=FLAGS.n_candidates,
            temperature=FLAGS.temperature,
            top_p=FLAGS.top_p,
        )


        generated = np.array(generated)




        output_batch = einops.repeat(
            batch,
            'b s h w c -> b n s h w c',
            n=FLAGS.n_candidates,
        )


        combined = einops.rearrange(
            np.concatenate([output_batch, generated], axis=2),
            'b n s h w c -> b (n h) (s w) c'
        )

        
        combined = (np.clip(combined, 0, 1) * 255).astype(np.uint8)
        save_img_pool.imap(save_image, zip(combined, filenames))


if __name__ == '__main__':
    mlxu.run(main)