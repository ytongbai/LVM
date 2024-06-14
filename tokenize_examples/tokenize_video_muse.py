from base64 import b64encode
from tqdm import tqdm, trange
import numpy as np
np.float = np.float64
np.int = np.int_
from utils import read_frames_from_video, is_video

import einops
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
from muse import VQGANModel
from base64 import b64encode
import json
import os
import mlxu



FLAGS, _ = mlxu.define_flags_with_default(
    input_dir='DAVIS/JPEGImages/480p',
    output_file='vqlm/muse/running_script/tokenized_muse/davis.jsonl',
    batch_size=32,
    n_frames=16,
    n_workers=32,
    strides='8',
    n_epochs=1,
    dtype='fp32',
)


class VideoDataset(torch.utils.data.Dataset):

    def __init__(self, videos, n_frames=8, stride=1):
        self.videos = videos
        self.n_frames = n_frames
        self.stride = stride

    def __getitem__(self, index):
        frames = read_frames_from_video(self.videos[index], self.n_frames, self.stride)
        if frames is None:
            return self[np.random.randint(0, len(self))]
        return frames

    def __len__(self):
        return len(self.videos)


def main(argv):
    assert FLAGS.input_dir != ''
    assert FLAGS.output_file != ''

    # Load the pre-trained vq model from the hub
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = VQGANModel.from_pretrained('vqlm/muse/ckpts/laion').to(device)
    net.eval()

    videos = []
    for root, _, files in os.walk(FLAGS.input_dir):
        for file in files:
            if is_video(file):
                videos.append(os.path.join(root, file))

    with open(FLAGS.output_file, 'w') as fout:
        with torch.no_grad():
            for epoch in trange(FLAGS.n_epochs, ncols=0):
                for stride in tqdm(FLAGS.strides.split(','), ncols=0):
                    stride = int(stride)
                    dataset = VideoDataset(videos, n_frames=FLAGS.n_frames, stride=stride)
                    dataloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=FLAGS.batch_size,
                        shuffle=False,
                        num_workers=FLAGS.n_workers,
                        prefetch_factor=4,
                        drop_last=True,
                    )
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
                            data = {'tokens': b64encode(tokens[i].tobytes()).decode('utf-8'),}
                            fout.write(json.dumps(data) + '\n')



if __name__ == '__main__':
    mlxu.run(main)
