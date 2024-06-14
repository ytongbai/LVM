import os
from multiprocessing import Pool
import numpy as np
import random
from PIL import Image
import re
import cv2
import glob
from natsort import natsorted


class MultiProcessImageSaver(object):

    def __init__(self, n_workers=1):
        self.pool = Pool(n_workers)

    def __call__(self, images, output_files, resizes=None):
        if resizes is None:
            resizes = [None for _ in range(len(images))]
        return self.pool.imap(
            self.save_image,
            zip(images, output_files, resizes),
        )

    def close(self):
        self.pool.close()
        self.pool.join()

    @staticmethod
    def save_image(args):
        image, filename, resize = args
        image = Image.fromarray(image)
        if resize is not None:
            image = image.resize(tuple(resize))
        image.save(filename)


def list_dir_with_full_path(path):
    return [os.path.join(path, f) for f in os.listdir(path)]


def find_all_files_in_dir(path):
    files = []
    for root, _, files in os.walk(path):
        for file in files:
            files.append(os.path.join(root, file))
    return files


def is_image(path):
    return (
        path.endswith('.jpg')
        or path.endswith('.png')
        or path.endswith('.jpeg')
        or path.endswith('.JPG')
        or path.endswith('.PNG')
        or path.endswith('.JPEG')
    )


def is_video(path):
    return (
        path.endswith('.mp4')
        or path.endswith('.avi')
        or path.endswith('.MP4')
        or path.endswith('.AVI')
        or path.endswith('.webm')
        or path.endswith('.WEBM')
        or path.endswith('.mkv')
        or path.endswith('.MVK')
    )


def random_square_crop(img, random_generator=None):
    # If no random generator is provided, use numpy's default
    if random_generator is None:
        random_generator = np.random.default_rng()

    # Get the width and height of the image
    width, height = img.size

    # Determine the shorter side
    min_size = min(width, height)

    # Randomly determine the starting x and y coordinates for the crop
    if width > height:
        left = random_generator.integers(0, width - min_size)
        upper = 0
    else:
        left = 0
        upper = random_generator.integers(0, height - min_size)

    # Calculate the ending x and y coordinates for the crop
    right = left + min_size
    lower = upper + min_size

    # Crop the image
    return img.crop((left, upper, right, lower))


def read_image_to_tensor(path, center_crop=1.0):
    pil_im = Image.open(path).convert('RGB')
    if center_crop < 1.0:
        width, height = pil_im.size
        pil_im = pil_im.crop((
            int((1 - center_crop) * height / 2), int((1 + center_crop) * height / 2),
            int((1 - center_crop) * width / 2), int((1 + center_crop) * width / 2),
        ))
    input_img = pil_im.resize((256, 256))
    input_img = np.array(input_img) / 255.0
    input_img = input_img.astype(np.float32)
    return input_img


def match_mulitple_path(root_dir, regex):
    videos = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            videos.append(os.path.join(root, file))

    videos = [v for v in videos if not v.split('/')[-1].startswith('.')]

    grouped_path = {}
    for r in regex:
        r = re.compile(r)
        for v in videos:
            matched = r.findall(v)
            if len(matched) > 0:
                groups = matched[0]
                if groups not in grouped_path:
                    grouped_path[groups] = []
                grouped_path[groups].append(v)

    grouped_path = {
        k: tuple(v) for k, v in grouped_path.items()
        if len(v) == len(regex)
    }
    return list(grouped_path.values())


def randomly_subsample_frame_indices(length, n_frames, max_stride=30, random_start=True):
    assert length >= n_frames
    max_stride = min(
        (length - 1) // (n_frames - 1),
        max_stride
    )
    stride = np.random.randint(1, max_stride + 1)
    if random_start:
        start = np.random.randint(0, length - (n_frames - 1) * stride)
    else:
        start = 0
    return np.arange(n_frames) * stride + start


def read_frames_from_dir(dir_path, n_frames, stride, random_start=True, center_crop=1.0):
    files = [os.path.join(dir_path, x) for x in os.listdir(dir_path)]
    files = natsorted([x for x in files if is_image(x)])

    total_frames = len(files)

    if total_frames < n_frames:
        return None

    max_stride = (total_frames - 1) // (n_frames - 1)
    stride = min(max_stride, stride)

    if random_start:
        start = np.random.randint(0, total_frames - (n_frames - 1) * stride)
    else:
        start = 0
    frame_indices = np.arange(n_frames) * stride + start

    frames = []
    for frame_index in sorted(frame_indices):
        # Check if the frame_index is valid
        frames.append(read_image_to_tensor(files[frame_index], center_crop=center_crop))
    if len(frames) < n_frames:
        return None
    frames = np.stack(frames, axis=0)
    return frames


def read_frames_from_video(video_path, n_frames, stride, random_start=True, center_crop=1.0):

    frames = []
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < n_frames:
        cap.release()
        return None

    max_stride = (total_frames - 1) // (n_frames - 1)
    stride = min(max_stride, stride)

    if random_start:
        start = np.random.randint(0, total_frames - (n_frames - 1) * stride)
    else:
        start = 0
    frame_indices = np.arange(n_frames) * stride + start

    for frame_index in sorted(frame_indices):
        # Check if the frame_index is valid
        if 0 <= frame_index < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                if center_crop < 1.0:
                    height, width, _ = frame.shape
                    frame = frame[
                        int((1 - center_crop) * height / 2):int((1 + center_crop) * height / 2),
                        int((1 - center_crop) * width / 2):int((1 + center_crop) * width / 2),
                        :
                    ]
                frame = cv2.resize(frame, (256, 256))

                frames.append(frame)

        else:
            print(f"Frame index {frame_index} is out of bounds. Skipping...")

    cap.release()
    if len(frames) < n_frames:
        return None
    frames = np.stack(frames, axis=0).astype(np.float32) / 255.0

    # From BGR to RGB
    return np.stack(
        [frames[..., 2], frames[..., 1], frames[..., 0]], axis=-1
    )


def read_all_frames_from_video(video_path, center_crop=1.0):

    frames = []
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    for frame_index in range(total_frames):
        # Check if the frame_index is valid
        if 0 <= frame_index < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                if center_crop < 1.0:
                    height, width, _ = frame.shape
                    frame = frame[
                        int((1 - center_crop) * height / 2):int((1 + center_crop) * height / 2),
                        int((1 - center_crop) * width / 2):int((1 + center_crop) * width / 2),
                        :
                    ]
                frames.append(cv2.resize(frame, (256, 256)))
        else:
            print(f"Frame index {frame_index} is out of bounds. Skipping...")

    cap.release()
    if len(frames) == 0:
        return None
    frames = np.stack(frames, axis=0).astype(np.float32) / 255.0
    # From BGR to RGB
    return np.stack(
        [frames[..., 2], frames[..., 1], frames[..., 0]], axis=-1
    )


def read_max_span_frames_from_video(video_path, n_frames):
    frames = []
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < n_frames:
        cap.release()
        return None
    stride = (total_frames - 1) // (n_frames - 1)
    frame_indices = np.arange(n_frames) * stride

    frames = []
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.resize(frame, (256, 256)))

    cap.release()
    if len(frames) < n_frames:
        return None

    frames = np.stack(frames, axis=0).astype(np.float32) / 255.0
    # From BGR to RGB
    return np.stack(
        [frames[..., 2], frames[..., 1], frames[..., 0]], axis=-1
    )

