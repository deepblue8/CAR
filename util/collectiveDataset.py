import torch
from torch.utils import data
import torchvision.models as models
import torchvision.transforms as transforms
import cv2 as cv
import random
from PIL import Image
import numpy as np

from collections import Counter
from skimage.util import random_noise  # 添加噪声模块
from .data_augmentation import random_crop, random_flip, random_rotation, random_color

FRAMES_NUM = {1: 302, 2: 347, 3: 194, 4: 257, 5: 536, 6: 401, 7: 968, 8: 221, 9: 356, 10: 302, 11: 1813, 12: 1084,
              13: 851, 14: 723, 15: 464, 16: 1021, 17: 905, 18: 600, 19: 203, 20: 342, 21: 650, 22: 361, 23: 311,
              24: 321, 25: 617, 26: 734, 27: 1804, 28: 470, 29: 635, 30: 356, 31: 690, 32: 194, 33: 193, 34: 395,
              35: 707, 36: 914, 37: 1049, 38: 653, 39: 518, 40: 401, 41: 707, 42: 420, 43: 410, 44: 356, 45: 151,
              46: 174, 47: 218, 48: 47, 49: 223, 50: 365, 51: 362, 52: 781, 53: 401, 54: 486, 55: 695, 56: 462,
              57: 443, 58: 629, 59: 899, 60: 550, 61: 373, 62: 200, 63: 433, 64: 319, 65: 443, 66: 315, 67: 391,
              68: 945, 69: 1011, 70: 449, 71: 351, 72: 751, 73: 325, 74: 400}

FRAMES_SIZE = {1: (480, 720), 2: (480, 720), 3: (480, 720), 4: (480, 720), 5: (480, 720), 6: (480, 720), 7: (480, 720),
               8: (480, 720), 9: (480, 720), 10: (480, 720), 11: (480, 720), 12: (480, 720), 13: (480, 720),
               14: (480, 720),
               15: (450, 800), 16: (480, 720), 17: (480, 720), 18: (480, 720), 19: (480, 720), 20: (450, 800),
               21: (450, 800),
               22: (450, 800), 23: (450, 800), 24: (450, 800), 25: (480, 720), 26: (480, 720), 27: (480, 720),
               28: (480, 720),
               29: (480, 720), 30: (480, 720), 31: (480, 720), 32: (480, 720), 33: (480, 720), 34: (480, 720),
               35: (480, 720),
               36: (480, 720), 37: (480, 720), 38: (480, 720), 39: (480, 720), 40: (480, 720), 41: (480, 720),
               42: (480, 720),
               43: (480, 720), 44: (480, 720), 45: (480, 640), 46: (480, 640), 47: (480, 640), 48: (480, 640),
               49: (480, 640),
               50: (480, 640), 51: (480, 640), 52: (480, 640), 53: (480, 640), 54: (480, 640), 55: (480, 640),
               56: (480, 640),
               57: (480, 640), 58: (480, 640), 59: (480, 640), 60: (720, 1280), 61: (720, 1280), 62: (720, 1280),
               63: (720, 1280),
               64: (720, 1280), 65: (720, 1280), 66: (720, 1280), 67: (720, 1280), 68: (720, 1280), 69: (720, 1280),
               70: (720, 1280),
               71: (720, 1280), 72: (720, 1280), 73: (720, 1280), 74: (720, 1280)}

ACTIONS = ['NA', 'Crossing', 'Waiting', 'Queueing', 'Walking', 'Talking']
ACTIVITIES = ['Crossing', 'Waiting', 'Queueing', 'Walking', 'Talking']

ACTIONS_ID = {a: i for i, a in enumerate(ACTIONS)}
ACTIVITIES_ID = {a: i for i, a in enumerate(ACTIVITIES)}

def collective_read_annotations(path, sid):
    annotations = {}
    path = path + '/seq%02d/annotations.txt' % sid
    with open(path, mode='r') as f:
        frame_id = None  # 用来记录那一帧
        group_activity = None
        actions = []
        bboxes = []
        for line in f.readlines():
            values = line[:-1].split('	')
            if int(values[0]) != frame_id:
                if frame_id is not None:
                    counter = Counter(actions).most_common(2)
                    group_activity = counter[0][0] - 1 if counter[0][0] != 0 else counter[1][0] - 1
                    annotations[frame_id] = {
                        'frame_id': frame_id,
                        'group_activity': group_activity,
                        'actions': actions,
                        'bboxes': bboxes
                    }

                frame_id = int(values[0])
                group_activity = None
                actions = []
                bboxes = []

            actions.append(int(values[5]) - 1)  # 群体行为
            x, y, w, h = (int(values[i]) for i in range(1, 5))  # 边界框
            H, W = FRAMES_SIZE[sid]  # 高宽
            bboxes.append((y / H, x / W, (y + h) / H, (x + w) / W))

    return annotations

def collective_read_dataset(path, seqs):
    data = {}
    for sid in seqs:
        data[sid] = collective_read_annotations(path, sid)
        break
    return data

def collective_all_frames(anns):
    return [(s, f) for s in anns for f in anns[s]]

class CollectiveDatasetAll(data.Dataset):
    def __init__(self, anns, frames, images_path, image_size, feature_size, num_boxes=13, num_frames=10,
                 is_training=True, is_finetune=False):
        self.anns = anns
        self.frames = frames
        self.images_path = images_path
        self.image_size = image_size
        self.feature_size = feature_size

        self.num_boxes = num_boxes
        self.num_frames = num_frames

        self.is_training = is_training
        self.is_finetune = is_finetune
        print(len(self.frames))

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, i):
        select_frames_all = self.get_all_frames(self.frames[i])
        allSample = self.load_all_sequence(i, select_frames_all)

        return allSample

    def get_all_frames(self, frame):

        sid, src_fid = frame

        self.is_finetune = False
        sample_frames = range(1, 1 + len(self.frames))
        sample_frames = sorted(sample_frames)
        return [(sid, src_fid, fid) for fid in sample_frames]

    def load_all_sequence(self, index, select_frames_all):
        OH, OW = self.feature_size
        images, bboxes = [], []
        activities, actions = [], []
        bboxes_num = []
        frames = []

        for i, (sid, src_fid, fid) in enumerate(select_frames_all):

            frames.append((sid, src_fid, fid))
            img = Image.open(self.images_path + '/seq%02d/frame%04d.jpg' % (sid, fid))
            img = transforms.functional.resize(img, self.image_size)
            img = np.array(img)

            if self.is_training and src_fid > self.__len__() / 2:
                H, W = FRAMES_SIZE[sid]

                temp_boxes = []
                for box in self.anns[sid][src_fid]['bboxes']:
                    y1, x1, y2, x2 = box
                    w1, h1, w2, h2 = x1 * W, y1 * H, x2 * W, y2 * H
                    temp_boxes.append([w1, h1, w2, h2])
                boxes = temp_boxes
                boxes = np.array(boxes)

                labels = self.anns[sid][src_fid]['actions'][:]
                labels = np.array(labels, dtype=np.int32)

                img, boxes, labels = random_flip(img, boxes, labels)
                img, boxes, labels = random_rotation(img, boxes, labels)
                temp_actions = labels.tolist()

                img = random_color(img)

                temp_boxes = []
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    w1, h1, w2, h2 = x1 / W * OW, y1 / H * OH, x2 / W * OW, y2 / H * OH
                    temp_boxes.append((w1, h1, w2, h2))

            else:
                temp_boxes = []
                for box in self.anns[sid][src_fid]['bboxes']:
                    y1, x1, y2, x2 = box
                    w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
                    temp_boxes.append((w1, h1, w2, h2))

                temp_actions = self.anns[sid][src_fid]['actions'][:]

            img = img.transpose(2, 0, 1)
            images.append(img)
            bboxes_num.append(len(temp_boxes))

            while len(temp_boxes) != self.num_boxes:
                temp_boxes.append((0, 0, 0, 0))
                temp_actions.append(-1)
            bboxes.append(temp_boxes)
            actions.append(temp_actions)
            activities.append(self.anns[sid][src_fid]['group_activity'])

        images = np.stack(images)
        activities = np.array(activities, dtype=np.int32)
        bboxes_num = np.array(bboxes_num, dtype=np.int32)
        bboxes = np.array(bboxes, dtype=np.float32).reshape(-1, self.num_boxes, 4)
        actions = np.array(actions, dtype=np.int32).reshape(-1, self.num_boxes)
        frames = np.array(frames, dtype=np.int32)

        images = torch.from_numpy(images).float()
        bboxes = torch.from_numpy(bboxes).float()
        actions = torch.from_numpy(actions).long()
        activities = torch.from_numpy(activities).long()
        bboxes_num = torch.from_numpy(bboxes_num).int()
        frames = torch.from_numpy(frames)

        return images, bboxes, actions, activities, bboxes_num, frames

    def get_frames(self, frame):

        sid, src_fid = frame

        self.is_finetune = False
        if self.is_finetune:
            if self.is_training:
                fid = random.randint(src_fid, src_fid + self.num_frames - 1)
                return [(sid, src_fid, fid)]

            else:
                return [(sid, src_fid, fid)
                        for fid in range(src_fid, src_fid + self.num_frames)]

        else:
            if self.is_training:
                sample_frames = random.sample(range(src_fid, src_fid + self.num_frames), 10)
                sample_frames = sorted(sample_frames)
                return [(sid, src_fid, fid) for fid in sample_frames]
            else:
                sample_frames = random.sample(range(src_fid, src_fid + self.num_frames), 10)
                sample_frames = sorted(sample_frames)
                return [(sid, src_fid, fid) for fid in sample_frames]

    def load_samples_sequence(self, index, select_frames):
        OH, OW = self.feature_size
        images, bboxes = [], []
        activities, actions = [], []
        bboxes_num = []
        frames = []

        for i, (sid, src_fid, fid) in enumerate(select_frames):

            frames.append((sid, src_fid, fid))
            img = Image.open(self.images_path + '/seq%02d/frame%04d.jpg' % (sid, fid))
            img = transforms.functional.resize(img, self.image_size)
            img = np.array(img)

            if self.is_training and src_fid > self.__len__() / 2:
                H, W = FRAMES_SIZE[sid]  # 高宽

                temp_boxes = []
                for box in self.anns[sid][src_fid]['bboxes']:
                    y1, x1, y2, x2 = box
                    w1, h1, w2, h2 = x1 * W, y1 * H, x2 * W, y2 * H
                    temp_boxes.append([w1, h1, w2, h2])
                boxes = temp_boxes
                boxes = np.array(boxes)

                labels = self.anns[sid][src_fid]['actions'][:]
                labels = np.array(labels, dtype=np.int32)

                img, boxes, labels = random_flip(img, boxes, labels)
                img, boxes, labels = random_rotation(img, boxes, labels)
                temp_actions = labels.tolist()
                img = random_color(img)

                temp_boxes = []
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    w1, h1, w2, h2 = x1 / W * OW, y1 / H * OH, x2 / W * OW, y2 / H * OH
                    temp_boxes.append((w1, h1, w2, h2))

            else:
                temp_boxes = []
                for box in self.anns[sid][src_fid]['bboxes']:
                    y1, x1, y2, x2 = box
                    w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
                    temp_boxes.append((w1, h1, w2, h2))

                temp_actions = self.anns[sid][src_fid]['actions'][:]

            img = img.transpose(2, 0, 1)
            images.append(img)
            bboxes_num.append(len(temp_boxes))

            while len(temp_boxes) != self.num_boxes:
                temp_boxes.append((0, 0, 0, 0))
                temp_actions.append(-1)
            bboxes.append(temp_boxes)
            actions.append(temp_actions)
            activities.append(self.anns[sid][src_fid]['group_activity'])

        images = np.stack(images)
        activities = np.array(activities, dtype=np.int32)
        bboxes_num = np.array(bboxes_num, dtype=np.int32)
        bboxes = np.array(bboxes, dtype=np.float32).reshape(-1, self.num_boxes, 4)
        actions = np.array(actions, dtype=np.int32).reshape(-1, self.num_boxes)
        frames = np.array(frames, dtype=np.int32)

        images = torch.from_numpy(images).float()
        bboxes = torch.from_numpy(bboxes).float()
        actions = torch.from_numpy(actions).long()
        activities = torch.from_numpy(activities).long()
        bboxes_num = torch.from_numpy(bboxes_num).int()
        frames = torch.from_numpy(frames)

        return images, bboxes, actions, activities, bboxes_num, frames

class CollectiveDataset(data.Dataset):
    def __init__(self, anns, frames, images_path, image_size, feature_size, num_boxes=13, num_frames=10,
                 is_training=True, is_finetune=False):
        self.anns = anns
        self.frames = frames
        self.images_path = images_path
        self.image_size = image_size
        self.feature_size = feature_size

        self.num_boxes = num_boxes
        self.num_frames = num_frames

        self.is_training = is_training
        self.is_finetune = is_finetune
        print(len(self.frames))
        for i in range(len(anns)):
            select_frames_all = self.get_all_frames(self.frames[i])
            allSample = self.load_all_sequence(i, select_frames_all)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        select_frames = self.get_frames(self.frames[index])
        sample = self.load_samples_sequence(index, select_frames)

        return sample

    def get_all_frames(self, frame):
        sid, src_fid = frame
        self.is_finetune = False
        sample_frames = range(1, 1 + len(self.frames))
        sample_frames = sorted(sample_frames)
        return [(sid, src_fid, fid) for fid in sample_frames]

    def load_all_sequence(self, index, select_frames_all):
        OH, OW = self.feature_size
        images, bboxes = [], []
        activities, actions = [], []
        bboxes_num = []
        frames = []

        for i, (sid, src_fid, fid) in enumerate(select_frames_all):

            frames.append((sid, src_fid, fid))
            img = Image.open(self.images_path + '/seq%02d/frame%04d.jpg' % (sid, fid))
            img = transforms.functional.resize(img, self.image_size)
            img = np.array(img)

            if self.is_training and src_fid > self.__len__() / 2:
                H, W = FRAMES_SIZE[sid]  # 高宽

                temp_boxes = []
                for box in self.anns[sid][src_fid]['bboxes']:
                    y1, x1, y2, x2 = box
                    w1, h1, w2, h2 = x1 * W, y1 * H, x2 * W, y2 * H
                    temp_boxes.append([w1, h1, w2, h2])
                boxes = temp_boxes
                boxes = np.array(boxes)

                labels = self.anns[sid][src_fid]['actions'][:]
                labels = np.array(labels, dtype=np.int32)

                img, boxes, labels = random_flip(img, boxes, labels)
                img, boxes, labels = random_rotation(img, boxes, labels)
                temp_actions = labels.tolist()

                img = random_color(img)

                temp_boxes = []
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    w1, h1, w2, h2 = x1 / W * OW, y1 / H * OH, x2 / W * OW, y2 / H * OH
                    temp_boxes.append((w1, h1, w2, h2))

            else:
                temp_boxes = []
                for box in self.anns[sid][src_fid]['bboxes']:
                    y1, x1, y2, x2 = box
                    w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
                    temp_boxes.append((w1, h1, w2, h2))

                temp_actions = self.anns[sid][src_fid]['actions'][:]

            img = img.transpose(2, 0, 1)
            images.append(img)
            bboxes_num.append(len(temp_boxes))

            while len(temp_boxes) != self.num_boxes:
                temp_boxes.append((0, 0, 0, 0))
                temp_actions.append(-1)
            bboxes.append(temp_boxes)
            actions.append(temp_actions)
            activities.append(self.anns[sid][src_fid]['group_activity'])

        images = np.stack(images)
        activities = np.array(activities, dtype=np.int32)
        bboxes_num = np.array(bboxes_num, dtype=np.int32)
        bboxes = np.array(bboxes, dtype=np.float32).reshape(-1, self.num_boxes, 4)
        actions = np.array(actions, dtype=np.int32).reshape(-1, self.num_boxes)
        frames = np.array(frames, dtype=np.int32)

        images = torch.from_numpy(images).float()
        bboxes = torch.from_numpy(bboxes).float()
        actions = torch.from_numpy(actions).long()
        activities = torch.from_numpy(activities).long()
        bboxes_num = torch.from_numpy(bboxes_num).int()
        frames = torch.from_numpy(frames)

        return images, bboxes, actions, activities, bboxes_num, frames

    def get_frames(self, frame):

        sid, src_fid = frame

        self.is_finetune = False
        if self.is_finetune:
            if self.is_training:
                fid = random.randint(src_fid, src_fid + self.num_frames - 1)
                return [(sid, src_fid, fid)]

            else:
                return [(sid, src_fid, fid)
                        for fid in range(src_fid, src_fid + self.num_frames)]

        else:
            if self.is_training:
                sample_frames = random.sample(range(src_fid, src_fid + self.num_frames), 10)
                sample_frames = sorted(sample_frames)
                return [(sid, src_fid, fid) for fid in sample_frames]
            else:
                sample_frames = random.sample(range(src_fid, src_fid + self.num_frames), 10)
                sample_frames = sorted(sample_frames)
                return [(sid, src_fid, fid) for fid in sample_frames]

    def load_samples_sequence(self, index, select_frames):
        OH, OW = self.feature_size
        images, bboxes = [], []
        activities, actions = [], []
        bboxes_num = []
        frames = []

        for i, (sid, src_fid, fid) in enumerate(select_frames):

            frames.append((sid, src_fid, fid))
            img = Image.open(self.images_path + '/seq%02d/frame%04d.jpg' % (sid, fid))
            img = transforms.functional.resize(img, self.image_size)
            img = np.array(img)

            if self.is_training and src_fid > self.__len__() / 2:
                H, W = FRAMES_SIZE[sid]

                temp_boxes = []
                for box in self.anns[sid][src_fid]['bboxes']:
                    y1, x1, y2, x2 = box
                    w1, h1, w2, h2 = x1 * W, y1 * H, x2 * W, y2 * H
                    temp_boxes.append([w1, h1, w2, h2])
                boxes = temp_boxes
                boxes = np.array(boxes)

                labels = self.anns[sid][src_fid]['actions'][:]
                labels = np.array(labels, dtype=np.int32)

                img, boxes, labels = random_flip(img, boxes, labels)
                img, boxes, labels = random_rotation(img, boxes, labels)
                temp_actions = labels.tolist()

                img = random_color(img)

                temp_boxes = []
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    w1, h1, w2, h2 = x1 / W * OW, y1 / H * OH, x2 / W * OW, y2 / H * OH
                    temp_boxes.append((w1, h1, w2, h2))

            else:
                temp_boxes = []
                for box in self.anns[sid][src_fid]['bboxes']:
                    y1, x1, y2, x2 = box
                    w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
                    temp_boxes.append((w1, h1, w2, h2))

                temp_actions = self.anns[sid][src_fid]['actions'][:]

            img = img.transpose(2, 0, 1)
            images.append(img)
            bboxes_num.append(len(temp_boxes))

            while len(temp_boxes) != self.num_boxes:
                temp_boxes.append((0, 0, 0, 0))
                temp_actions.append(-1)
            bboxes.append(temp_boxes)
            actions.append(temp_actions)
            activities.append(self.anns[sid][src_fid]['group_activity'])

        images = np.stack(images)
        activities = np.array(activities, dtype=np.int32)
        bboxes_num = np.array(bboxes_num, dtype=np.int32)
        bboxes = np.array(bboxes, dtype=np.float32).reshape(-1, self.num_boxes, 4)
        actions = np.array(actions, dtype=np.int32).reshape(-1, self.num_boxes)
        frames = np.array(frames, dtype=np.int32)

        images = torch.from_numpy(images).float()
        bboxes = torch.from_numpy(bboxes).float()
        actions = torch.from_numpy(actions).long()
        activities = torch.from_numpy(activities).long()
        bboxes_num = torch.from_numpy(bboxes_num).int()
        frames = torch.from_numpy(frames)

        return images, bboxes, actions, activities, bboxes_num, frames
