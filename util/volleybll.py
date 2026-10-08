import numpy as np
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data

ACTIVITIES = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
              'l_set', 'l-spike', 'l-pass', 'l_winpoint']
NUM_ACTIVITIES = 8
#人群行为种类和数量
ACTIONS = ['blocking', 'digging', 'falling', 'jumping',
           'moving', 'setting', 'spiking', 'standing',
           'waiting']
NUM_ACTIONS = 9

def volley_read_annotations(path):
    annotations = {}

    gact_to_id = {name: i for i, name in enumerate(ACTIVITIES)}
    act_to_id = {name: i for i, name in enumerate(ACTIONS)}

    with open(path) as f:
        for l in f.readlines():
            values = l[:-1].split(' ')
            file_name = values[0]
            activity = gact_to_id[values[1]]

            values = values[2:]
            num_people = len(values) // 5

            action_names = values[4::5]
            actions = [act_to_id[name]
                       for name in action_names]

            def _read_bbox(xywh):
                x, y, w, h = map(int, xywh)
                return y, x, y+h, x+w
            bboxes = np.array([_read_bbox(values[i:i+4])
                               for i in range(0, 5*num_people, 5)])

            fid = int(file_name.split('.')[0])
            annotations[fid] = {
                'file_name': file_name,
                'group_activity': activity,
                'actions': actions,
                'bboxes': bboxes,
            }
    return annotations


def volleyball_read_labelannotations(path, seqs):
    data = {}
    for sid in seqs:
        data[sid] = volley_read_annotations(path + '/%d/annotations.txt' % sid)
    return data

#获取视频序列和帧
def volleyball_all_frames(data):
    frames = []
    for sid, anns in data.items():
        for fid, ann in anns.items():
            frames.append((sid, fid))
    return frames

class VolleyballDataset(data.DataLoader):
    def __init__(self,anns,frames,images_path,image_size,feature_size,num_boxes=12,num_before=4,num_after=4,is_training=True,is_finetune=False):
        # super(VolleyballDataset, self).__init__()
        self.anns=anns
        self.frames=frames
        self.images_path=images_path
        self.image_size=image_size
        self.feature_size=feature_size

        self.num_boxes=num_boxes
        self.num_before=num_before
        self.num_after=num_after

        self.is_training=is_training
        self.is_finetune=is_finetune

    def __len__(self):
        return len(self.frames)

    def __getitem__(self,index):
        select_frames=self.volley_frames_sample(self.frames[index])
        sample=self.load_samples_sequence(select_frames)

        return sample

    def volley_frames_sample(self,frame):
        sid, src_fid = frame

        self.is_finetune = False
        self.is_training = True
        if self.is_finetune:
            if self.is_training:
                fid=random.randint(src_fid-self.num_before, src_fid+self.num_after)
                return [(sid, src_fid, fid)]
            else:
                return [(sid, src_fid, fid)
                        for fid in range(src_fid-self.num_before, src_fid+self.num_after+1)]
        else:
            if self.is_training:
                sample_frames=random.sample(range(src_fid-self.num_before, src_fid+self.num_after+1), 8)
                return [(sid, src_fid, fid)
                        for fid in sample_frames]
            else:
                return [(sid, src_fid, fid)
                        for fid in  [src_fid-3,src_fid,src_fid+3, src_fid-4,src_fid-1,src_fid+2, src_fid-2,src_fid+1,src_fid+4 ]]

    def load_samples_sequence(self,select_frames):
        OH, OW=self.feature_size

        images, boxes = [], []
        bboxes_num = []
        activities, actions = [], []
        for i, (sid, src_fid, fid) in enumerate(select_frames):

            img = Image.open(self.images_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid))

            img=transforms.functional.resize(img,self.image_size)
            img=np.array(img)

            img=img.transpose(2,0,1)
            images.append(img)

            temp_boxes=[]
            for box in self.anns[sid][src_fid]['bboxes']:
                y1,x1,y2,x2=box
                w1,h1,w2,h2 = x1*OW, y1*OH, x2*OW, y2*OH
                temp_boxes.append((w1,h1,w2,h2))

            boxes.append(temp_boxes)
            actions.append(self.anns[sid][src_fid]['actions'])

            while len(boxes[-1]) != self.num_boxes:
                boxes[-1] = np.vstack([boxes[-1], boxes[-1][:self.num_boxes-len(boxes[-1])]])
                actions[-1] = actions[-1] + actions[-1][:self.num_boxes-len(actions[-1])]
            activities.append(self.anns[sid][src_fid]['group_activity'])
            bboxes_num.append(len(temp_boxes))

        images = np.stack(images)
        activities = np.array(activities, dtype=np.int32)
        bboxes = np.vstack(boxes).reshape([-1, self.num_boxes, 4])
        bboxes_num = np.array(bboxes_num, dtype=np.int32)
        actions = np.hstack(actions).reshape([-1, self.num_boxes])

        images=torch.from_numpy(images).float()
        bboxes=torch.from_numpy(bboxes).float()
        bboxes_num=torch.from_numpy(bboxes_num).int()
        actions=torch.from_numpy(actions).long()
        activities=torch.from_numpy(activities).long()

        return images, bboxes,  actions, activities, bboxes_num
