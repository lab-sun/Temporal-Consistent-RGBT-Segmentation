# By Yuxiang Sun, Jul. 3, 2021
# Email: sun.yuxiang@outlook.com

import os, torch
from torch.utils.data.dataset import Dataset
import numpy as np
import PIL
from config import config
from util.util import label2onehot

class MF_dataset(Dataset):

    def __init__(self, data_dir, split, input_h=config.image_height, input_w=config.image_width ,transform=[]):
        super(MF_dataset, self).__init__()

        #assert split in ['train', 'val', 'test', 'test_day', 'test_night', 'val_test', 'most_wanted'], \
        #    'split must be "train"|"val"|"test"|"test_day"|"test_night"|"val_test"|"most_wanted"'  # test_day, test_night

        with open(os.path.join(data_dir, split+'.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]

        self.data_dir  = data_dir
        self.split     = split
        self.input_h   = input_h
        self.input_w   = input_w
        self.transform = transform
        self.n_data    = len(self.names)

    def read_image(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s.png' % (folder, name))
        image     = np.asarray(PIL.Image.open(file_path))
        return image

    def __getitem__(self, index):
        name  = self.names[index]
        image = self.read_image(name, 'RGB')        #type=array('uint8'), shape:(512,640,3)
        label = self.read_image(name, 'Label')      #type=array('uint8'), shape:(512,640)
        depth = self.read_image(name, 'Thermal')    #type=array('uint8'), shape:(512,640)
        # binary_label = self.read_image(name, 'binary_labels','label')
        #for func in self.transform:
        #    image, label = func(image, label)
        image = np.asarray(PIL.Image.fromarray(image).resize((self.input_w, self.input_h)))
        image = image.astype('float32')                 #type=array('float32'), shape:(512,640,3)
        image = np.transpose(image, (2,0,1))/255.0      #type=array('float32'), shape:(3,512,640)
        depth = np.asarray(PIL.Image.fromarray(depth).resize((self.input_w, self.input_h)))
        depth = depth.astype('float32')
        M = depth.max()
        depth = depth/M                                 #type=array('float32'), shape:(512,640)
        label = np.asarray(PIL.Image.fromarray(label).resize((self.input_w, self.input_h), resample=PIL.Image.NEAREST))
        label = label.astype('int64')                   #type=array('int64'), shape:(512,640)

        # binary_label = np.asarray(PIL.Image.fromarray(binary_label).resize((self.input_w, self.input_h), resample=PIL.Image.NEAREST))/255.0
        # binary_label = binary_label.astype('int64')
        # newlabel = label2onehot(label,3,input_w=self.input_w,input_h=self.input_h)
        # newlabel = newlabel.permute(1,2,0)
        # newlabel = PIL.Image.fromarray(np.uint8(newlabel))
        # newlabel.save('PotSeg_V2R_18/OneHot'+ name + '.png')


        #return torch.tensor(image), torch.tensor(depth).unsqueeze(0), torch.tensor(label), name

        return torch.cat((torch.tensor(image), torch.tensor(depth).unsqueeze(0)), dim=0), torch.tensor(label), name

    def __len__(self):
        return self.n_data

