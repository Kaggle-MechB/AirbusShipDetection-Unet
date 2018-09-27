
# coding: utf-8

# In[1]:


import os
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
import random


import torchvision
from torchvision.transforms import ToTensor, Normalize, Compose


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


class Params():
    BATCH_SIZE = 8
    SHIP_DIR = '../input'
    TRAIN_DIR = os.path.join(SHIP_DIR, 'train')
    TEST_DIR = os.path.join(SHIP_DIR, 'test')
    TRAIN_SEG_CSV_PATH = os.path.join(SHIP_DIR, 'train_ship_segmentations.csv')
    TEST_SEG_CSV_PATH = os.path.join(SHIP_DIR, 'test_ship_segmentations.csv')
    N_EPOCHS = 3
    LEARNING_RATE = 1e-4

    
# def printPaths():
#     paths = os.listdir(Params.TRAIN_DIR)
#     for i, path in enumerate(paths, 0):
#         if i < 10:
#             print(path)
            
# def testLoadingSegFile():
#     df = pd.read_csv(Params.TRAIN_SEG_FILE)


# In[4]:


def rle_decoder(encoded_pixels, shape=(768, 768)):
    s = encoded_pixels.split()
    #     x, y = [np.asarray(x, dtype=int) for x in (a[0:][::2], a[1:][::2])]でも同じ
    starts, length = [np.asarray(x, dtype = int) for x in (s[0::2], s[1::2])]
    starts -= 1 #broad cast
    ends = starts + length
    img = np.zeros(shape[0]*shape[1], dtype = np.uint8) #img.shape  => (768*768, )
    for low, high in zip(starts, ends):
        img[low:high] = 1
    return img.reshape(shape).T #img.shape => (768, 768) 

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

#there are pictures contain multiple ships
#this function sum up mask images of such pictures
def sum_up_mask_images(encoded_masks, size=(768, 768)):
    summed_up_mask = np.zeros(size, dtype=np.uint16)
    for encoded_mask in encoded_masks:
        if isinstance(encoded_mask, str):
            summed_up_mask += rle_decoder(encoded_mask)
    return np.expand_dims(summed_up_mask, -1) #size => e.g (768, 768, 1) i.e GrayScaledImage


def mask_overlay(image, mask, color=(0, 1, 0)):
    #ここでcolorのdtypeに統一されるらしい。cv2.addWeightedが同じdtypeじゃないとエラーを吐くので予めcolorのdtypeをimageのdtypeにしておく。
    mask= np.dstack((mask, mask, mask)) * np.array(color, dtype=image.dtype)
    
    #cv.addWeighted→アルファブレンド→image1 * weight1 + image2 * weight2 + gannmaの値が出力される。ここではgannma＝0
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    #     このままだとmaskが真っ黒な画像として認識されている。
    img = image.copy()
    ind = mask[:, :, 1] > 0
    img[ind] = weighted_sum[ind]
    return img
    

def imshow(img, mask, title=None):
    """Imshow for Tensor."""
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    mask = mask.numpy().transpose((1, 2, 0))
    mask = np.clip(mask, 0, 1)
    fig = plt.figure(figsize = (6,6))
    plt.imshow(mask_overlay(img, mask))
    if title is not None:
        plt.title(title)
    plt.show()


# def visualize():
#     df = pd.read_csv(Params.TRAIN_SEG_FILE)
#     image_name = df["ImageId"][1]
#     mask = rle_decoder(df["EncodedPixels"][1])
#     image = imread(os.path.join(Params.TRAIN_DIR, image_name))
#     mask = np.expand_dims(mask, -1)
    
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
    
#     image = std * image + mean
#     image = np.clip(image, 0, 1)
#     mask = np.clip(mask, 0, 1)
#     fig = plt.figure(figsize = (6, 6))
    
#     plt.imshow(mask_overlay(image, mask))
#     plt.show()


# In[5]:


#torch.utils.data の　Datasetを継承する。
#dataframeは予めEncodedPixelsが無い行をdropしておく。

#格納されるデータ型について


class ShipDataset(Dataset):
    def __init__(self, in_df, transform = None, mode = 'train'):
        #なんでクラス変数にしたの？
        group = list(in_df.groupby('ImageId'))
        self.image_ids = [_id for _id, _ in group]
        self.image_masks = [m["EncodedPixels"].values for _, m in group]
        self.transform = transform
        #モードを切り替えることでデータを自動的に切り替えられる。→データフレームの入力と一緒に変えないといけないのか。まぁここはそんなに問題ないかな。
        self.mode = mode
        #これは何に使うんだ？
        self.image_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            #おそらくuint8で読み込まれるので正規化していると同時にfloat型のテンソルとなることに注意
            
        ])
        
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_file_name = self.image_ids[idx]
        if self.mode == 'train':
            image_path = os.path.join(Params.TRAIN_DIR, image_file_name)
        else:
            image_path = os.path.join(Params.TEST_DIR, image_file_name)
            
        image = imread(image_path) #np.ndarray() →　dtype = uint8, shape => (768, 768, 3)
        mask = sum_up_mask_images(self.image_masks[idx]) #np.ndarray()  eshape => (768, 768, 1) のように帰ってくる
       
        
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        
        if self.mode == 'train':
            
            return self.image_transform(image), torch.from_numpy(np.moveaxis(mask.astype(np.float32), -1, 0))
        else:
            return self.image_transform(image), str(image_path)


# In[6]:


masks = pd.read_csv(Params.TRAIN_SEG_CSV_PATH)
print(masks.shape[0], "input dataframe")
masks = masks.drop(masks[masks.EncodedPixels.isnull()].sample(70000, random_state=42).index)
unique_img_ids = masks.groupby('ImageId').size().reset_index(name='counts')
#DataFrame型を入れる
#masksを直接入れると同じ名前のものがかぶる。
train_ids, valid_ids = train_test_split(unique_img_ids,
                                     test_size=0.05,
                                     stratify = unique_img_ids['counts'],
                                     random_state=42
                                    )

#masksからtrain_dfを取り出すためだけの処理
#なんでmaskをそのまま使わないんだろう
train_df = pd.merge(masks, train_ids)
valid_df = pd.merge(masks, valid_ids)
print(train_df.shape[0], "training masks")
print(valid_df.shape[0], "validation masks")


# In[7]:


#transformをつくる
#ここは流用で問題ないと思う。

"""
    Implementation from  https://github.com/ternaus/robot-surgery-segmentation
"""

def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)

class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, mask=None):
        for t in self.transforms:
            x, mask = t(x, mask)
        return x, mask
    
class OneOf:
    def __init__(self, transforms, prob=0.5):
        self.transforms = transforms
        self.prob = prob

    def __call__(self, x, mask=None):
        if random.random() < self.prob:
            t = random.choice(self.transforms)
            t.prob = 1.
            x, mask = t(x, mask)
        return x, mask

class OneOrOther:
    def __init__(self, first, second, prob=0.5):
        self.first = first
        first.prob = 1.
        self.second = second
        second.prob = 1.
        self.prob = prob

    def __call__(self, x, mask=None):
        if random.random() < self.prob:
            x, mask = self.first(x, mask)
        else:
            x, mask = self.second(x, mask)
        return x, mask


class ImageOnly:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, x, mask=None):
        return self.trans(x), mask


class VerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 0)
            if mask is not None:
                mask = cv2.flip(mask, 0)
        return img, mask


class HorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)
        return img, mask


class RandomFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            img = cv2.flip(img, d)
            if mask is not None:
                mask = cv2.flip(mask, d)
        return img, mask


class Transpose:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = img.transpose(1, 0, 2)
            if mask is not None:
                mask = mask.transpose(1, 0, 2)
        return img, mask


class RandomRotate90:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            factor = random.randint(0, 4)
            img = np.rot90(img, factor)
            if mask is not None:
                mask = np.rot90(mask, factor)
        return img.copy(), mask.copy()


class Rotate:
    def __init__(self, limit=90, prob=0.5):
        self.prob = prob
        self.limit = limit

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            angle = random.uniform(-self.limit, self.limit)

            height, width = img.shape[0:2]
            mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
            img = cv2.warpAffine(img, mat, (height, width),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            if mask is not None:
                mask = cv2.warpAffine(mask, mat, (height, width),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101)

        return img, mask


class RandomCrop:
    def __init__(self, size):
        self.h = size[0]
        self.w = size[1]

    def __call__(self, img, mask=None):
        height, width, _ = img.shape

        h_start = np.random.randint(0, height - self.h)
        w_start = np.random.randint(0, width - self.w)

        img = img[h_start: h_start + self.h, w_start: w_start + self.w,:]

        assert img.shape[0] == self.h
        assert img.shape[1] == self.w

        if mask is not None:
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=2)
            mask = mask[h_start: h_start + self.h, w_start: w_start + self.w,:]

        return img, mask


class Shift:
    def __init__(self, limit=4, prob=.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            limit = self.limit
            dx = round(random.uniform(-limit, limit))
            dy = round(random.uniform(-limit, limit))

            height, width, channel = img.shape
            y1 = limit + 1 + dy
            y2 = y1 + height
            x1 = limit + 1 + dx
            x2 = x1 + width

            img1 = cv2.copyMakeBorder(img, limit + 1, limit + 1, limit + 1, limit + 1,
                                      borderType=cv2.BORDER_REFLECT_101)
            img = img1[y1:y2, x1:x2, :]
            if mask is not None:
                msk1 = cv2.copyMakeBorder(mask, limit + 1, limit + 1, limit + 1, limit + 1,
                                          borderType=cv2.BORDER_REFLECT_101)
                mask = msk1[y1:y2, x1:x2, :]

        return img, mask


class ShiftScale:
    def __init__(self, limit=4, prob=.25):
        self.limit = limit
        self.prob = prob

    def __call__(self, img, mask=None):
        limit = self.limit
        if random.random() < self.prob:
            height, width, channel = img.shape
            assert (width == height)
            size0 = width
            size1 = width + 2 * limit
            size = round(random.uniform(size0, size1))

            dx = round(random.uniform(0, size1 - size))
            dy = round(random.uniform(0, size1 - size))

            y1 = dy
            y2 = y1 + size
            x1 = dx
            x2 = x1 + size

            img1 = cv2.copyMakeBorder(img, limit, limit, limit, limit, borderType=cv2.BORDER_REFLECT_101)
            img = (img1[y1:y2, x1:x2, :] if size == size0
            else cv2.resize(img1[y1:y2, x1:x2, :], (size0, size0), interpolation=cv2.INTER_LINEAR))

            if mask is not None:
                msk1 = cv2.copyMakeBorder(mask, limit, limit, limit, limit, borderType=cv2.BORDER_REFLECT_101)
                mask = (msk1[y1:y2, x1:x2, :] if size == size0
                else cv2.resize(msk1[y1:y2, x1:x2, :], (size0, size0), interpolation=cv2.INTER_LINEAR))

        return img, mask


class ShiftScaleRotate:
    def __init__(self, shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, prob=0.5):
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            height, width, channel = img.shape

            angle = random.uniform(-self.rotate_limit, self.rotate_limit)
            scale = random.uniform(1 - self.scale_limit, 1 + self.scale_limit)
            dx = round(random.uniform(-self.shift_limit, self.shift_limit)) * width
            dy = round(random.uniform(-self.shift_limit, self.shift_limit)) * height

            cc = math.cos(angle / 180 * math.pi) * scale
            ss = math.sin(angle / 180 * math.pi) * scale
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)
            img = cv2.warpPerspective(img, mat, (width, height),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101)
            if mask is not None:
                mask = cv2.warpPerspective(mask, mat, (width, height),
                                           flags=cv2.INTER_NEAREST,
                                           borderMode=cv2.BORDER_REFLECT_101)
 
        return img, mask


class CenterCrop:
    def __init__(self, size):
        self.height = size[0]
        self.width = size[1]

    def __call__(self, img, mask=None):
        h, w, c = img.shape
        dy = (h - self.height) // 2
        dx = (w - self.width) // 2
        y1 = dy
        y2 = y1 + self.height
        x1 = dx
        x2 = x1 + self.width
        img = img[y1:y2, x1:x2,:]
        if mask is not None:
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=2)
            mask = mask[y1:y2, x1:x2,:]

        return img, mask
    
class RandomBrightness:
    def __init__(self, limit=0.1, prob=0.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            alpha = 1.0 + self.limit * random.uniform(-1, 1)

            maxval = np.max(img[..., :3])
            dtype = img.dtype
            img[..., :3] = clip(alpha * img[..., :3], dtype, maxval)
        return img


class RandomContrast:
    def __init__(self, limit=.1, prob=.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            alpha = 1.0 + self.limit * random.uniform(-1, 1)

            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
            maxval = np.max(img[..., :3])
            dtype = img.dtype
            img[:, :, :3] = clip(alpha * img[:, :, :3] + gray, dtype, maxval)
        return img


# In[8]:


#transformの設定
#あとで実装したGithubを見てみる。
train_transform = DualCompose([
    HorizontalFlip(),
    VerticalFlip(),
    RandomCrop((256, 256, 3))
])

val_transform = DualCompose([
    CenterCrop((512, 512, 3))
])


# In[9]:


#データロード関数をラップ
def make_loader(in_df, batch_size, shuffle=False,transform=None):
    return DataLoader(dataset = ShipDataset(in_df, transform=transform),
                   batch_size = batch_size,
                   shuffle = shuffle,
                   num_workers = 0,
                   )

train_loader = make_loader(train_df, Params.BATCH_SIZE, shuffle=True, transform=train_transform)
valid_loader = make_loader(valid_df, Params.BATCH_SIZE // 2, shuffle=False, transform=val_transform)


# In[10]:


class Bottleneck(nn.Module):
    
    def __init__(self, in_channels, out_channels, prev_channels=0, upsample = False, downsample=False):
        super(Bottleneck, self).__init__()
        self.upsample = upsample
        self.downsample = downsample
        self.channels = prev_channels + in_channels
        
        self.conv1 = nn.Conv2d(self.channels, out_channels, kernel_size = 1, padding = 0)
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(self.channels, in_channels, kernel_size = 1, padding = 0)        
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size = 3, padding = 1)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear')
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, padding = 0)
        self.dropout = nn.Dropout2d(p=0.2)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

    
    def forward(self, x):
        residual = self.conv1(x)
        
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.pool(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
       
        x = self.bn3(x)
        x = self.relu(x)
        x = self.up(x)
        x = self.conv4(x)
        x = self.dropout(x)
        
        x += residual
        
        if self.downsample:
            x = self.pool(x)
        if self.upsample:
            x = self.up(x)
        
        return x


# In[11]:


class SimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, prev_channels=0, upsample = False, downsample = False):
        super(SimpleBlock, self).__init__()
        
        self.upsample = upsample
        self.downsample = downsample
        self.channels = prev_channels + in_channels
        
        self.conv1 = nn.Conv2d(self.channels, out_channels, kernel_size = 1, padding = 0)
        self.bn = nn.BatchNorm2d(prev_channels + in_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(self.channels, out_channels, kernel_size = 3, padding = 1)
        self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear')
        self.dropout = nn.Dropout2d(p=0.2)
        
    def forward(self, x):
        residual = self.conv1(x)
        
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.up(x)
        x = self.dropout(x)
        
        x +=  residual
        if self.downsample:
            x = self.pool(x)
        if self.upsample:
            x = self.up(x)
            
        return x


# In[12]:


#forwardのレイヤーが分かり辛いのでなんとかならないか
class ResUNet(nn.Module):
    
    def __init__(self):
        super(ResUNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, padding = 1)
        self.simple_d1 = SimpleBlock(16, 16, downsample = True)
        self.bottleneck_d1 = Bottleneck(16, 32, downsample = True)
        self.bottleneck_d2 = Bottleneck(32, 64)
        self.bottleneck_d3 = Bottleneck(64, 128)
        self.bottleneck_d4 = Bottleneck(128, 256, downsample = True)
        self.bottleneck_d5 = Bottleneck(256, 512)
        self.bottleneck_d6 = Bottleneck(512, 1024)
        self.bottleneck_d7 = Bottleneck(1024, 2048)
        self.bottleneck_m1 = Bottleneck(2048, 2048, downsample = True)
        self.bottleneck_m2 = Bottleneck(2048, 2048, upsample = True)
        self.bottleneck_u1 = Bottleneck(2048, 1024, prev_channels = 2048)
        self.bottleneck_u2 = Bottleneck(1024, 512)
        self.bottleneck_u3 = Bottleneck(512, 256)
        self.bottleneck_u4 = Bottleneck(256, 128, upsample = True)
        self.bottleneck_u5 = Bottleneck(128, 64, prev_channels = 128)
        self.bottleneck_u6 = Bottleneck(64, 32)
        self.bottleneck_u7 = Bottleneck(32, 16, upsample = True)
        self.simple_u1 = SimpleBlock(16, 16, prev_channels = 16, upsample = True)
        self.conv2 = nn.Conv2d(in_channels = 16 + 16, out_channels = 16, kernel_size = 3, padding = 1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 1, kernel_size = 1, padding = 0)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
#         print('x : {}'.format(x.size()))
        x1 = self.conv1(x)
        
#         print('x1 : {}'.format(x1.size()))
        x2 = self.simple_d1(x1)
        
#         print('x2 : {}'.format(x2.size()))
        x3 = self.bottleneck_d1(x2)
#         print('x3 : {}'.format(x3.size()))
        x3 = self.bottleneck_d2(x3)
#         print('x3 : {}'.format(x3.size()))
        x3 = self.bottleneck_d3(x3)
#         print('x3 : {}'.format(x3.size()))
        x4 = self.bottleneck_d4(x3)
#         print('x4 : {}'.format(x4.size()))
        x4 = self.bottleneck_d5(x4)
#         print('x4 : {}'.format(x4.size()))
        x4 = self.bottleneck_d6(x4)
#         print('x4 : {}'.format(x4.size()))
        x4 = self.bottleneck_d7(x4)
#         print('x4 : {}'.format(x4.size()))
        x5 = self.bottleneck_m1(x4)
#         print('x5 : {}'.format(x5.size()))
        x5 = self.bottleneck_m2(x5)
#         print('x5 : {}'.format(x5.size()))
        x6 = self.bottleneck_u1(torch.cat((x4, x5), dim = 1))
#         print('x6 : {}'.format(x6.size()))
        x6 = self.bottleneck_u2(x6)
#         print('x6 : {}'.format(x6.size()))
        x6 = self.bottleneck_u3(x6)
#         print('x6 : {}'.format(x6.size()))
        x6 = self.bottleneck_u4(x6)
#         print('x6 : {}'.format(x6.size()))
        x7 = self.bottleneck_u5(torch.cat((x3, x6), dim = 1))
#         print('x7 : {}'.format(x7.size()))
        x7 = self.bottleneck_u6(x7)
#         print('x7 : {}'.format(x7.size()))
        x7 = self.bottleneck_u7(x7)
#         print('x7 : {}'.format(x7.size()))
        x8 = self.simple_u1(torch.cat((x2, x7), dim = 1))
#         print('x8 : {}'.format(x8.size()))
        x9 = self.conv2(torch.cat((x1, x8), dim = 1))
#         print('x9 : {}'.format(x9.size()))
        x9 = self.bn(x9)
#         print('x9 : {}'.format(x9.size()))
        x9 = self.relu(x9)
#         print('x9 : {}'.format(x9.size()))
        x9 = self.conv3(x9)
#         print('x9 : {}'.format(x9.size()))
        x9 = self.tanh(x9)
        
        return x9


# In[13]:


class LossBinary:
    """
     Implementation from  https://github.com/ternaus/robot-surgery-segmentation
    """
    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight
        
    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)
        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1.0).float()
            jaccard_output = F.sigmoid(outputs)
            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()
            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss


# In[14]:


def dice(pred, targs):
    pred = (pred>0).float()
    return 2.0 * (pred*targs).sum() / ((pred+targs).sum() + 1.0)

def IoU(pred, targs):
    pred = (pred>0).float()
    intersection = (pred*targs).sum()
    return intersection / ((pred+targs).sum() - intersection + 1.0)


# In[15]:


# sume helper functions
#これはそのまま写した
#なぜ、Variableクラスを使わないんだろう
def variable(x, volatile=False):
    if isinstance(x, (list, tuple)):
        return [variable(y, volatile=volatile) for y in x]
    return cuda(Variable(x, volatile=volatile))

def cuda(x):
    return x.cuda(async=True) if torch.cuda.is_available() else x


# In[16]:


#validationについても写した
def validation(model:nn.Module, criterion, valid_loader):
    print('Validation on hold-out.....')
    model.eval()
    losses = []
    for inputs, targets in valid_loader:
        inputs = variable(inputs, volatile=True)
        targets = variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.data[0])
    valid_loss = np.mean(losses)
    
    print('Valid loss: {:.5f}'.format(valid_loss))
    metrics = {'valid_loss': valid_loss}
    return metrics


# In[ ]:


def train(model, criterion, train_loader, valid_loader, validation, init_optimizer):
    #最適化関数の初期化
    lr = Params.LEARNING_RATE
    optimizer = init_optimizer(lr)
    
    #GPU使える？
    if torch.cuda.is_available():
        model.cuda()
    
    #エポック数
    n_epochs = Params.N_EPOCHS
    
    #ファイル保存用パス
    fold = 1
    model_path = Path('model_{fold}.pt'.format(fold=fold))
    
    #モデルの読み込み
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
        
    else:
        epoch = 1
        step = 0
        
    #モデルの保存
    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step' : step,
    }, str(model_path))
    
    
    #ここは後でParamsクラスのインスタンスにするほうがいいかも
    report_each = 50
    valid_losses = []
    
    for _epoch in range(epoch, n_epochs + 1):
        model.train()
        random.seed()
        tq = tqdm(total = len(train_loader) * Params.BATCH_SIZE)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        tl = train_loader
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                inputs, targets = variable(inputs), variable(targets)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.data[0])
                #直近５０個の平均損失を計算している
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
        
            tq.close()
            save(epoch+1)
            valid_metrics = validation(model, criterion, valid_loader)
        
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)
        
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl + C, saving snapshot')
            save(epoch)
            print('done')
            return


# In[ ]:


#学習を行ってみる
model = ResUNet()
train(init_optimizer = lambda lr: optim.Adam(model.parameters()),
         model = model,
         criterion= LossBinary(jaccard_weight = 5),
         train_loader=train_loader,
         valid_loader=valid_loader,
         validation=validation,
     )


# In[ ]:


dataset_valid = ShipDataset(valid_df)
imshow(*dataset_valid[0])


# In[ ]:


model = UNet()
model_path = 'model_1.pt'
state = torch.load(str(model_path))
state = {key.replace('module.', ''): value for key, value in state[
    'model'].items()}
model.load_state_dict(state)
if torch.cuda.is_available():
    model.cuda()
    

model.eval()


# In[ ]:


valid_ds = ShipDataset(valid_df)
img, _  = valid_ds[0]
input_img = torch.unsqueeze(variable(img, volatile = True), dim=0)

mask = model(input_img)
out_mask = torch.squeeze(mask.data.cpu(), dim = 0)
imshow(img, out_mask)


# In[ ]:


test_paths = os.listdir(Params.TEST_DIR)
print(len(test_paths), 'test images found')


# In[107]:


test_df = pd.DataFrame({'ImageId': test_paths, 'EncodedPixels': None})
from skimage.morphology import binary_opening, disk
loader = DataLoader(
       dataset=ShipDataset(test_df, transform=None, mode='predict'),
       shuffle=False,
       num_workers=0,
       pin_memory=torch.cuda.is_available()
)


# In[ ]:


#閾値が0.5で固定のようだが、ハイパーパラメタなのでこれをマジックナンバーにする。
out_pred_rows = []
for batch_num, (inputs, paths) in enumerate(tqdm(loader, desc='Predict')):
    inputs = variable(inputs, volatile=True)
    outputs = model(inputs)
#     print('mask generated')
    
    for i, image_name in enumerate(paths):
        mask = F.sigmoid(outputs[i,0]).data.cpu().numpy()
#         print('checkpoint 1')

        cur_seg = binary_opening(mask>0.5, disk(2))
        cur_rles = multi_rle_encode(cur_seg)
#         print('checkpoint 2')
        
        if len(cur_rles)>0:
            for c_rle in cur_rles:
                out_pred_rows += [{'ImageId': image_name, 'EncodedPixels': c_rle}]
        else:
            out_pred_rows += [{'ImageId': image_name, 'EncodedPixels': None}]


# In[ ]:


submission_df = pd.DataFrame(out_pred_rows)[['ImageId', 'EncodedPixels']]
submission_df.to_csv('submission.csv', index = False)
print(submission_df.shape)
submission_df.sample(10)

