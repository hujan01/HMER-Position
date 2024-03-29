'''
Author: sigmoid
Description: 
Email: 595495856@qq.com
Date: 2020-12-13 19:51:52
LastEditTime: 2021-01-13 14:38:38
'''
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from skimage import transform
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data
from torch import optim

class custom_dset(data.Dataset):
    """ 格式化数据 """
    def __init__(self, train, train_label, uidList):
        self.train = train
        self.train_label = train_label
        self.uidList = uidList
    def __getitem__(self, index):
        train_setting = torch.from_numpy(np.array(self.train[index]))
        label_setting = torch.from_numpy(np.array(self.train_label[index])).type(torch.LongTensor)
        uid_setting   = self.uidList[index]

        size = train_setting.size()
        train_setting = train_setting.view(1, size[2], size[3])
        label_setting = label_setting.view(-1)
        return train_setting, label_setting, uid_setting
    def __len__(self):
        return len(self.train)
        
def collate_fn_single(batch):
    """ 不引入掩码 单通道 """
    batch.sort(key=lambda x: len(x[1]), reverse=True) # 按图片大小排序
    img, label = zip(*batch)

    # 一个batch中最大的高宽
    maxH = 0
    maxW = 0
    for j in range(len(img)):
        size = img[j].size()
        if size[1] > maxH:
            maxH = size[1]
        if size[2] > maxW:
            maxW = size[2]

    k = 0
    for ii in img:
        ii = ii.float()
        img_size_h = ii.size()[1]
        img_size_w = ii.size()[2]

        # padding 图片
        padding_h = maxH-img_size_h
        padding_w = maxW-img_size_w
        m = torch.nn.ConstantPad2d((0, padding_w, 0, padding_h), 255.)
        img_sub_padding = m(ii)
        img_sub_padding = img_sub_padding.unsqueeze(0)

        if k==0:
            img_padding = img_sub_padding
        else:
            img_padding = torch.cat((img_padding, img_sub_padding), dim=0)
        k = k+1
        
    max_len = len(label[0])+1  
    k1 = 0
    for ii1 in label:
        ii1 = ii1.long()
        ii1 = ii1.unsqueeze(0)
        ii1_len = ii1.size()[1]
        m = torch.nn.ZeroPad2d((0, max_len-ii1_len, 0, 0))
        ii1_padding = m(ii1)
        if k1 == 0:
            label_padding = ii1_padding
        else:
            label_padding = torch.cat((label_padding, ii1_padding), dim=0)
        k1 = k1+1

    img_padding = img_padding/255.0
    return img_padding, label_padding

def collate_fn_double(batch):
    """ 引入掩码 双通道"""
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    img, label, uid = zip(*batch)
    aa1 = 0
    bb1 = 0
    k = 0
    k1 = 0
    max_len = len(label[0])+1
    for j in range(len(img)):
        size = img[j].size()
        if size[1] > aa1:
            aa1 = size[1]
        if size[2] > bb1:
            bb1 = size[2]

    for ii in img:
        ii = ii.float()
        img_size_h = ii.size()[1]
        img_size_w = ii.size()[2]
        img_mask_sub_s = torch.ones(1,img_size_h,img_size_w).type(torch.FloatTensor)
        img_mask_sub_s = img_mask_sub_s*255.0
        img_mask_sub = torch.cat((ii,img_mask_sub_s),dim=0)
        padding_h = aa1-img_size_h
        padding_w = bb1-img_size_w
        m = torch.nn.ZeroPad2d((0,padding_w,0,padding_h))
        img_mask_sub_padding = m(img_mask_sub)
        img_mask_sub_padding = img_mask_sub_padding.unsqueeze(0)
        if k==0:
            img_padding_mask = img_mask_sub_padding
        else:
            img_padding_mask = torch.cat((img_padding_mask,img_mask_sub_padding),dim=0)
        k = k+1

    for ii1 in label:
        ii1 = ii1.long()
        ii1 = ii1.unsqueeze(0)
        ii1_len = ii1.size()[1]
        m = torch.nn.ZeroPad2d((0,max_len-ii1_len,0,0))
        ii1_padding = m(ii1)
        if k1 == 0:
            label_padding = ii1_padding
        else:
            label_padding = torch.cat((label_padding,ii1_padding),dim=0)
        k1 = k1+1

    img_padding_mask = img_padding_mask/255.0
    return img_padding_mask, label_padding, uid[0]

def cmp_result(label, rec):
    """ 编辑距离 """
    dist_mat = np.zeros((len(label)+1, len(rec)+1),dtype='int32')
    dist_mat[0,:] = range(len(rec) + 1)
    dist_mat[:,0] = range(len(label) + 1)
    for i in range(1, len(label) + 1):
        for j in range(1, len(rec) + 1):
            hit_score = dist_mat[i-1, j-1] + (label[i-1] != rec[j-1])
            ins_score = dist_mat[i,j-1] + 1
            del_score = dist_mat[i-1, j] + 1
            dist_mat[i,j] = min(hit_score, ins_score, del_score)
    dist = dist_mat[len(label), len(rec)]
    return dist, len(label)

def get_all_dist(label, rec):
    """ 得到插入，删除，修改 """
    dist_mat = np.zeros((len(label)+1, len(rec)+1), dtype='int32')
    dist_mat[0,:] = range(len(rec) + 1)
    dist_mat[:,0] = range(len(label) + 1)
    for i in range(1, len(label) + 1):
        for j in range(1, len(rec) + 1):
            sub_score = dist_mat[i-1, j-1] + (label[i-1] != rec[j-1]) #替换， 相同加0，不同加1
            ins_score = dist_mat[i,j-1] + 1 #插入
            del_score = dist_mat[i-1, j] + 1 #删除
            dist_mat[i,j] = min(sub_score, ins_score, del_score)

    dist = dist_mat[len(label), len(rec)]
    return dist, len(label), sub_score, ins_score, del_score

def load_dict(dictFile):
    fp = open(dictFile)
    stuff = fp.readlines()
    fp.close()
    lexicon = {}
    for l in stuff:
        w = l.strip().split()
        lexicon[w[0]] = int(w[1])
    print('total words/phones', len(lexicon))
    return lexicon

def show_attention_images(img, preds, attn, attn_h, attn_w, smooth=False):
    w, h = img.size
    attn = attn.squeeze(0)
    preds = ['<sos>']+preds
    for i, (pred, a) in enumerate(zip(preds, attn)):
        fig, ax = plt.subplots(1, 1)
        # Resize attentions from flat to 2D (L = H x W)
        a_2d = a.view(attn_h, attn_w)
        a_2d = a_2d.detach().cpu().numpy()
        upscale_factor = max(h // attn_h, w // attn_w)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(img, 'gray')
        attn_mask = (
            transform.pyramid_expand(a_2d, upscale=upscale_factor, multichannel=False)
            if smooth
            else transform.resize(a_2d, (h, w), mode="reflect", anti_aliasing=True)
        )
        attn_img = ax.imshow(attn_mask, 'gray', alpha=0.3)
        # attn_img.set_cmap(cm.Greys)
        ax.text(1.05, 0.5, "{:>5}".format(pred), transform=ax.transAxes, fontsize=15)
        plt.savefig("results/filename_%d.png"%(i))
        print("save %d symbol attention image"%(i))
        plt.show()
        plt.close()