'''
@Author: your name
@Date: 2020-01-06 15:08:07
LastEditTime: 2021-01-07 10:46:01
LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /HMER/config.py
'''

class Config():  
    seed = 2020
    
    datasets = ['data/train.pkl', 'data/label/train_caption.txt']
    valid_datasets = ['data/valid.pkl', 'data/label/test_caption_2014.txt']
    dictionaries = 'data/dictionary.txt'

    batch_Imagesize = 500000
    valid_batch_Imagesize = 500000 
    maxImagesize = 100000

    maxlen = 70
    hidden_size = 256
    num_class = 112

    num_epoch = 60
    lr = 0.0001
    batch_size = 2
    batch_size_t = 2
    teacher_forcing_ratio = 0.8
 
    num_workers = 4

cfg = Config()