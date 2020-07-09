import os
import random
import shutil


def splitter(root, ratio):
    file_list = os.listdir(root + 'train' + '/image/')
    train_item = [file for file in file_list if file.endswith(".PNG")]
    train_item = random.shuffle(train_item)
    gaesu = len(train_item) * (int(ratio) / 100)
    for i in range(0, gaesu):
        shutil.move(root + 'train' + '/image/' + train_item[gaesu], root + 'val' + '/image/' + train_item[gaesu])
        shutil.move(root + 'train' + '/mask/' + train_item[gaesu], root + 'val' + '/mask/' + train_item[gaesu])
