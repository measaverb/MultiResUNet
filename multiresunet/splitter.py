import os
import random
import shutil


def splitter(root, ratio):
    file_list = os.listdir(root + 'train' + '/image/')
    # print(file_list)
    # train_item = [file for file in file_list if file.endswith(".PNG")]
    # print(train_item)
    train_item = random.shuffle(file_list)
    gaesu = len(file_list) // int(ratio)
    for i in range(0, gaesu):
        shutil.move(root + 'train' + '/image/' + file_list[i], root + 'val' + '/image/' + file_list[i])
        shutil.move(root + 'train' + '/mask/' + file_list[i], root + 'val' + '/mask/' + file_list[i])
