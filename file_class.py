import glob
import os
from itertools import chain
import random
from shutil import copy
from tqdm import tqdm

def dele_file(path_top,file_type = '/*',n = 1):
    path_top1 = '%s'
    for i in range(0, n-1):
        path_top1 = path_top1 + '/*'
    path_top2 = path_top1 +'/*'+ file_type
    path_top1 = path_top1 + file_type
    path_top1 = path_top1 % path_top
    path_top2 = path_top2 % path_top
    paths1 = glob.iglob(path_top1)
    paths2 = glob.iglob(path_top2)
    paths = chain(paths1, paths2)
    pathss = list(paths)
    print('删除的文件举例：{},确认删除输入1,否则输入0:'.format(random.choice(pathss)))
    i = input()
    if i == '1':
        for path1 in pathss:
            os.remove(path1)
            print('删除的文件：{}'.format(path1))
        print('删除完成')
    else:
        print('取消删除')

def copy_file(path_top,path_to,file_type = '/*',n=1,rename = False):
    path_top1 = '%s'
    for i in range(0, n-1):
        path_top1 = path_top1 + '/*'
    path_top2 = path_top1 +'/*'+ file_type
    path_top1 = path_top1 + file_type
    # print(path_top1)
    path_top1 = path_top1 % path_top
    path_top2 = path_top2 % path_top
    paths1 = glob.iglob(path_top1)
    paths2 = glob.iglob(path_top2)
    paths = chain(paths1, paths2)
    i = 0
    for path in tqdm(paths):
        try:
            if rename:
                i = i + 1
                path_to_new = os.path.join(path_to, os.path.split(path)[1][
                                                    :-len(os.path.split(path)[1].split('.')[-1]) - 1] + '_' + str(
                    i) + '.' + os.path.split(path)[1].split('.')[-1])
                # print(path_to_new)
                copy(path, path_to_new)
            else:
                copy(path, path_to)
        except:
            continue


#保持原文件层次
def copy_file_name(path_top,path_to,file_type = '/*',n = 1):
    path_top1 = '%s'
    for i in range(0, n-1):
        path_top1 = path_top1 + '/*'
    path_top2 = path_top1 +'/*'+ file_type
    path_top1 = path_top1 + file_type
    path_top1 = path_top1 % path_top
    path_top2 = path_top2 % path_top
    paths1 = glob.iglob(path_top1)
    paths2 = glob.iglob(path_top2)
    paths = chain(paths1, paths2)
    for path in tqdm(paths):
        try:
            path_all = str(path).split('\\')
            path_to1 = path_to
            for i in path_all[len(str(path_top).split('\\')):]:
                path_to1 = os.path.join(path_to1, i)
            # print(os.path.split(path_to1)[0])
            if not os.path.exists(os.path.split(path_to1)[0]):
                os.makedirs(os.path.split(path_to1)[0])
                print(os.path.split(path_to1)[0])
            copy(path, path_to1)
        except:
            continue
# 遍历文件夹下的文件，返回文件具体文件名
def ergodic(path_top,arrangement = 1, file_type = '/*',flag = True):
    if flag:
        path_top1 = '%s'
        ans = []
        for i in range(0, arrangement - 1):
            path_top1 = path_top1 + '/*'
        path_top1 = path_top1 + file_type
        path_top2 = path_top1 % path_top
        paths = glob.iglob(path_top2)
        for path in paths:
            ans.append(path)
    else:

        ans = []
        for i in range(0, arrangement):
            path_top1 = '%s'
            for j in range(0, i - 1):
                path_top1 = path_top1 + '/*'
            path_top1 = path_top1 + file_type
            path_top2 = path_top1 % path_top
            paths = glob.iglob(path_top2)
            try:
                for path in paths:
                    ans.append(path)
            except:
                continue
    return ans


class FileClass:

    def __init__(self,path_from,path_to):
        self.path_from = path_from
        self.path_to = path_to
    def copy_to(self):
        copy(self.path_from,self.path_to)
        print('复制成功')

if __name__ == '__main__':
    path = 'D:\laboratory\ecg_com'
    dele_file(path,n = 2)



