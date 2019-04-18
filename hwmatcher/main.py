import os
import sys
import cv2
from argparse import ArgumentParser
from collections import namedtuple
from recordtype import recordtype
import munkres
import numpy as np

def add_paths():
    THIRD_PARTY_ROOT = 'third-party'
    sys.path.insert(0, os.path.join(THIRD_PARTY_ROOT, 'EAST'))
    sys.path.insert(0, os.path.join(THIRD_PARTY_ROOT, 'hwnet'))

add_paths()
from east import EASTWrapper
from utils import HWNetInferenceEngine
import torch
from torch.nn.modules.distance import CosineSimilarity
from munkres import Munkres



class Comparator:
    def __init__(self, east_path, hwnet_path):
        self.eastw = EASTWrapper(checkpoint_path=east_path)
        self.hwnet = HWNetInferenceEngine(hwnet_path)

    def __call__(self, image_A, image_B):
        Sample = recordtype('Sample', 'image bboxes features')
        def sample(image):
            _image, unit_bboxes = self.eastw.predict(image)
            return Sample(image=image, bboxes=unit_bboxes, features=None)

        a = sample(image_A)
        b = sample(image_B)
        a.features = self.hwnet(a)
        b.features = self.hwnet(b)
        with torch.no_grad():
            matrix = self.compute_cost_matrix(a.features, b.features)
            matrix = 1 - matrix

        matrix = matrix.tolist()
        m = Munkres()
        indexes = m.compute(matrix)
        total = 0
        for row, column in indexes:
            value = matrix[row][column]
            total += value
            print(row, column, '->', value)
            # print(f'({row}, {column}) -> {value}')
        # print(f'total cost: {total}')
        print('total cost', total)

            

    def compute_cost_matrix(self, A, B):
        similarity = CosineSimilarity()
        num_A, _ = A.size()
        num_B, _ = B.size()
        _matrix = np.zeros((num_A, num_B))
        for i in range(num_A):
            vA = A[i, :].unsqueeze(0)
            vA = vA.cuda()
            for j in range(num_B):
                vB = B[j, :].unsqueeze(0)
                vB = vB.cuda()
                value = similarity(vA, vB)
                _matrix[i, j] = value.item()
        return _matrix






if __name__ == '__main__':
    east_path = '/ssd_scratch/cvit/jerin/acl-workspace/east_icdar2015_resnet_v1_50_rbox/'
    hwnet_path = os.path.join('third-party/hwnet/pretrained', 'iam-morph0.t7')
    compare = Comparator(east_path, hwnet_path)
    images = []
    for i in range(2):
        image = cv2.imread("data/sample-page-{}.png".format(i))
        images.append(image)
        # cv2.imwrite("data/bbox-sample-page-{}.png".format(i), image)

    compare(images[0], images[1])


