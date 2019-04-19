import os
import sys
import cv2
from argparse import ArgumentParser
from collections import namedtuple
from recordtype import recordtype
import munkres
import numpy as np
import random

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
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import patches

def plot(a, b, indices):
    """ Plots connections with matplotlib """

    ha, wa, ca = a.image.shape
    hb, wb, cb = b.image.shape

    assert(ca == cb)

    buffer_width = 20
    W = wa + buffer_width + wb
    H = max(ha, hb)

    offset_x, offset_y = wa + buffer_width, 0

    C = np.full((H, W, ca), 255, dtype=a.image.dtype)
    C[:ha, :wa] = a.image.copy()
    C[offset_y:offset_y+hb, offset_x:offset_x+wb] = b.image.copy()
    plt.figure(figsize=(21, 15), dpi=300)
    plt.imshow(C)
    axes = plt.gca()

    def draw(bboxes, offset_x, offset_y, color):
        for bbox in bboxes:
            x, y, X, Y = (
                    bbox.x + offset_x,
                    bbox.y + offset_y,
                    bbox.X + offset_x,
                    bbox.Y + offset_y
            )
            rect = patches.Rectangle(
                (x, y), 
                (X-x+1), (Y-y+1),
                linewidth=1,
                edgecolor=color,
                facecolor='none'
            )
            axes.add_patch(rect)

    draw(a.bboxes, 0, 0, color='r')
    draw(b.bboxes, offset_x, offset_y, color='g')

    def draw_arrows(ba, bb):
        dx = offset_x + bb.x - ba.X
        dy = offset_y + bb.y - ba.Y
        plt.arrow(ba.X, ba.Y, dx, dy, alpha=0.3)

    # indices = random.sample(indices, 25)
    for index in indices:
        f, s = index
        draw_arrows(a.bboxes[f], b.bboxes[s])

    plt.savefig('data/matching.png')





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
        # total = 0
        # for row, column in indexes:
        #     value = matrix[row][column]
        #     total += value
        #     print(row, column, '->', value)
        #     # print(f'({row}, {column}) -> {value}')
        # # print(f'total cost: {total}')
        # print('total cost', total)
        plot(a, b, indexes)
            

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


