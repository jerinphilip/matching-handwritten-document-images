import os
import sys
import cv2
from argparse import ArgumentParser
from collections import namedtuple
from recordtype import recordtype


def add_paths():
    THIRD_PARTY_ROOT = 'third-party'
    sys.path.insert(0, os.path.join(THIRD_PARTY_ROOT, 'EAST'))
    sys.path.insert(0, os.path.join(THIRD_PARTY_ROOT, 'hwnet'))

add_paths()
from east import EASTWrapper
from utils import HWNetInferenceEngine

class Comparator:
    def __init__(self, east_path, hwnet_path):
        self.eastw = EASTWrapper(checkpoint_path=east_path)
        self.hwnet = HWNetInferenceEngine(hwnet_path)

    def __call__(self, image_A, image_B):
        Sample = recordtype('Sample', 'image bboxes features')
        def sample(image):
            _image, unit_bboxes = self.eastw.predict(image_A)
            return Sample(image=image, bboxes=unit_bboxes, features=None)

        a = sample(image_A)
        b = sample(image_B)
        a.features = self.hwnet(a)
        b.features = self.hwnet(b)
        # print(len(a.bboxes), len(b.bboxes))


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


