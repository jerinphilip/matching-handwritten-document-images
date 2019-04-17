import os
import sys
import cv2
from argparse import ArgumentParser

def add_paths():
    THIRD_PARTY_ROOT = 'third-party'
    sys.path.insert(0, os.path.join(THIRD_PARTY_ROOT, 'EAST'))

if __name__ == '__main__':
    add_paths()
    from east import EASTWrapper
    eastw = EASTWrapper(
        checkpoint_path='/ssd_scratch/cvit/jerin/acl-workspace/east_icdar2015_resnet_v1_50_rbox/'
    )

    for i in range(2):
        image = cv2.imread("data/sample-page-{}.png".format(i))
        image, unit_bboxes = eastw.predict(image)
        cv2.imwrite("data/bbox-sample-page-{}.png".format(i), image)


