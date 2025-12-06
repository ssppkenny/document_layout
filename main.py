import numpy as np
from doctr.models import (
    ocr_predictor,
    detection_predictor,
    recognition_predictor,
    kie_predictor,
)
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import doctr
from doctr.io import DocumentFile
from shapely import LineString
import shapely
from operator import itemgetter


def margins(words):
    left_margin = []
    right_margin = []
    left_points = np.array(
        [[xmin, (ymin + ymax) / 2] for xmin, ymin, xmax, ymax, _ in words]
    )
    right_points = np.array(
        [[xmax, (ymin + ymax) / 2] for xmin, ymin, xmax, ymax, _ in words]
    )

    points = np.vstack((left_points, right_points))

    left_point_to_word = dict(
        [
            ((xmin, (ymin + ymax) / 2), (xmin, ymin, xmax, ymax))
            for xmin, ymin, xmax, ymax, _ in words
        ]
    )
    right_point_to_word = dict(
        [
            ((xmax, (ymin + ymax) / 2), (xmin, ymin, xmax, ymax))
            for xmin, ymin, xmax, ymax, _ in words
        ]
    )

    point_to_word = left_point_to_word | right_point_to_word

    kdtree = KDTree(points)
    dists_left, inds_left = kdtree.query(left_points, k=50)
    dists_right, inds_right = kdtree.query(right_points, k=50)

    for nbs_inds in inds_left:
        p_ind = nbs_inds[0]
        nbs_inds = nbs_inds[1:]
        nbs = points[nbs_inds]
        x, y = points[p_ind]
        xmin1, ymin1, xmax1, ymax1 = point_to_word[(x, y)]
        points_to_side = []
        for nb in nbs:
            xmin, ymin, xmax, ymax = point_to_word[(nb[0], nb[1])]
            ls1 = LineString([(0, ymin), (0, ymax)])
            ls2 = LineString([(0, ymin1), (0, ymax1)])
            s = shapely.intersection(ls1, ls2)
            if nb[0] < x and not s.is_empty and s.length > (ymax - ymin) / 2:
                points_to_side.append((nb[0], nb[1]))
        if len(points_to_side) == 0:
            left_margin.append((int(x), int(y)))
            # cv2.rectangle(img, (int(x),int(y)), (int(x),int(y)), (255,0,0), 10)
    for nbs_inds in inds_right:
        p_ind = nbs_inds[0]
        nbs_inds = nbs_inds[1:]
        nbs = points[nbs_inds]
        x, y = points[p_ind]
        xmin1, ymin1, xmax1, ymax1 = point_to_word[(x, y)]
        points_to_side = []
        for nb in nbs:
            xmin, ymin, xmax, ymax = point_to_word[(nb[0], nb[1])]
            ls1 = LineString([(0, ymin), (0, ymax)])
            ls2 = LineString([(0, ymin1), (0, ymax1)])
            s = shapely.intersection(ls1, ls2)
            if nb[0] > x and not s.is_empty and s.length > (ymax - ymin) / 2:
                points_to_side.append((nb[0], nb[1]))
        if len(points_to_side) == 0:
            right_margin.append((int(x), int(y)))
            # cv2.rectangle(img, (int(x),int(y)), (int(x),int(y)), (255,0,0), 10)

    return sorted(left_margin, key=itemgetter(1)), sorted(
        right_margin, key=itemgetter(1)
    )


if __name__ == "__main__":
    model = detection_predictor(pretrained=True)
    filename = "dvurog_p007.png"
    docs = DocumentFile.from_images([filename])
    img = cv2.imread(filename)
    img_h, img_w, _ = img.shape
    result = model(docs)
    words = result[0]["words"]
    words[:, 0] = (words[:, 0] * img_w).astype(np.int32)
    words[:, 1] = (words[:, 1] * img_h).astype(np.int32)
    words[:, 2] = (words[:, 2] * img_w).astype(np.int32)
    words[:, 3] = (words[:, 3] * img_h).astype(np.int32)
    words = words.astype(np.int32)
    left_margin, right_margin = margins(words)
    for l, r in zip(left_margin, right_margin):
        cv2.line(img, l, r, (255, 0, 0), 5)
    plt.imshow(img)
    plt.show()
