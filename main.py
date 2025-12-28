import numpy as np
import sys
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
from shapely import LineString, box
import shapely
from operator import itemgetter

def find_baseline(img, line_words):
    rects = []
    for xmin,ymin,xmax,ymax in line_words:
        r = img[ymin:ymax,xmin:xmax,:].copy()
        r = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        _, r = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(r, 4, cv2.CV_32S)
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            rects.append((x+xmin,y+ymin,x+w+xmin,y+h+ymin))
    return rects

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
            b1 = box(xmin1, ymin1, xmax1, ymax1)
            b2 = box(xmin, ymin, xmax, ymax)
            s = shapely.intersection(ls1, ls2)
            m = min(abs(xmin-xmax), abs(xmin1-xmax1))
            mv = min(abs(ymin-ymax), abs(ymin1-ymax1))
            if (nb[0] <= x or abs(x-nb[0]) < m/2) and not s.is_empty and (s.length > 0.6*mv):
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
            b1 = box(xmin1, ymin1, xmax1, ymax1)
            b2 = box(xmin, ymin, xmax, ymax)
            m = min(abs(xmin-xmax), abs(xmin1-xmax1))
            mv = min(abs(ymin-ymax), abs(ymin1-ymax1))
            if (nb[0] >= x or abs(x-nb[0]) < m/2) and not s.is_empty and (s.length > 0.6*mv):
                points_to_side.append((nb[0], nb[1]))
        if len(points_to_side) == 0:
            right_margin.append((int(x), int(y)))
            # cv2.rectangle(img, (int(x),int(y)), (int(x),int(y)), (255,0,0), 10)

    return sorted(left_margin, key=itemgetter(1)), sorted(
        right_margin, key=itemgetter(1)
    )

if __name__ == "__main__":
    filename = sys.argv[1]
    model = detection_predictor(pretrained=True)
    # filename = "dvurog_p007.png"
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

    rectangles = dict(
        [
            (box(xmin, ymin, xmax, ymax), (int(xmin), int(ymin), int(xmax), int(ymax)))
            for (xmin, ymin, xmax, ymax, p) in words
        ]
    )

    img = cv2.imread(filename)
    left_margins, right_margins = margins(words)

    rectangles = dict([(box(xmin, ymin, xmax, ymax), (int(xmin), int(ymin), int(xmax), int(ymax))) for (xmin, ymin, xmax, ymax, p) in words])

    lines = []
    for l,r in zip(left_margins, right_margins):
        line = LineString([l, r])
        line_words = []
        for b in rectangles:
            if line.intersects(b):
                line_words.append(rectangles[b])
        lines.append(sorted(line_words.copy()))


    for line in lines:
        line_letters = find_baseline(img, line)
        heights = [ymax - ymin for xmin,ymin,xmax,ymax in line_letters]
        m = np.median(heights)
        values, counts = np.unique(heights, return_counts=True)
        fh = values[np.argmax(counts)]
        sd = np.std(heights)
        normal_letters = [(xmin,ymin,xmax,ymax) for xmin,ymin,xmax,ymax in line_letters if abs((ymax-ymin)-m) < sd]
        lower_points = [((xmin+xmax)/2,ymax) for xmin,ymin,xmax,ymax in normal_letters]
        try:
            x = [x for x,y in lower_points]
            y = [y for x,y in lower_points]
            m, c = np.polyfit(x, y, 1)
            cv2.line(img, (int(x[0]), int(m*x[0]+c)), (int(x[-1]), int(np.ceil(m*x[-1]+c))), (255,0,0), 2)
        except:
            pass
        
    plt.imshow(img)
    plt.show()
