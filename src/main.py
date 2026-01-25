import numpy as np
from math import ceil
import sys
from doctr.models import (
    detection_predictor,
)
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from doctr.io import DocumentFile
from shapely import LineString, box
import shapely
from operator import itemgetter
from dataclasses import dataclass
from reflow import create_page_with_word_wrapping
from divide_conquer_4d import divide_conquer_4d, Point4D

@dataclass
class Letter:
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    bl: int


def find_rects(img, line_words):
    rects = []
    for xmin,ymin,xmax,ymax in line_words:
        r = img[ymin:ymax,xmin:xmax,:].copy()
        r = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        _, r = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(r, 8, cv2.CV_32S)
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            rects.append((x+xmin,y+ymin,x+w+xmin,y+h+ymin))
    rectangles = [(int(xmin), int(xmax), int(ymin), int(ymax)) for xmin, ymin, xmax, ymax in rects]

    points4 = [
        Point4D(l, b, -r, -t, index=i)
        for i, (l, r, b, t) in enumerate(rectangles)
    ]
    pairs = divide_conquer_4d(points4)
    ind_to_remove = [i for i, j in sorted(pairs)]
    # for i, j in sorted(pairs):
    #     print(f"  Rectangle R{i} encloses Rectangle R{j}")
    #     print(f"Enclosing {rectangles[j]}")
    #     print(f"Enclosed {rectangles[i]}")
    # print(rects)

    rects = [v for i, v in enumerate(rects) if i not in ind_to_remove]
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
    words[:, 1] = (words[:, 1] * img_h).astype(np.int32) + 2
    words[:, 2] = (words[:, 2] * img_w).astype(np.int32)
    words[:, 3] = (words[:, 3] * img_h).astype(np.int32) - 2
    words = words.astype(np.int32)

    img = cv2.imread(filename)
    img1 = cv2.imread(filename)
    img2 = cv2.imread(filename)
    left_margins, right_margins = margins(words)

    rectangles = dict([(box(xmin, ymin, xmax, ymax), (int(xmin), int(ymin), int(xmax), int(ymax))) for (xmin, ymin, xmax, ymax, p) in words])

    lines = []
    for l,r in zip(left_margins, right_margins):
        line = LineString([(l[0], l[1]), (r[0], r[1])])
        line_words = []
        for b in rectangles:
            if line.intersects(b):
                line_words.append(rectangles[b])
        lw = line_words.copy()
        for xmin, ymin, xmax,ymax in lw:
            cv2.rectangle(img2, (xmin,ymin), (xmax, ymax), (255,0,0), 1)
        lines.append(sorted(lw))

    # Configuration parameters moved outside the loop
    zoom_factor = 2.5
    new_page_width = 2000

    # Detect background color from the original image
    # Use the median color value of the image as background
    # This works well for documents with light backgrounds
    flat_img = img.reshape(-1, 3)
    background_color = np.median(flat_img, axis=0).astype(np.uint8)
    print(f"Detected background color (BGR): {background_color}")

    all_letters = []
    all_lines = []
    for ln ,line in enumerate(lines):
        line_letters = find_rects(img, line)
        line_letters = sorted(line_letters, key=itemgetter(0))
        heights = [ymax - ymin for xmin,ymin,xmax,ymax in line_letters]
        m_height = np.median(heights)
        values, counts = np.unique(heights, return_counts=True)
        fh = values[np.argmax(counts)]
        sd = np.std(heights)
        normal_letters = [(xmin,ymin,xmax,ymax) for xmin,ymin,xmax,ymax in line_letters if abs((ymax-ymin)-m_height) < sd]
        lower_points = [((xmin+xmax)/2,ymax) for xmin,ymin,xmax,ymax in normal_letters]
        try:
            x_coords = [x for x,y in lower_points]
            y_coords = [y for x,y in lower_points]
            m, c = np.polyfit(x_coords, y_coords, 1)
            # cv2.line(img, (int(x_coords[0]), int(m*x_coords[0]+c)), (int(x_coords[-1]), int(np.ceil(m*x_coords[-1]+c))), (255,0,0), 2)
        except:
            m, c = 0, 0
        letters = [Letter(xmin,ymin,xmax,ymax,ymax-ceil(m*((xmin+xmax)/2)+c)) for xmin,ymin,xmax,ymax in line_letters]
        all_letters.extend(letters)
        all_lines.append(letters)
       
        red = (255,0,0)
        green = (0,255,0)
        for l in letters:
            if ln%2 == 0:
                cv2.rectangle(img1, (l.xmin,l.ymin), (l.xmax, l.ymax), red, 1)
            else:
                cv2.rectangle(img1, (l.xmin,l.ymin), (l.xmax, l.ymax), green, 1)

    page_with_letters = create_page_with_word_wrapping(all_lines, img, zoom_factor, new_page_width, background_color=tuple(background_color))
    cv2.imwrite("out.png", page_with_letters)
    cv2.imwrite("out1.png", img1)
    cv2.imwrite("out2.png", img2)
    plt.imshow(page_with_letters)
    plt.show()

