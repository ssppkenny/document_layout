from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from shapely.geometry import box
from shapely.ops import unary_union
import networkx as nx
from collections import defaultdict

from torch.xpu import device

filepath = hf_hub_download(repo_id="juliozhao/DocLayout-YOLO-DocStructBench", filename="doclayout_yolo_docstructbench_imgsz1024.pt")
model = YOLOv10(filepath)

def find_grouped_bounding_boxes(boxes, types):
    # Step 1: Index boxes by type
    type_to_indices = defaultdict(list)
    for idx, t in enumerate(types):
        type_to_indices[t].append(idx)

    result = []

    # Step 2: Handle "figure" and "figure_caption"
    figures = set(type_to_indices.get("figure", []))
    captions = set(type_to_indices.get("figure_caption", []))
    used_figures = set()
    used_captions = set()

    # Pair each caption to the nearest unpaired figure
    for cap_idx in captions:
        cap_box = boxes[cap_idx]
        min_dist = float("inf")
        nearest_fig = None
        for fig_idx in figures - used_figures:
            fig_box = boxes[fig_idx]
            dist = cap_box.centroid.distance(fig_box.centroid)
            if dist < min_dist:
                min_dist = dist
                nearest_fig = fig_idx
        if nearest_fig is not None:
            union = unary_union([boxes[nearest_fig], boxes[cap_idx]])
            result.append((box(*union.bounds), "figure_and_caption"))
            used_figures.add(nearest_fig)
            used_captions.add(cap_idx)

    # Add unpaired figures
    for fig_idx in figures - used_figures:
        result.append((boxes[fig_idx], "figure"))

    # Add unpaired captions
    for cap_idx in captions - used_captions:
        result.append((boxes[cap_idx], "figure_caption"))

    # Step 3: Handle "isolate_formula" and "formula_caption"
    formulas = set(type_to_indices.get("isolate_formula", []))
    formula_captions = set(type_to_indices.get("formula_caption", []))
    used_formulas = set()
    used_formula_captions = set()

    # Pair each formula caption to the nearest unpaired formula
    for cap_idx in formula_captions:
        cap_box = boxes[cap_idx]
        min_dist = float("inf")
        nearest_formula = None
        for formula_idx in formulas - used_formulas:
            formula_box = boxes[formula_idx]
            dist = cap_box.centroid.distance(formula_box.centroid)
            if dist < min_dist:
                min_dist = dist
                nearest_formula = formula_idx
        if nearest_formula is not None:
            union = unary_union([boxes[nearest_formula], boxes[cap_idx]])
            result.append((box(*union.bounds), "isolate_formula_and_caption"))
            used_formulas.add(nearest_formula)
            used_formula_captions.add(cap_idx)

    # Add unpaired formulas
    for formula_idx in formulas - used_formulas:
        result.append((boxes[formula_idx], "isolate_formula"))

    # Add unpaired formula captions
    for cap_idx in formula_captions - used_formula_captions:
        result.append((boxes[cap_idx], "formula_caption"))

    # Step 4: Group "plain text" boxes by intersection
    plaintext_indices = type_to_indices.get("plain text", [])
    if plaintext_indices:
        G = nx.Graph()
        for i in plaintext_indices:
            G.add_node(i)
        for i_idx, i in enumerate(plaintext_indices):
            for j in plaintext_indices[i_idx + 1:]:
                if boxes[i].intersects(boxes[j]):
                    G.add_edge(i, j)
        for component in nx.connected_components(G):
            subset = [boxes[i] for i in component]
            union = unary_union(subset)
            result.append((box(*union.bounds), "plain text"))

    # Step 5: Add all other types as individual boxes
    for t, indices in type_to_indices.items():
        if t in {"figure", "figure_caption", "isolate_formula", "formula_caption", "plain text"}:
            continue
        for idx in indices:
            result.append((boxes[idx], t))

    return result

def layout(image_path):
    device = "cuda:0" if model.device.type == "cuda" else "cpu"
    det_res = model.predict(
        image_path,  # Image to predict
        imgsz=1024,  # Prediction image size
        conf=0.2,  # Confidence threshold
        device=device
    )
    names = det_res[0].names
    blocknames = [names[int(n)] for n in det_res[0].boxes.cls]
    xyxy = [a.tolist() for a in det_res[0].boxes.xyxy]
    # Ensure coordinates are in correct order: (minx, miny, maxx, maxy)
    rect_list = []
    for x1, y1, x2, y2 in xyxy:
        minx = min(x1, x2)
        maxx = max(x1, x2)
        miny = min(y1, y2)
        maxy = max(y1, y2)
        rect_list.append(box(minx, miny, maxx, maxy))
    return find_grouped_bounding_boxes(rect_list, blocknames)

