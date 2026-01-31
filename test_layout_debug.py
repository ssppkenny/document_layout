from shapely.geometry import box
from shapely.ops import unary_union
import networkx as nx
from collections import defaultdict

# Test data from user
box_coords = [
    (1339.399169921875, 1439.374267578125, 11.946115493774414, 1661.189453125),
    (1338.629638671875, 981.1558837890625, 13.27794361114502, 1202.9298095703125),
    (1340.7044677734375, 100.22525024414062, 10.679261207580566, 284.215087890625),
    (1339.2156982421875, 1665.9715576171875, 13.355371475219727, 1796.74755859375),
    (1338.988525390625, 285.3494873046875, 12.128990173339844, 419.837890625),
    (585.4254150390625, 449.7438049316406, 225.81094360351562, 761.110107421875),  # figure 1
    (1123.9862060546875, 449.6356201171875, 767.9382934570312, 764.2161865234375),  # figure 2
    (903.2875366210938, 2050.45361328125, 436.7130126953125, 2136.911865234375),
    (908.178466796875, 1873.9114990234375, 440.8843078613281, 1962.5882568359375),
    (1337.1124267578125, 886.4088745117188, 12.824210166931152, 978.9760131835938),
    (941.9853515625, 1220.684814453125, 413.5063171386719, 1275.734130859375),
    (621.6704711914062, 1304.0303955078125, 80.06095886230469, 1346.42822265625),
    (906.3920288085938, 1360.86279296875, 450.265869140625, 1410.4779052734375),
    (472.1832275390625, 791.2596435546875, 347.51092529296875, 830.4391479492188),  # caption 1
    (1154.8177490234375, 1984.8465576171875, 76.11897277832031, 2034.284912109375),
    (1002.564208984375, 789.9856567382812, 877.0025634765625, 830.7816772460938),  # caption 2
    (1339.138671875, 1887.787109375, 1282.4029541015625, 1937.62255859375),
    (1340.8890380859375, 10.036866188049316, 1299.8629150390625, 46.01227569580078),
    (1337.283447265625, 2064.782958984375, 1280.6611328125, 2115.208740234375),
    (68.1417007446289, 20.26910972595215, 11.516654014587402, 56.23842239379883),
    (1161.931884765625, 10.273686408996582, 192.83705139160156, 48.40536117553711),
    (1152.446533203125, 1809.9078369140625, 89.29908752441406, 1857.168701171875),
    (1181.55615234375, 1807.5421142578125, 75.62442779541016, 1858.030029296875),
    (1151.79296875, 1985.663330078125, 89.50835418701172, 2034.095947265625)
]

types = [
    'plain text', 'plain text', 'plain text', 'plain text', 'plain text',
    'figure', 'figure',  # indices 5, 6
    'isolate_formula', 'isolate_formula', 'plain text', 'isolate_formula',
    'plain text', 'isolate_formula',
    'figure_caption', 'plain text', 'figure_caption',  # indices 13, 15
    'formula_caption', 'abandon', 'formula_caption', 'abandon', 'abandon',
    'plain text', 'plain text', 'plain text'
]

# Create actual box objects - note the coordinate order should be (minx, miny, maxx, maxy)
boxes = []
for coords in box_coords:
    x1, y1, x2, y2 = coords
    minx = min(x1, x2)
    maxx = max(x1, x2)
    miny = min(y1, y2)
    maxy = max(y1, y2)
    boxes.append(box(minx, miny, maxx, maxy))

print("Figure boxes:")
print(f"  Index 5 (figure): {boxes[5]}")
print(f"  Index 6 (figure): {boxes[6]}")
print("\nCaption boxes:")
print(f"  Index 13 (caption): {boxes[13]}")
print(f"  Index 15 (caption): {boxes[15]}")

print("\nDistances:")
print(f"  Caption 13 to Figure 5: {boxes[13].centroid.distance(boxes[5].centroid)}")
print(f"  Caption 13 to Figure 6: {boxes[13].centroid.distance(boxes[6].centroid)}")
print(f"  Caption 15 to Figure 5: {boxes[15].centroid.distance(boxes[5].centroid)}")
print(f"  Caption 15 to Figure 6: {boxes[15].centroid.distance(boxes[6].centroid)}")

print("\n\nFormula boxes:")
print(f"  Index 7 (formula): {boxes[7]}")
print(f"  Index 8 (formula): {boxes[8]}")
print(f"  Index 10 (formula): {boxes[10]}")
print(f"  Index 12 (formula): {boxes[12]}")
print("\nFormula Caption boxes:")
print(f"  Index 16 (formula_caption): {boxes[16]}")
print(f"  Index 18 (formula_caption): {boxes[18]}")

print("\nFormula-Caption Distances:")
for formula_idx in [7, 8, 10, 12]:
    for caption_idx in [16, 18]:
        dist = boxes[caption_idx].centroid.distance(boxes[formula_idx].centroid)
        print(f"  Formula caption {caption_idx} to Formula {formula_idx}: {dist}")


def find_grouped_bounding_boxes(boxes, types):
    # Step 1: Index boxes by type
    type_to_indices = defaultdict(list)
    for idx, t in enumerate(types):
        type_to_indices[t].append(idx)

    result = []

    # Step 2: Handle "figure" and "figure_caption"
    figures = set(type_to_indices.get("figure", []))
    captions = set(type_to_indices.get("figure_caption", []))

    print(f"\nFigure indices: {figures}")
    print(f"Caption indices: {captions}")

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
            print(f"  Caption {cap_idx} to Figure {fig_idx}: distance = {dist}")
            if dist < min_dist:
                min_dist = dist
                nearest_fig = fig_idx
        if nearest_fig is not None:
            print(f"  -> Pairing caption {cap_idx} with figure {nearest_fig}")
            union = unary_union([boxes[nearest_fig], boxes[cap_idx]])
            result.append((box(*union.bounds), "figure and caption"))
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

    print(f"\nFormula indices: {formulas}")
    print(f"Formula caption indices: {formula_captions}")

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
            print(f"  Formula caption {cap_idx} to Formula {formula_idx}: distance = {dist}")
            if dist < min_dist:
                min_dist = dist
                nearest_formula = formula_idx
        if nearest_formula is not None:
            print(f"  -> Pairing formula caption {cap_idx} with formula {nearest_formula}")
            union = unary_union([boxes[nearest_formula], boxes[cap_idx]])
            result.append((box(*union.bounds), "isolate_formula and caption"))
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

result = find_grouped_bounding_boxes(boxes, types)

print("\n\nResults:")
for r in result:
    print(f"  {r[1]}: {r[0]}")

print(f"\n\nTotal results: {len(result)}")
figure_and_caption_count = sum(1 for r in result if r[1] == "figure and caption")
formula_and_caption_count = sum(1 for r in result if r[1] == "isolate_formula and caption")
print(f"Figure and caption pairs: {figure_and_caption_count}")
print(f"Formula and caption pairs: {formula_and_caption_count}")
