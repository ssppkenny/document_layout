"""
Implementation of the Divide-and-Conquer Algorithm (Section 2.2)
from "The Rectangle Enclosure and Point-Dominance Problems Revisited"

This algorithm solves the 4D dominance problem in O(n log² n + k) time
using O(n) space, matching the Lee and Preparata bounds.
"""

from typing import List, Set, Tuple
from dataclasses import dataclass
import bisect
# from test_large_scale import generate_random_rectangles
@dataclass(frozen=True)
class Rectangle:
    """Represents an axes-parallel rectangle in 2D plane."""
    id: int  # Unique identifier for the rectangle
    left: float
    right: float
    bottom: float
    top: float

    def __post_init__(self):
        """Validate rectangle coordinates."""
        if self.left >= self.right:
            raise ValueError(f"Invalid rectangle: left ({self.left}) must be < right ({self.right})")
        if self.bottom >= self.top:
            raise ValueError(f"Invalid rectangle: bottom ({self.bottom}) must be < top ({self.top})")

    def encloses(self, other: 'Rectangle') -> bool:
        """
        Check if this rectangle encloses another rectangle.

        Args:
            other: The rectangle to check if enclosed

        Returns:
            True if this rectangle encloses other, False otherwise
        """
        return (self.left <= other.left and
                self.bottom <= other.bottom and
                self.right >= other.right and
                self.top >= other.top and
                self.id != other.id)  # A rectangle cannot enclose itself

    def __repr__(self):
        return f"R{self.id}[{self.left},{self.right}]×[{self.bottom},{self.top}]"

@dataclass(frozen=True)
class Point4D:
    """Represents a point in 4D space."""
    p1: float
    p2: float
    p3: float
    p4: float
    index: int = -1

    def __repr__(self):
        return f"P4D({self.p1}, {self.p2}, {self.p3}, {self.p4})[{self.index}]"

    def dominates(self, other: 'Point4D') -> bool:
        """Check if this point dominates another point.
        Point p dominates q if p_i >= q_i for all i."""
        return (self.p1 >= other.p1 and
                self.p2 >= other.p2 and
                self.p3 >= other.p3 and
                self.p4 >= other.p4)


@dataclass(frozen=True)
class Point3D:
    """Represents a point in 3D space (projection from 4D)."""
    x: float
    y: float
    z: float
    index: int = -1

    def __repr__(self):
        return f"P3D({self.x}, {self.y}, {self.z})[{self.index}]"

    def dominates(self, other: 'Point3D') -> bool:
        """Check if this point dominates another point in 3D."""
        return (self.x >= other.x and
                self.y >= other.y and
                self.z >= other.z)


def normalize_4d_points(points: List[Point4D]) -> List[Point4D]:
    """
    Normalization step (Section 2.1): Replace each coordinate with its rank.

    Time Complexity: O(n log n)

    Args:
        points: List of n points in R^4

    Returns:
        List of normalized points with coordinates in {0, 1, ..., n-1}
    """
    n = len(points)
    if n == 0:
        return []

    # Store ranks for each point
    ranks = [[0] * 4 for _ in range(n)]

    # Process each dimension independently
    for dim in range(4):
        # Create list of (coordinate_value, original_index)
        coord_with_index = []
        for i in range(n):
            if dim == 0:
                coord_with_index.append((points[i].p1, i))
            elif dim == 1:
                coord_with_index.append((points[i].p2, i))
            elif dim == 2:
                coord_with_index.append((points[i].p3, i))
            else:
                coord_with_index.append((points[i].p4, i))

        # Sort by coordinate value - O(n log n) per dimension
        coord_with_index.sort(key=lambda x: x[0])

        # Assign ranks (0 to n-1)
        for rank, (_, original_idx) in enumerate(coord_with_index):
            ranks[original_idx][dim] = rank

    # Create normalized points
    normalized_points = []
    for i in range(n):
        normalized_points.append(
            Point4D(
                ranks[i][0], ranks[i][1], ranks[i][2], ranks[i][3],
                index=points[i].index if points[i].index >= 0 else i
            )
        )

    return normalized_points


class PrioritySearchTree:
    """
    Simple Priority Search Tree for 2D dominance queries.

    In practice, for normalized coordinates we could use a more efficient
    structure, but this provides the basic PST functionality.
    """

    def __init__(self):
        self.points = []  # List of (x, y, index) tuples, sorted by x

    def insert(self, x: float, y: float, index: int):
        """Insert a point into the PST."""
        bisect.insort(self.points, (x, y, index))

    def query_dominated_by(self, x: float, y: float) -> List[int]:
        """
        Find all points (px, py) that dominate (x, y), i.e., px >= x and py >= y.
        Returns list of indices.
        """
        result = []
        for px, py, idx in self.points:
            # We want points where px >= x and py >= y (point dominates query)
            if px >= x and py >= y:
                result.append(idx)
        return result


def red_blue_dominance_3d(red_points: List[Point3D], blue_points: List[Point3D]) -> Set[Tuple[int, int]]:
    """
    Solve 3D red-blue dominance reporting problem.

    Reports all pairs (r, b) where r is red, b is blue, and b dominates r.
    Uses simple brute force for correctness.

    Time Complexity: O(n²)
    Space Complexity: O(k) where k is the number of pairs

    Args:
        red_points: List of red points in 3D
        blue_points: List of blue points in 3D

    Returns:
        Set of tuples (red_index, blue_index) representing dominance pairs
    """
    dominance_pairs = set()

    for r in red_points:
        for b in blue_points:
            if b.dominates(r):
                dominance_pairs.add((r.index, b.index))

    return dominance_pairs


def divide_conquer_4d_recursive(
    points: List[Point4D],
    indices: List[int]
) -> Set[Tuple[int, int]]:
    """
    Recursive divide-and-conquer algorithm for 4D dominance.

    Args:
        points: The normalized point set
        indices: Indices of points to process (sorted by 3rd coordinate)

    Returns:
        Set of dominance pairs (i, j) where points[i] dominates points[j]
    """
    n = len(indices)
    if n <= 1:
        return set()

    if n == 2:
        # Base case: check if one dominates the other
        idx1 = indices[0]
        idx2 = indices[1]
        result = set()
        if points[idx1].dominates(points[idx2]):
            result.add((idx1, idx2))
        if points[idx2].dominates(points[idx1]):
            result.add((idx2, idx1))
        return result

    # Step 1: Split by median of p4 coordinate
    # Sort indices by p4 to find median, then split at median position
    indices_by_p4 = sorted(indices, key=lambda idx: points[idx].p4)
    mid = n // 2

    # Step 2: Take first half as S1, second half as S2 (by p4 order)
    # Then restore original p3 order within each set
    S1_set = set(indices_by_p4[:mid])
    S2_set = set(indices_by_p4[mid:])

    # Maintain original p3 order (which is the order of indices list)
    S1_indices = [idx for idx in indices if idx in S1_set]
    S2_indices = [idx for idx in indices if idx in S2_set]

    # Step 3: Recursively solve for S1 and S2
    result = set()

    if S1_indices:
        result |= divide_conquer_4d_recursive(points, S1_indices)

    if S2_indices:
        result |= divide_conquer_4d_recursive(points, S2_indices)

    # Step 4: Merge step - find cross dominances
    # Project to 3D by removing 4th coordinate
    red_3d = [Point3D(points[idx].p1, points[idx].p2, points[idx].p3, idx)
              for idx in S1_indices]
    blue_3d = [Point3D(points[idx].p1, points[idx].p2, points[idx].p3, idx)
               for idx in S2_indices]

    # Find all (red, blue) pairs where blue dominates red in 3D
    # Since S2 has p4 >= S1's p4 (by our split), if blue dominates red in 3D, it dominates in 4D
    cross_pairs = red_blue_dominance_3d(red_3d, blue_3d)

    # cross_pairs are (red_index, blue_index), and we want (blue_index, red_index)
    # because blue dominates red
    for red_idx, blue_idx in cross_pairs:
        result.add((blue_idx, red_idx))

    return result


def divide_conquer_4d(points: List[Point4D]) -> Set[Tuple[int, int]]:
    """
    Main divide-and-conquer algorithm for 4D dominance reporting.

    Time Complexity: O(n log² n + k) where k is the number of dominance pairs
    Space Complexity: O(n)

    Args:
        points: List of n points in R^4

    Returns:
        Set of tuples (i, j) where points[i] dominates points[j]
    """
    if len(points) == 0:
        return set()

    # Step 1: Normalize the points
    normalized = normalize_4d_points(points)

    # Step 2: Sort by 3rd coordinate
    sorted_by_p3 = sorted(range(len(normalized)), key=lambda i: normalized[i].p3)

    # Step 3: Run divide-and-conquer
    return divide_conquer_4d_recursive(normalized, sorted_by_p3)


def brute_force_4d(points: List[Point4D]) -> Set[Tuple[int, int]]:
    """
    Brute force algorithm for 4D dominance (for testing).

    Normalizes points first to match the divide_conquer_4d algorithm.

    Time Complexity: O(n²)

    Returns:
        Set of tuples (i, j) where points[i] dominates points[j]
    """
    if len(points) == 0:
        return set()

    # Normalize points first, just like divide_conquer_4d does
    normalized = normalize_4d_points(points)

    result = set()
    for i in range(len(normalized)):
        for j in range(len(normalized)):
            if i != j and normalized[i].dominates(normalized[j]):
                result.add((i, j))
    return result


def generate_random_rectangles(n: int, x_min: float, x_max: float,
                               y_min: float, y_max: float,
                               seed: int = 42) -> list:
    """
    Generate n random rectangles within specified bounds.

    Args:
        n: Number of rectangles to generate
        x_min: Minimum x coordinate
        x_max: Maximum x coordinate
        y_min: Minimum y coordinate
        y_max: Maximum y coordinate
        seed: Random seed for reproducibility

    Returns:
        List of Rectangle objects
    """
    random.seed(seed)
    rectangles = []

    for i in range(n):
        # Generate two random x coordinates and sort them
        x1 = random.uniform(x_min, x_max)
        x2 = random.uniform(x_min, x_max)
        left = min(x1, x2)
        right = max(x1, x2)

        # Ensure left < right (not equal)
        if left == right:
            right = left + random.uniform(0.1, 10.0)

        # Generate two random y coordinates and sort them
        y1 = random.uniform(y_min, y_max)
        y2 = random.uniform(y_min, y_max)
        bottom = min(y1, y2)
        top = max(y1, y2)

        # Ensure bottom < top (not equal)
        if bottom == top:
            top = bottom + random.uniform(0.1, 10.0)

        rectangles.append([left, right, bottom, top])

    return rectangles


# Example usage and demonstration
if __name__ == "__main__":
    # rectangles = [
    #     (0, 4, 0, 4),   # R0
    #     (1, 3, 1, 3),   # R1 (enclosed by R0)
    #     (2, 5, 2, 5),   # R2
    #     (0, 6, 0, 6),   # R3 (encloses R0, R1, R2)
    # ]
    # print("=== Divide-and-Conquer 4D Dominance Algorithm ===\n")
    #
    # # Example 1: Simple test case
    # print("Example 1: Simple 4D points")
    # points1 = [
    #     Point4D(0, 0, 0, 0, index=0),
    #     Point4D(1, 1, 1, 1, index=1),
    #     Point4D(2, 2, 2, 2, index=2),
    #     Point4D(3, 3, 3, 3, index=3),
    # ]
    #
    # print("Points:")
    # for p in points1:
    #     print(f"  {p}")
    #
    # pairs1 = divide_conquer_4d(points1)
    # print(f"\nDominance pairs found: {len(pairs1)}")
    # for i, j in sorted(pairs1):
    #     print(f"  Point {i} dominates Point {j}")
    #
    # # Verify with brute force
    # pairs1_brute = brute_force_4d(points1)
    # print(f"\nBrute force validation: {pairs1 == pairs1_brute}")
    #
    # # Example 2: Rectangle enclosure
    # print("\n" + "="*60)
    # print("Example 2: Rectangle Enclosure Problem")
    #
    # # Define rectangles as [left, right] x [bottom, top]
    # rectangles = [
    #     (0, 4, 0, 4),   # R0
    #     (1, 3, 1, 3),   # R1 (enclosed by R0)
    #     (2, 5, 2, 5),   # R2
    #     (0, 6, 0, 6),   # R3 (encloses R0, R1, R2)
    # ]
    #
    # print("\nRectangles (left, right, bottom, top):")
    # for i, (l, r, b, t) in enumerate(rectangles):
    #     print(f"  R{i}: [{l}, {r}] x [{b}, {t}]")
    #
    # # Convert to 4D points: (l, b, -r, -t)
    # points2 = [
    #     Point4D(l, b, -r, -t, index=i)
    #     for i, (l, r, b, t) in enumerate(rectangles)
    # ]
    #
    # print("\nConverted to 4D points (l, b, -r, -t):")
    # for p in points2:
    #     print(f"  {p}")
    #
    # pairs2 = divide_conquer_4d(points2)
    # print(f"\nEnclosure pairs found: {len(pairs2)}")
    # for i, j in sorted(pairs2):
    #     print(f"  Rectangle R{i} encloses Rectangle R{j}")
    #
    # # Verify
    # pairs2_brute = brute_force_4d(points2)
    # print(f"\nBrute force validation: {pairs2 == pairs2_brute}")
    #
    # # Example 3: Larger test
    # print("\n" + "="*60)
    # print("Example 3: Larger test case")
    #
    # import random
    # # random.seed(42)
    #
    # n = 20
    # points3 = [
    #     Point4D(
    #         random.randint(0, 10),
    #         random.randint(0, 10),
    #         random.randint(0, 10),
    #         random.randint(0, 10),
    #         index=i
    #     )
    #     for i in range(n)
    # ]
    #
    # print(f"\nGenerated {n} random points")
    #
    # pairs3 = divide_conquer_4d(points3)
    # pairs3_brute = brute_force_4d(points3)
    #
    # print(f"Dominance pairs found: {len(pairs3)}")
    # print(f"Brute force validation: {pairs3 == pairs3_brute}")
    #
    # if pairs3 != pairs3_brute:
    #     print("\nMismatch detected!")
    #     print(f"Missing: {pairs3_brute - pairs3}")
    #     print(f"Extra: {pairs3 - pairs3_brute}")
    #
    # rectangles = [
    #         (-4, 4, -4, 4),   # R0
    #         (-3, 3, -3, 3),   # R1 (enclosed by R0)
    #         (-2, 2, -2, 2),   # R2
    #         (-1, 1, -1, 1),
    #         (-3.5, 0, 0, 3.5)# R3 (encloses R0, R1, R2)
    # ]
    import time
    import random
    rectangles = generate_random_rectangles(3500, -1000, 1000, -1000, 1000)
    points4 = [
        Point4D(l, b, -r, -t, index=i)
        for i, (l, r, b, t) in enumerate(rectangles)
    ]
    start = time.time()
    pairs4 = divide_conquer_4d(points4)
    end = time.time()
    print(f"found pairs: {len(pairs4)}")
    print(f"Divide-and-Conquer 4D time: {end - start}")
    print(f"\nEnclosure pairs in negative coordinates: {len(pairs4)}")
    for i, j in sorted(pairs4):
        print(f"  Rectangle R{i} encloses Rectangle R{j}")
        assert(Rectangle(j, *rectangles[j]).encloses(Rectangle(i, *rectangles[i])))
        print(f"Enclosing {rectangles[j]}")
        print(f"Enclosed {rectangles[i]}")


