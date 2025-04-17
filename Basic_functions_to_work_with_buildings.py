import math
from collections import defaultdict
from itertools import combinations, product

import networkx as nx
import numpy as np
from scipy.spatial import ConvexHull

epsilon = 10e-5


# Number of unique levels including 0
def nlevels(poly):
    return len(np.unique([i[3] for i in poly]))

# Number of non-zero spaces in poly

def nspaces(poly):
    # list_spaces: [1 2 3]
    list_spaces = np.unique([i[4] for i in poly])

    if 0 in list_spaces:
        nspaces = len(list_spaces) - 1
    else:
        nspaces = len(list_spaces)

    return nspaces


#List of vertices on each level

def vertices(poly):
    v = defaultdict(list)
    for i in range(nlevels(poly)):
        for j in poly:
            if j[3] == i:
                if [x for y in [j[0], [i], [j[4]]] for x in y] not in v[i]:
                    v[i].append([x for y in [j[0], [i], [j[4]]] for x in y])
                if [x for y in [j[1], [i], [j[4]]] for x in y] not in v[i]:
                    v[i].append([x for y in [j[1], [i], [j[4]]] for x in y])
                if [x for y in [j[2], [i], [j[4]]] for x in y] not in v[i]:
                    v[i].append([x for y in [j[2], [i], [j[4]]] for x in y])
    return v


# List of unique vertices on each level

def unique_v(poly):
    v = defaultdict(list)
    for i in range(nlevels(poly)):
        for j in poly:
            if j[3] == i:
                if [x for y in [j[0], [i]] for x in y] not in v[i]:
                    v[i].append([x for y in [j[0], [i]] for x in y])
                if [x for y in [j[1], [i]] for x in y] not in v[i]:
                    v[i].append([x for y in [j[1], [i]] for x in y])
                if [x for y in [j[2], [i]] for x in y] not in v[i]:
                    v[i].append([x for y in [j[2], [i]] for x in y])
    return v

#List of edges on each level

def e(poly):
    e = defaultdict(list)
    for i in range(nlevels(poly)):
        for j in poly:
            if j[3] == i:
                if ([j[0], j[1], i, j[4]] not in e[i]) and ([j[1], j[0], i, j[4]] not in e[i]):
                    e[i].append([x for y in [j[0], j[1], [i], [j[4]]] for x in y])
                if ([j[1], j[2], i, j[4]] not in e[i]) and ([j[2], j[1], i, j[4]] not in e[i]):
                    e[i].append([x for y in [j[1], j[2], [i], [j[4]]] for x in y])
                if ([j[0], j[2], i, j[4]] not in e[i]) and ([j[2], j[0], i, j[4]] not in e[i]):
                    e[i].append([x for y in [j[0], j[2], [i], [j[4]]] for x in y])
    return e


# List of unique edges on each level

def unique_e(poly):
    e = defaultdict(list)

    # i - level including zero and maximal level
    for i in range(nlevels(poly)):

        # j - cell on level i
        for j in poly:
            if j[3] == i:

                # To key i we add value [0, 0, 2, 0, level] if it's not already in e
                if ([x for y in [j[0], j[1], [i]] for x in y] not in e[i]) and (
                        [x for y in [j[1], j[0], [i]] for x in y] not in e[i]):
                    e[i].append([x for y in [j[0], j[1], [i]] for x in y])
                if ([x for y in [j[1], j[2], [i]] for x in y] not in e[i]) and (
                        [x for y in [j[2], j[1], [i]] for x in y] not in e[i]):
                    e[i].append([x for y in [j[1], j[2], [i]] for x in y])
                if ([x for y in [j[0], j[2], [i]] for x in y] not in e[i]) and (
                        [x for y in [j[2], j[0], [i]] for x in y] not in e[i]):
                    e[i].append([x for y in [j[0], j[2], [i]] for x in y])
    return e

#List of cells belonging to each space

def spaces_cells(poly):
    s = defaultdict(list)
    for i in range(nspaces(poly) + 1):
        for j in poly:
            if j[4] == i:
                s[i].append(j)
    return s

#List of edges belonging to ech space

def spaces_edges(poly):
    # s - return of the function:  space 3: edges [[0, 0, 1, 0, 1, 3], ...]
    s = defaultdict(list)

    # i goes through all occuring space numbers
    for i in np.unique([u[4] for u in poly]):

        # j - cells related to space i
        for j in poly:
            if j[4] == i:
                s[i].append([x for y in [j[0], j[1], [j[3]], [i]] for x in y])
                s[i].append([x for y in [j[1], j[2], [j[3]], [i]] for x in y])
                s[i].append([x for y in [j[0], j[2], [j[3]], [i]] for x in y])

    return s

#List of cells on each level

def spacesbylevel(poly):
    s = defaultdict(list)
    for i in range(nlevels(poly)):
        for j in poly:
            if j[3] == i:
                s[i].append(j)
    return s

#This function returns dictionary for each space with its external edges.
# So only the contour of the space on each level.

def ext_edges_spaces(poly):
    # print(poly)
    # s - space 3: edges [[0, 0, 1, 0, 1, 3], ...]
    s = spaces_edges(poly)
    # k - space number
    # print(s)
    for k in s:

        # List of indices - internal edges of space k, i.e. edges which occur more than once
        r = []

        # i, j  - two different edges in space k
        for i in range(len(s[k])):
            for j in range(len(s[k])):
                if i < j:

                    # If i = j or just vertices are swapped (the edge occured twice) => add i and j to "bad" indices
                    if ((np.abs(s[k][i][0] - s[k][j][0]) < epsilon) and (np.abs(s[k][i][1] - s[k][j][1]) < epsilon) and \
                        (np.abs(s[k][i][2] - s[k][j][2]) < epsilon) and (np.abs(s[k][i][3] - s[k][j][3]) < epsilon) and \
                        (np.abs(s[k][i][4] - s[k][j][4]) < epsilon) and (np.abs(s[k][i][5] - s[k][j][5]) < epsilon)) or \
                            ((np.abs(s[k][i][0] - s[k][j][2]) < epsilon) and (
                                    np.abs(s[k][i][1] - s[k][j][3]) < epsilon) and \
                             (np.abs(s[k][i][2] - s[k][j][0]) < epsilon) and (
                                     np.abs(s[k][i][3] - s[k][j][1]) < epsilon) and \
                             (np.abs(s[k][i][4] - s[k][j][4]) < epsilon) and (
                                     np.abs(s[k][i][5] - s[k][j][5]) < epsilon)):
                        r.append(i)
                        r.append(j)

        # Delete all internal edges from the list
        # print(s[k], r)
        s[k] = delete_from_list(s[k], r)
        # print("after deletion", s[k])
        # List of indices - edges which continue each other ("bad") and should be replaces with one edge
        r = []
        # print('sk', s[k])
        # v, u - two not "bad" parallel external edges
        for v in range(len(s[k])):
            for u in range(len(s[k])):
                if (v < u) and (parallelism(s[k][v], s[k][u]) == True) and (u not in r) and (v not in r):

                    # Four possible option for parallel edges to continue each other (one vertex must coincide)
                    if ((np.abs(s[k][u][0] - s[k][v][0]) < epsilon) and (np.abs(s[k][u][1] - s[k][v][1]) < epsilon) and
                            (np.abs(s[k][u][4] - s[k][v][4]) < epsilon) and (
                                    np.abs(s[k][u][5] - s[k][v][5]) < epsilon)):

                        # Add good big edge

                        sdkbglrkb = list(s[k]) + list(
                            [[s[k][v][2], s[k][v][3], s[k][u][2], s[k][u][3], s[k][v][4], s[k][v][5]]])
                        # print("s[k]", s[k], "hz",  [[s[k][v][2], s[k][v][3], s[k][u][2], s[k][u][3], s[k][v][4], s[k][v][5]]], "sddkjfbn", sdkbglrkb)
                        s[k] = sdkbglrkb
                        # print('cpnc', s[k])
                        # Add "bad"edges to list r for removal
                        r.append(u)
                        r.append(v)

                    elif ((np.abs(s[k][u][0] - s[k][v][2]) < epsilon) and (
                            np.abs(s[k][u][1] - s[k][v][3]) < epsilon) and \
                          (np.abs(s[k][u][4] - s[k][v][4]) < epsilon) and (np.abs(s[k][u][5] - s[k][v][5]) < epsilon)):

                        sdxbjnjld = list(s[k]) + list(
                            [[s[k][v][0], s[k][v][1], s[k][u][2], s[k][u][3], s[k][v][4], s[k][v][5]]])
                        s[k] = sdxbjnjld
                        r.append(u)
                        r.append(v)


                    elif ((np.abs(s[k][u][2] - s[k][v][2]) < epsilon) and (
                            np.abs(s[k][u][3] - s[k][v][3]) < epsilon) and \
                          (np.abs(s[k][u][4] - s[k][v][4]) < epsilon) and (np.abs(s[k][u][5] - s[k][v][5]) < epsilon)):

                        s[k] = np.concatenate(
                            (s[k], [[s[k][v][0], s[k][v][1], s[k][u][0], s[k][u][1], s[k][v][4], s[k][v][5]]]))
                        r.append(u)
                        r.append(v)

                    elif ((np.abs(s[k][u][2] - s[k][v][0]) < epsilon) and (
                            np.abs(s[k][u][3] - s[k][v][1]) < epsilon) and \
                          (np.abs(s[k][u][4] - s[k][v][4]) < epsilon) and (np.abs(s[k][u][5] - s[k][v][5]) < epsilon)):

                        s[k] = np.concatenate(
                            (s[k], [[s[k][v][2], s[k][v][3], s[k][u][0], s[k][u][1], s[k][v][4], s[k][v][5]]]), axis=0)
                        r.append(u)
                        r.append(v)

        # Delete all extra edges
        # print(s[k], r)
        s[k] = delete_from_list(s[k], r)

    return s


# Binary function that
# - returns "True" if edges are parallel or coincide
# - returns "False" if not

# a = [7/3, 1, 2, 0, 0, 1]
# b = [2, 0, 0, 1, 0, 2]

def parallelism(a, b):
    if np.abs((a[0] - a[2]) * (b[1] - b[3]) - (a[1] - a[3]) * (b[0] - b[2])) < epsilon:

        return True

    else:

        return False

# Binary function that
# - returns "True" if edges a and b are not parallel and intersect in the interior of any of edges
# - returns "False" if edges are parallel or don't intersection or it's intersection - common vertex

# a = [4, 2, 5, 3, 3, 0] (3 is the number of level, 0 is the number of space)
# b = [2.3476259661391095, 1.026174593643856, 5, 3, 3, 6]

def intersection_inmid(a, b):
    if parallelism(a, b) == True:

        return False

    else:

        # p - intersection of edges'cintinuations
        p = edges_intersection(a, b)

        if (((p[0] - a[0] > epsilon) and (a[2] - p[0] > epsilon)) or (
                (a[0] - p[0] > epsilon) and (p[0] - a[2] > epsilon)) or \
            ((np.abs(p[0] - a[0]) < epsilon) and (np.abs(p[0] - a[2]) < epsilon))) and \
                (((p[0] - b[0] > epsilon) and (b[2] - p[0] > epsilon)) or (
                        (b[0] - p[0] > epsilon) and (p[0] - b[2] > epsilon)) or \
                 ((np.abs(p[0] - b[0]) < epsilon) and (np.abs(p[0] - b[2]) < epsilon))) and \
                (((p[1] - a[1] > epsilon) and (a[3] - p[1] > epsilon)) or (
                        (a[1] - p[1] > epsilon) and (p[1] - a[3] > epsilon)) or \
                 ((np.abs(p[1] - a[1]) < epsilon) and (np.abs(p[1] - a[3]) < epsilon))) and \
                (((p[1] - b[1] > epsilon) and (b[3] - p[1] > epsilon)) or (
                        (b[1] - p[1] > epsilon) and (p[1] - b[3] > epsilon)) or \
                 ((np.abs(p[1] - b[1]) < epsilon) and (np.abs(p[1] - b[3]) < epsilon))):

            return True

        else:

            return False

# Intersection point of two edges or their continuations

# a = [3, 3, 0, 1, 0, 1]
# b = [2, 1, 1, 1, 0, 2]

def edges_intersection(a, b):
    return [((a[0] * a[3] - a[1] * a[2]) * (b[0] - b[2]) - (a[0] - a[2]) * (b[0] * b[3] - b[1] * b[2])) / \
            ((a[0] - a[2]) * (b[1] - b[3]) - (a[1] - a[3]) * (b[0] - b[2])), \
            (((a[0] * a[3] - a[1] * a[2]) * (b[1] - b[3]) - (a[1] - a[3]) * (b[0] * b[3] - b[1] * b[2])) / \
             ((a[0] - a[2]) * (b[1] - b[3]) - (a[1] - a[3]) * (b[0] - b[2])))]

# Order point in clockwise order

def order(origin, refvec):
    def clockwiseangle_and_distance(point):

        # Vector between point and the origin: v = p - o
        vector = [point[0] - origin[0], point[1] - origin[1]]
        # Length of vector: ||v||
        lenvector = math.hypot(vector[0], vector[1])
        # If length is zero there is no angle
        if lenvector == 0:
            return -math.pi, 0
        # Normalize vector: v/||v||
        normalized = [vector[0] / lenvector, vector[1] / lenvector]
        dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]  # x1*x2 + y1*y2
        diffprod = refvec[1] * normalized[0] - refvec[0] * normalized[1]  # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)
        # Negative angles represent counter-clockwise angles so we need to subtract them
        # from 2*pi (360 degrees)
        if angle < 0:
            return 2 * math.pi + angle, lenvector
        # I return first the angle because that's the primary sorting criterium
        # but if two vectors have the same angle then the shorter distance should come first.
        return angle, lenvector

    return clockwiseangle_and_distance


# List of vertices of space p in buildling poly

def find_vertices_for_polyhedron(poly, p, heights):
    # A - not unique vertices of external edges of space p [1, 2, level]
    A = []
    #     print(poly, p, heights)

    for a in ext_edges_spaces(poly)[p]:
        #         print("a", a)
        A.append([a[0], a[1], a[4]])
        A.append([a[2], a[3], a[4]])

    #     print("A", A)
    # B - unique vertices of external edges of space p [1, 2, level]
    B = np.unique(A, axis=0)
    #     print("B", B)

    # b - list of levels in on which space p is located [1, 2, 3]
    b = []
    for k in B:
        b.append(k[2])
    #     print("b", b)

    # lower hight of a space
    h = 0
    for l in range(0, int(np.min(b))):
        h += heights[l]

    # height of the space
    s = 0
    # print(A, int(np.min(b)), int(np.max(b))+1)
    for l in range(int(np.min(b)), int(np.max(b)) + 1):
        s += heights[l]

    # If space p is located on one level = max(b), we create a polyhhedron with coordinates [max(b), max(b) + 1]

    if max(b) == np.amin(b):

        points_low = [[i[0], i[1], h] for i in B]
        points_high = [[i[0], i[1], h + heights[i[2]]] for i in B]

    else:

        points_high = [[i[0], i[1], h + s] for i in B if (i[2] == max(b))]
        points_low = [[i[0], i[1], h] for i in B if (i[2] == np.amin(b))]

    s0 = 0
    s1 = 0
    for k in points_low:
        s0 += k[0]
        s1 += k[1]
    origin = [s0 / len(points_low), s1 / len(points_low)]
    refvec = [0, 1]

    key_func = f(origin, refvec)
    spoints_low = sorted(points_low, key=key_func)

    s0 = 0
    s1 = 0
    for k in points_high:
        s0 += k[0]
        s1 += k[1]
    origin = [s0 / len(points_high), s1 / len(points_high)]
    refvec = [0, 1]

    key_func = f(origin, refvec)
    spoints_high = sorted(points_high, key=key_func)
    return (spoints_low + spoints_high)


# Use ConvexHull to get faces from unordered vertices.

def get_faces_from_hull(vertices):
    hull = ConvexHull(vertices)
    faces = []
    for simplex in hull.simplices:
        faces.append([vertices[i] for i in simplex])
    return faces


# Check if there is a separating axis between two polyhedra.

def separating_axis_test(poly1, poly2):
    axes = []

    # Add face normals of both polyhedra
    for poly in (poly1, poly2):
        for face in get_faces_from_hull(poly):
            edge1 = np.array(face[1]) - np.array(face[0])
            edge2 = np.array(face[2]) - np.array(face[0])
            normal = np.cross(edge1, edge2)
            if np.linalg.norm(normal) > 0:  # Ignore zero-length normals
                axes.append(normal / np.linalg.norm(normal))

    # Add cross products of edges between the two polyhedra
    edges1 = get_edges(poly1)
    edges2 = get_edges(poly2)
    for edge1, edge2 in product(edges1, edges2):
        dir1 = np.array(edge1[1]) - np.array(edge1[0])
        dir2 = np.array(edge2[2]) - np.array(edge2[0])
        axis = np.cross(dir1, dir2)
        if np.linalg.norm(axis) > 0:  # Ignore zero-length axes
            axes.append(axis / np.linalg.norm(axis))

    # Test for overlap on all axes
    for axis in axes:
        proj1 = [np.dot(axis, v) for v in poly1]
        proj2 = [np.dot(axis, v) for v in poly2]
        if max(proj1) < min(proj2) or max(proj2) < min(proj1):
            return False  # Found a separating axis

    return True  # No separating axis found, polyhedra intersect


# Generate edges of a polyhedron.

def get_edges(vertices):
    return [(vertices[i], vertices[j]) for i, j in combinations(range(len(vertices)), 2) if
            np.any(vertices[i] != vertices[j])]


# Main function to check if two polyhedra intersect.

def check_polyhedra_intersection(poly1_vertices, poly2_vertices):
    return 1 if separating_axis_test(poly1_vertices, poly2_vertices) else 0


# Create a graph of connectivity of spaces

def graph_con_spaces(poly, heights):
    # A = nspaces x nspaces
    A = np.zeros([len(ext_edges_spaces(poly)) - 1, len(ext_edges_spaces(poly)) - 1], dtype=float, order='C')

    # i, j in range [1, nspaces]
    for i in range(1, nspaces(poly) + 1):
        for j in range(1, nspaces(poly) + 1):
            if (i < j):

                # If i and j intersect, then connect them in graph
                if check_polyhedra_intersection(find_vertices_for_polyhedron(poly, i, heights),
                                                find_vertices_for_polyhedron(poly, j, heights)):
                    A[i - 1][j - 1] = 1
                # if do_spaces_intersect(poly, i, j, heights) is not None:
                #     A[i-1][j-1] = 1
    #     print(A)
    G = nx.from_numpy_array(A)
    nx.set_node_attributes(G, {0: 'thistle', 1: 'cornflowerblue', 2: "skyblue", 3: "pink"}, name="color")
    return G


# Spaces on the ground floor

def ground_spaces(poly):
    # a - list of ground spaces
    a = []

    # i - non-zero space number
    for i in ext_edges_spaces(poly):
        if i != 0:

            # If at least part of the space is on the ground floor, then flag = 1
            f = 0
            for j in ext_edges_spaces(poly)[i]:
                if j[4] == 0:
                    f = 1

            if f == 1:
                a.append(i)
    return a


# Order point in clockwise order

def f(origin, refvec):
    def clockwiseangle_and_distance(point):

        # Vector between point and the origin: v = p - o
        vector = [point[0] - origin[0], point[1] - origin[1]]
        # Length of vector: ||v||
        lenvector = math.hypot(vector[0], vector[1])
        # If length is zero there is no angle
        if lenvector == 0:
            return -math.pi, 0
        # Normalize vector: v/||v||
        normalized = [vector[0] / lenvector, vector[1] / lenvector]
        dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]  # x1*x2 + y1*y2
        diffprod = refvec[1] * normalized[0] - refvec[0] * normalized[1]  # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)
        # Negative angles represent counter-clockwise angles so we need to subtract them
        # from 2*pi (360 degrees)
        if angle < 0:
            return 2 * math.pi + angle, lenvector
        # I return first the angle because that's the primary sorting criterium
        # but if two vectors have the same angle then the shorter distance should come first.
        return angle, lenvector

    return clockwiseangle_and_distance


def is_convex_polygon(polygon):
    try:  # needed for any bad points or direction changes
        # Check for too few points
        if len(polygon) < 3:
            #             print("0")
            return False
        # Get starting information
        old_x, old_y = polygon[-2]
        new_x, new_y = polygon[-1]
        new_direction = math.atan2(new_y - old_y, new_x - old_x)
        angle_sum = 0.0
        # Check each point (the side ending there, its angle) and accum. angles
        for ndx, newpoint in enumerate(polygon):
            # Update point coordinates and side directions, check side length
            old_x, old_y, old_direction = new_x, new_y, new_direction
            new_x, new_y = newpoint
            new_direction = math.atan2(new_y - old_y, new_x - old_x)
            if old_x == new_x and old_y == new_y:
                #                 print("1")
                return False  # repeated consecutive points
            # Calculate & check the normalized direction-change angle
            angle = new_direction - old_direction
            if angle <= -math.pi:
                angle += math.pi * 2  # make it in half-open interval (-Pi, Pi]
            elif angle > math.pi:
                angle -= math.pi * 2
            if ndx == 0:  # if first time through loop, initialize orientation
                if angle == 0.0:
                    #                     print("2")
                    return False
                orientation = 1.0 if angle > 0.0 else -1.0
            else:  # if other time through loop, check orientation is stable
                if orientation * angle <= 0.0:  # not both pos. or both neg.
                    #                     print("3")
                    return False
            # Accumulate the direction-change angle
            angle_sum += angle

        # Check that the total number of full turns is plus-or-minus 1

        return abs(round(angle_sum / (math.pi * 2))) == 1
    except (ArithmeticError, TypeError, ValueError) as e:
        raise e
        return False  # any exception means not a proper convex polygon


# Do cells p and q located on the same level have common edge:
# True - they do
# False - they don't

def do_cells_comedge(poly, p, q):
    # t1 - list of vertices in cell p: [[3, 3], [2.3333333333333335, 1], [0, 1]]
    t1 = [poly[p][0], poly[p][1], poly[p][2]]
    t2 = [poly[q][0], poly[q][1], poly[q][2]]

    # nt1 transforms couples of vertices in t1 into tuples
    nt1 = map(tuple, t1)
    nt2 = map(tuple, t2)

    # st1 - sets of vertices in cell p: {(0, 1), (2.3333333333333335, 1), (3, 3)}
    st1 = set(nt1)
    st2 = set(nt2)

    # inter - intersection of two sets of vertices
    inter = st1.intersection(st2)

    # Does the intersection have 2 elements, i.e. two cells have common edge
    if len(inter) == 2:
        return True
    else:
        return False


# List of zero cells indices on level l which have an edge on the boundary of the building

def extereme_zero_cells(poly, l, parameters):
    M = []

    for i in range(len(poly)):
        if (((poly[i][0][0] == parameters[0] and poly[i][1][0] == parameters[0]) or \
             (poly[i][0][0] == parameters[0] and poly[i][2][0] == parameters[0]) or \
             (poly[i][1][0] == parameters[0] and poly[i][2][0] == parameters[0])) or \
            ((poly[i][0][1] == parameters[1] and poly[i][1][1] == parameters[1]) or \
             (poly[i][0][1] == parameters[1] and poly[i][2][1] == parameters[1]) or \
             (poly[i][1][1] == parameters[1] and poly[i][2][1] == parameters[1])) or \
            ((poly[i][0][0] == 0 and poly[i][1][0] == 0) or \
             (poly[i][0][0] == 0 and poly[i][2][0] == 0) or \
             (poly[i][1][0] == 0 and poly[i][2][0] == 0)) or \
            ((poly[i][0][1] == 0 and poly[i][1][1] == 0) or \
             (poly[i][0][1] == 0 and poly[i][2][1] == 0) or \
             (poly[i][1][1] == 0 and poly[i][2][1] == 0))) and \
                (poly[i][4] == 0) and (poly[i][3] == l):
            M.append(i)

    return M


# List of zero cells located on level l

def zero_cells(poly, l):
    L = []
    for j in poly:
        if (j[3] == l) and (j[4] == 0):
            L.append(j)
    return L


# |x1 y1 1|
# |x2 y2 1| - determinant function
# |x3 y3 1|

def ds(p1, p2, p3):
    return np.sign(p1[0] * p2[1] + p2[0] * p3[1] + p3[0] * p1[1] - p1[1] * p2[0] - p2[1] * p3[0] - p3[1] * p1[0])


# Is point p4 inside the triangle p1p2p3?
# -1: no,
# 0: on the side,
# 1: yes

def in_triangle(p1, p2, p3, p4):
    ds123 = ds(p1, p2, p3)
    if ds123 == -ds(p1, p2, p4):
        return -1
    elif (np.abs(ds(p1, p2, p4)) < epsilon) and ((p1[0] <= p4[0] <= p2[0]) or (p2[0] <= p4[0] <= p1[0])) and (
            (p1[1] <= p4[1] <= p2[1]) or (p2[1] <= p4[1] <= p1[1])):

        return 0
    if ds123 == -ds(p1, p4, p3):
        return -1
    elif (np.abs(ds(p1, p4, p3)) < epsilon) and ((p1[0] <= p4[0] <= p3[0]) or (p3[0] <= p4[0] <= p1[0])) and (
            (p1[1] <= p4[1] <= p3[1]) or (p3[1] <= p4[1] <= p1[1])):

        return 0
    if ds123 == -ds(p4, p2, p3):
        return -1
    elif (np.abs(ds(p4, p2, p3)) < epsilon) and ((p2[0] <= p4[0] <= p3[0]) or (p3[0] <= p4[0] <= p2[0])) and (
            (p2[1] <= p4[1] <= p3[1]) or (p3[1] <= p4[1] <= p2[1])):
        return 0
    return 1


# Are 4 points a triangle/convex quadrilateral/non-convex quadrilateral?
# [[0, 1], [0, 0], [1, 1], [2, 1]]: triangle where [2, 1] lies inside the triangle [[0, 1], [0, 0], [1, 1]]
# 4: convex quadrilateral
# -1: non-convex quadrilateral


# L = [[0, 1], [0, 0], [1, 1], [2, 1]]

def four_points(p1, p2, p3, p4):
    if in_triangle(p1, p2, p3, p4) == 0:
        return [p1, p2, p3, p4]
    elif in_triangle(p3, p4, p1, p2) == 0:
        return [p3, p4, p1, p2]
    elif in_triangle(p4, p1, p2, p3) == 0:
        return [p4, p1, p2, p3]
    elif in_triangle(p2, p3, p4, p1) == 0:
        return [p2, p3, p4, p1]
    elif (in_triangle(p1, p2, p3, p4) == 1) or (in_triangle(p4, p1, p2, p3) == 1) or (
            in_triangle(p3, p4, p1, p2) == 1) or (in_triangle(p2, p3, p4, p1) == 1):
        #        print("non-convex quadrilateral")
        return -1
    else:
        #        print ("convex quadrileteral")
        return 4

# This function just deletes elements with defined indices from input_list.
# This function instead of np.delete, because it doesn't work for me on Ubuntu for some reason
def delete_from_list(input_list, indices):
    # Convert indices to a set for faster lookups
    indices_set = set(indices)

    # Create a new list excluding the elements at the specified indices
    result_list = [item for i, item in enumerate(input_list) if i not in indices_set]

    return result_list