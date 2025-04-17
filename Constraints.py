###Area of a traingle should be non zero
import collections
from collections import defaultdict

import networkx as nx
import numpy as np

from Basic_functions_to_work_with_buildings import f, ext_edges_spaces, nlevels, nspaces, zero_cells, \
    extereme_zero_cells, do_cells_comedge, ground_spaces, spaces_cells, vertices, intersection_inmid, e, spacesbylevel, \
    is_convex_polygon, graph_con_spaces

epsilon = 10e-5


def C0(poly):

    for i in poly:

        if np.abs(i[0][0] * (i[1][1] - i[2][1]) + i[1][0] * (i[2][1] - i[0][1]) + i[2][0] * (
                i[0][1] - i[1][1])) < epsilon:
            return False

    ###No duplicates on the same level (even with different b-values)

    # j - level
    for j in spacesbylevel(poly):

        # A = [0, 0, 2, 0, 0, 1] for cell [list([0, 0]) list([2, 0]) list([0, 1]) 0 0]
        A = []

        # k - space on level j
        for k in spacesbylevel(poly)[j]:
            A.append([x for y in [np.round(k[0], 5), np.round(k[1], 5), np.round(k[2], 5)] for x in y])
            A.append([x for y in [np.round(k[0], 5), np.round(k[2], 5), np.round(k[1], 5)] for x in y])
            A.append([x for y in [np.round(k[1], 5), np.round(k[2], 5), np.round(k[0], 5)] for x in y])
            A.append([x for y in [np.round(k[1], 5), np.round(k[0], 5), np.round(k[2], 5)] for x in y])
            A.append([x for y in [np.round(k[2], 5), np.round(k[1], 5), np.round(k[0], 5)] for x in y])
            A.append([x for y in [np.round(k[2], 5), np.round(k[0], 5), np.round(k[1], 5)] for x in y])

        unique, counts = np.unique(A, return_counts=True, axis=0)

        # All of them are supposed to be equal to 1
        if len(counts) != sum(counts):
            return False

    ###All levels are present without passes

    list_lev = np.unique([u[3] for u in poly])

    # # Jacob added this:
    # if len(list_lev) == 0:
    #     return False
    #
    # if max(list_lev) != nlevels(poly) - 1:
    #     return False

    ###All spaces are present without passes

    list_spaces = np.unique([u[4] for u in poly])

    if max(list_spaces) != nspaces(poly):
        return False

    return True

# Each vertex should not lie inside any other cell and even on its edge
# Only acceptable is it coincides with vertex

def C1(poly):
    for i in range(nlevels(poly)):

        # k - cell [list([0, 0]) list([2, 0]) list([0, 1]) 0 0]
        for k in poly:

            # j - point [3, 3, 0, 1]
            for j in vertices(poly)[i]:

                # Only cells on level i
                if k[3] == i:

                    t1 = (k[0][0] - j[0]) * (k[1][1] - k[0][1]) - (k[0][1] - j[1]) * (k[1][0] - k[0][0])
                    t2 = (k[1][0] - j[0]) * (k[2][1] - k[1][1]) - (k[1][1] - j[1]) * (k[2][0] - k[1][0])
                    t3 = (k[2][0] - j[0]) * (k[0][1] - k[2][1]) - (k[2][1] - j[1]) * (k[0][0] - k[2][0])

                    # The same orientation -> point is inside
                    if ((t1 > epsilon) and (t2 > epsilon) and (t3 > epsilon)) or (
                            (t1 < -epsilon) and (t2 < -epsilon) and (t3 < -epsilon)):

                        return False

                    else:

                        # Point is on the edge -> not okay if it's not on the vertex
                        # With <= 0 it would be on the continuation of the edge which is okay
                        if ((np.abs(t1) < epsilon) and (t2 * t3 >= 0)) or \
                                ((np.abs(t2) < epsilon) and (t1 * t3 >= 0)) or \
                                ((np.abs(t3) < epsilon) and (t1 * t2 >= 0)):

                            # Vertex coincides with triangle vertex -> it's okay
                            if ((np.abs(j[0] - k[0][0]) < epsilon) and (np.abs(j[1] - k[0][1]) < epsilon)) or \
                                    ((np.abs(j[0] - k[1][0]) < epsilon) and (np.abs(j[1] - k[1][1]) < epsilon)) or \
                                    ((np.abs(j[0] - k[2][0]) < epsilon) and (np.abs(j[1] - k[2][1]) < epsilon)):
                                pass

                            # Point is on the edge
                            else:
                                return False

        # Two edges on the same floor should not intersect each other in any of their interiors

        # p - edge [2, 0, 3, 3, 0, 2]
        for p in e(poly)[i]:

            # q - edge [3, 3, 0, 1, 0, 1]
            for q in e(poly)[i]:

                if intersection_inmid(p, q) == True:
                    return False
    return True

###All vertices on each floor should lie within the boundaries of the building

def C2(poly, parameters):

    # Arrays of max/min coordinate of vertices on each level
    gxmax = []
    gxmin = []
    gymax = []
    gymin = []

    # i - level
    for i in vertices(poly):

        # x, y - arrays of coordinates of vertices on specific level
        x = []
        y = []

        for j in vertices(poly)[i]:
            x.append(j[0])
            y.append(j[1])

        # On each level max x should be equal to parameter
        if np.abs(max(x) - parameters[0]) > epsilon:
            return False
        if np.abs(min(x)) > epsilon:
            return False
        if np.abs(max(y) - parameters[1]) > epsilon:
            return False
        if np.abs(min(y)) > epsilon:
            return False

    ###Complete coverage of predefined volume

    # L - list of all edges on all levels without b-value
    L = []
    for i in e(poly):
        for j in e(poly)[i]:
            L.append([j[0], j[1], j[2], j[3], j[4]])

    for i in e(poly):
        for j in e(poly)[i]:

            # If the edge is on the contour, it should be in L exactly 1 time
            if ((np.abs(j[0]) < epsilon) and (np.abs(j[2]) < epsilon)) or \
                    ((np.abs(j[0] - parameters[0]) < epsilon) and (np.abs(j[2] - parameters[0]) < epsilon)) or \
                    ((np.abs(j[1]) < epsilon) and (np.abs(j[3]) < epsilon)) or \
                    ((np.abs(j[1] - parameters[1]) < epsilon) and (np.abs(j[3] - parameters[1]) < epsilon)):
                if (L.count([j[0], j[1], j[2], j[3], j[4]]) + L.count([j[2], j[3], j[0], j[1], j[4]])) != 1:
                    return False

            # If the edge is not on the contour, it should be in L exactly 2 times
            else:
                if (L.count([j[0], j[1], j[2], j[3], j[4]]) + L.count([j[2], j[3], j[0], j[1], j[4]])) != 2:
                    return False

    #     ###We want specific height of the building

    #     #Sum all height of levels
    #     H = 0
    #     for i in heights:
    #         H += heights[i]

    #     #It should be equal to the predefined height
    #     if (H != parameters[2]):
    #         return False

    return True

#On the same level cells with the same b-value should be connected in one dual graph component
#On adjacent levels external boundaries of the same space should coincide

def C3(poly):
    # i - number of the space {1, 2, ...}
    # nspaces - how many non-zero spaces are in poly. There should be at least one
    for i in range(1, nspaces(poly) + 1):

        ###On the same level cells with the same b-value should be connected in one dual graph component

        # j and k - two arbitrary cells with b=i
        for j in range(len(spaces_cells(poly)[i])):

            # a - flag:
            # 1 if j is near to at least one other cell k (on the same level with the same b-value);
            # 0 if not near to any
            a = 0

            # d - flag:
            # 1 if there are at least one more cells with the same b-value on the same level
            # 0 if j is the only cell
            d = 0

            for k in range(len(spaces_cells(poly)[i])):

                # Check if j and k are indices for different cells on the same level
                if (np.abs(spaces_cells(poly)[i][j][3] - spaces_cells(poly)[i][k][3]) < epsilon) and (j != k):

                    # There is at least one more cell with the same b-value on the same level as j
                    d += 1

                    # Intersection of cells j and k is two vertices
                    if ((np.abs(spaces_cells(poly)[i][j][0][0] - spaces_cells(poly)[i][k][0][0]) < epsilon) and (
                            np.abs(spaces_cells(poly)[i][j][0][1] - spaces_cells(poly)[i][k][0][1]) < epsilon) and \
                        (np.abs(spaces_cells(poly)[i][j][1][0] - spaces_cells(poly)[i][k][1][0]) < epsilon) and (
                                np.abs(spaces_cells(poly)[i][j][1][1] - spaces_cells(poly)[i][k][1][1]) < epsilon)) or \
                            ((np.abs(spaces_cells(poly)[i][j][0][0] - spaces_cells(poly)[i][k][0][0]) < epsilon) and (
                                    np.abs(
                                        spaces_cells(poly)[i][j][0][1] - spaces_cells(poly)[i][k][0][1]) < epsilon) and \
                             (np.abs(spaces_cells(poly)[i][j][1][0] - spaces_cells(poly)[i][k][2][0]) < epsilon) and (
                                     np.abs(
                                         spaces_cells(poly)[i][j][1][1] - spaces_cells(poly)[i][k][2][1]) < epsilon)) or \
                            ((np.abs(spaces_cells(poly)[i][j][0][0] - spaces_cells(poly)[i][k][0][0]) < epsilon) and (
                                    np.abs(
                                        spaces_cells(poly)[i][j][0][1] - spaces_cells(poly)[i][k][0][1]) < epsilon) and \
                             (np.abs(spaces_cells(poly)[i][j][2][0] - spaces_cells(poly)[i][k][1][0]) < epsilon) and (
                                     np.abs(
                                         spaces_cells(poly)[i][j][2][1] - spaces_cells(poly)[i][k][1][1]) < epsilon)) or \
                            ((np.abs(spaces_cells(poly)[i][j][0][0] - spaces_cells(poly)[i][k][0][0]) < epsilon) and (
                                    np.abs(
                                        spaces_cells(poly)[i][j][0][1] - spaces_cells(poly)[i][k][0][1]) < epsilon) and \
                             (np.abs(spaces_cells(poly)[i][j][2][0] - spaces_cells(poly)[i][k][2][0]) < epsilon) and (
                                     np.abs(
                                         spaces_cells(poly)[i][j][2][1] - spaces_cells(poly)[i][k][2][1]) < epsilon)) or \
                            ((np.abs(spaces_cells(poly)[i][j][0][0] - spaces_cells(poly)[i][k][1][0]) < epsilon) and (
                                    np.abs(
                                        spaces_cells(poly)[i][j][0][1] - spaces_cells(poly)[i][k][1][1]) < epsilon) and \
                             (np.abs(spaces_cells(poly)[i][j][1][0] - spaces_cells(poly)[i][k][0][0]) < epsilon) and (
                                     np.abs(
                                         spaces_cells(poly)[i][j][1][1] - spaces_cells(poly)[i][k][0][1]) < epsilon)) or \
                            ((np.abs(spaces_cells(poly)[i][j][0][0] - spaces_cells(poly)[i][k][1][0]) < epsilon) and (
                                    np.abs(
                                        spaces_cells(poly)[i][j][0][1] - spaces_cells(poly)[i][k][1][1]) < epsilon) and \
                             (np.abs(spaces_cells(poly)[i][j][1][0] - spaces_cells(poly)[i][k][2][0]) < epsilon) and (
                                     np.abs(
                                         spaces_cells(poly)[i][j][1][1] - spaces_cells(poly)[i][k][2][1]) < epsilon)) or \
                            ((np.abs(spaces_cells(poly)[i][j][0][0] - spaces_cells(poly)[i][k][1][0]) < epsilon) and (
                                    np.abs(
                                        spaces_cells(poly)[i][j][0][1] - spaces_cells(poly)[i][k][1][1]) < epsilon) and \
                             (np.abs(spaces_cells(poly)[i][j][2][0] - spaces_cells(poly)[i][k][0][0]) < epsilon) and (
                                     np.abs(
                                         spaces_cells(poly)[i][j][2][1] - spaces_cells(poly)[i][k][0][1]) < epsilon)) or \
                            ((np.abs(spaces_cells(poly)[i][j][0][0] - spaces_cells(poly)[i][k][1][0]) < epsilon) and (
                                    np.abs(
                                        spaces_cells(poly)[i][j][0][1] - spaces_cells(poly)[i][k][1][1]) < epsilon) and \
                             (np.abs(spaces_cells(poly)[i][j][2][0] - spaces_cells(poly)[i][k][2][0]) < epsilon) and (
                                     np.abs(
                                         spaces_cells(poly)[i][j][2][1] - spaces_cells(poly)[i][k][2][1]) < epsilon)) or \
                            ((np.abs(spaces_cells(poly)[i][j][0][0] - spaces_cells(poly)[i][k][2][0]) < epsilon) and (
                                    np.abs(
                                        spaces_cells(poly)[i][j][0][1] - spaces_cells(poly)[i][k][2][1]) < epsilon) and \
                             (np.abs(spaces_cells(poly)[i][j][1][0] - spaces_cells(poly)[i][k][0][0]) < epsilon) and (
                                     np.abs(
                                         spaces_cells(poly)[i][j][1][1] - spaces_cells(poly)[i][k][0][1]) < epsilon)) or \
                            ((np.abs(spaces_cells(poly)[i][j][0][0] - spaces_cells(poly)[i][k][2][0]) < epsilon) and (
                                    np.abs(
                                        spaces_cells(poly)[i][j][0][1] - spaces_cells(poly)[i][k][2][1]) < epsilon) and \
                             (np.abs(spaces_cells(poly)[i][j][1][0] - spaces_cells(poly)[i][k][1][0]) < epsilon) and (
                                     np.abs(
                                         spaces_cells(poly)[i][j][1][1] - spaces_cells(poly)[i][k][1][1]) < epsilon)) or \
                            ((np.abs(spaces_cells(poly)[i][j][0][0] - spaces_cells(poly)[i][k][2][0]) < epsilon) and (
                                    np.abs(
                                        spaces_cells(poly)[i][j][0][1] - spaces_cells(poly)[i][k][2][1]) < epsilon) and \
                             (np.abs(spaces_cells(poly)[i][j][2][0] - spaces_cells(poly)[i][k][0][0]) < epsilon) and (
                                     np.abs(
                                         spaces_cells(poly)[i][j][2][1] - spaces_cells(poly)[i][k][0][1]) < epsilon)) or \
                            ((np.abs(spaces_cells(poly)[i][j][0][0] - spaces_cells(poly)[i][k][2][0]) < epsilon) and (
                                    np.abs(
                                        spaces_cells(poly)[i][j][0][1] - spaces_cells(poly)[i][k][2][1]) < epsilon) and \
                             (np.abs(spaces_cells(poly)[i][j][2][0] - spaces_cells(poly)[i][k][1][0]) < epsilon) and (
                                     np.abs(
                                         spaces_cells(poly)[i][j][2][1] - spaces_cells(poly)[i][k][1][1]) < epsilon)) or \
                            ((np.abs(spaces_cells(poly)[i][j][1][0] - spaces_cells(poly)[i][k][0][0]) < epsilon) and (
                                    np.abs(
                                        spaces_cells(poly)[i][j][1][1] - spaces_cells(poly)[i][k][0][1]) < epsilon) and \
                             (np.abs(spaces_cells(poly)[i][j][2][0] - spaces_cells(poly)[i][k][1][0]) < epsilon) and (
                                     np.abs(
                                         spaces_cells(poly)[i][j][2][1] - spaces_cells(poly)[i][k][1][1]) < epsilon)) or \
                            ((np.abs(spaces_cells(poly)[i][j][1][0] - spaces_cells(poly)[i][k][0][0]) < epsilon) and (
                                    np.abs(
                                        spaces_cells(poly)[i][j][1][1] - spaces_cells(poly)[i][k][0][1]) < epsilon) and \
                             (np.abs(spaces_cells(poly)[i][j][2][0] - spaces_cells(poly)[i][k][2][0]) < epsilon) and (
                                     np.abs(
                                         spaces_cells(poly)[i][j][2][1] - spaces_cells(poly)[i][k][2][1]) < epsilon)) or \
                            ((np.abs(spaces_cells(poly)[i][j][1][0] - spaces_cells(poly)[i][k][1][0]) < epsilon) and (
                                    np.abs(
                                        spaces_cells(poly)[i][j][1][1] - spaces_cells(poly)[i][k][1][1]) < epsilon) and \
                             (np.abs(spaces_cells(poly)[i][j][2][0] - spaces_cells(poly)[i][k][0][0]) < epsilon) and (
                                     np.abs(
                                         spaces_cells(poly)[i][j][2][1] - spaces_cells(poly)[i][k][0][1]) < epsilon)) or \
                            ((np.abs(spaces_cells(poly)[i][j][1][0] - spaces_cells(poly)[i][k][1][0]) < epsilon) and (
                                    np.abs(
                                        spaces_cells(poly)[i][j][1][1] - spaces_cells(poly)[i][k][1][1]) < epsilon) and \
                             (np.abs(spaces_cells(poly)[i][j][2][0] - spaces_cells(poly)[i][k][2][0]) < epsilon) and (
                                     np.abs(
                                         spaces_cells(poly)[i][j][2][1] - spaces_cells(poly)[i][k][2][1]) < epsilon)) or \
                            ((np.abs(spaces_cells(poly)[i][j][1][0] - spaces_cells(poly)[i][k][2][0]) < epsilon) and (
                                    np.abs(
                                        spaces_cells(poly)[i][j][1][1] - spaces_cells(poly)[i][k][2][1]) < epsilon) and \
                             (np.abs(spaces_cells(poly)[i][j][2][0] - spaces_cells(poly)[i][k][0][0]) < epsilon) and (
                                     np.abs(
                                         spaces_cells(poly)[i][j][2][1] - spaces_cells(poly)[i][k][0][1]) < epsilon)) or \
                            ((np.abs(spaces_cells(poly)[i][j][1][0] - spaces_cells(poly)[i][k][2][0]) < epsilon) and (
                                    np.abs(
                                        spaces_cells(poly)[i][j][1][1] - spaces_cells(poly)[i][k][2][1]) < epsilon) and \
                             (np.abs(spaces_cells(poly)[i][j][2][0] - spaces_cells(poly)[i][k][1][0]) < epsilon) and (
                                     np.abs(
                                         spaces_cells(poly)[i][j][2][1] - spaces_cells(poly)[i][k][1][1]) < epsilon)):
                        a = 1

            # Cell i is the only cell on the level or it is separate from any other cell with the same b-value
            if (a == 0) and (d > 0):
                return False

        ###On adjacent levels external boundaries of the same space should coincide

        # A - dictionary:
        # key - level
        # value - list of vertices of external boundaries of space i on this level
        # {1: [(0, 0), (1, 0), (0, 0), (0, 3), (1, 0), (3, 3), (3, 3), (0, 3)], 2: [(0, 0), (0, 3), (0, 3), (3, 3), (3, 3), (1, 0), (1, 0), (0, 0)]})
        A = defaultdict(list)

        # List of levels on which space i is located
        ilev = np.unique([u[4] for u in ext_edges_spaces(poly)[i]])

        # Space i is located on more than 1 level
        if len(ilev) > 1:

            # Number of levels on which space i is located != max level - min level + 1, i.e. vertically not connected
            if len(ilev) != np.max(ilev) - np.min(ilev) + 1:
                return False

            # v - one of the levels on which space i is located
            for v in ilev:

                # w - all external boundaries of space i on level v
                for w in ext_edges_spaces(poly)[i]:
                    if w[4] == v:
                        # Add vertices of external booundaires to dictionary A from edges
                        A[v].append(tuple(w[:2]))
                        A[v].append(tuple(w[2:4]))

            # d, r - unique couple of levels in A
            for d in A:
                for r in A:
                    if d < r:

                        # A[d] should be completely the same with A[r]
                        # collections.Counter(A[d]) - how many times each element is in A[d]
                        if collections.Counter(A[d]) != collections.Counter(A[r]):
                            return False
    return True

#Convex quadrilateral

def C4(poly):
    for i in range(1, nspaces(poly) + 1):
        for p in range(nlevels(poly)):

            # s - number of external edges of space i on level p
            s = 0

            # A - list of all external edges of space i on level p ([0, 1, 1, 1])
            A = []
            # print(ext_edges_spaces(poly))
            for k in ext_edges_spaces(poly)[i]:
                if k[4] == p:
                    s = s + 1
                    A.append([k[0], k[1], k[2], k[3]])
            # print(s)
            if ((s < 4) or (s > 4)) and (s > 0):
                # print("not 4")
                return False

            B = []
            for t in range(len(A)):
                if [A[t][0], A[t][1]] not in B:
                    B.append([A[t][0], A[t][1]])
                if [A[t][2], A[t][3]] not in B:
                    B.append([A[t][2], A[t][3]])

            if B != []:
                s0 = 0
                s1 = 0
                for k in B:
                    s0 += k[0]
                    s1 += k[1]

                origin = [s0 / len(B), s1 / len(B)]
                refvec = [0, 1]

                key_func = f(origin, refvec)

                points = sorted(B, key=key_func)
                # print(points)
                if is_convex_polygon(points) == False:
                    return False

    return True

#All spaces are connected to the ground (maybe through other spaces)

def C5(poly, heights):
    # a - list of all spaces connected to all spaces on the ground floor
    a = []
    # print("poly", poly)
    # h - ground space
    for h in ground_spaces(poly):
        # descendants returns all nodes reachable from ground space h in graph of connectivity
        a.extend(list(np.array(list(nx.algorithms.descendants(graph_con_spaces(poly, heights), h - 1))) + 1) + [h])

    # All spaces must be in a at least once
    if len(np.unique(a)) == nspaces(poly):
        pass
    else:
        return False

    return True

#No inner yards (i.e. no cavities)

def C6(poly, parameters):
    # l - level
    for l in range(nlevels(poly)):

        # A - metrix of connectivity of zero cells on level l
        A = np.zeros([len(zero_cells(poly, l)), len(zero_cells(poly, l))], dtype=float, order='C')
        for i in range(len(zero_cells(poly, l))):
            for j in range(len(zero_cells(poly, l))):
                if (i < j):

                    # Two nodes in the graph will be connceted if the corresponding zero cells have a common edge
                    if do_cells_comedge(zero_cells(poly, l), i, j) == True:
                        A[i][j] = 1

        # Create a graph from matrix A
        G = nx.from_numpy_array(A)

        a = []

        # h - indices of zero cells located on the boundary of level l in the list of all zero cells on level l
        for h in extereme_zero_cells(zero_cells(poly, l), l, parameters):

            # Append to a boudary zero cell with index h
            a.append([zero_cells(poly, l)[h][0], zero_cells(poly, l)[h][1], zero_cells(poly, l)[h][2]])

            # Append to a all zero cells on level l that are connected with edge to boundary zero cell with index h
            for k in list(nx.algorithms.descendants(G, h)):
                a.append([zero_cells(poly, l)[k][0], zero_cells(poly, l)[k][1], zero_cells(poly, l)[k][2]])

        # If on level l all zero-cells are connected to the boundary zero cells, then the constraint is met
        # Otherwise it's violated
        # len(np.unique(a, axis = 0)) - number of zero cells on level l which are connceted to boudary zero cells
        if len(np.unique(a, axis=0)) == len(zero_cells(poly, l)):
            pass
        else:
            return False

    return True

# Minimal angle of a corner in a space

def C7(poly, minangle):
    # i - non-zero space
    for space in range(1, nspaces(poly) + 1):

        # p - level
        for p in range(nlevels(poly)):

            # A - list of external edges of space i on level p:
            # [] or [[2.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 3.0], [1.0, 3.0, 3.0, 3.0], [2.0, 0.0, 3.0, 3.0]]
            A = []

            # Counter of boudaries of space i on level p
            s = 0

            # k - external edges of space i on level p
            for k in ext_edges_spaces(poly)[space]:
                if k[4] == p:
                    s += 1
                    A.append([k[0], k[1], k[2], k[3]])

            # If there is a part of space i on level p:
            B = []
            for t in range(len(A)):
                if [A[t][0], A[t][1]] not in B:
                    B.append([A[t][0], A[t][1]])
                if [A[t][2], A[t][3]] not in B:
                    B.append([A[t][2], A[t][3]])
            if B:
                s0 = 0
                s1 = 0
                for k in B:
                    s0 += k[0]
                    s1 += k[1]

                origin = [s0 / len(B), s1 / len(B)]
                refvec = [0, 1]

                key_func = f(origin, refvec)

                points = sorted(B, key=key_func)

                for i in range(len(points)):
                    nom = (points[(i + 1) % len(points)][0] - points[i % len(points)][0]) * (
                                points[(i - 1) % len(points)][0] - points[i % len(points)][0]) + \
                          (points[(i + 1) % len(points)][1] - points[i % len(points)][1]) * (
                                      points[(i - 1) % len(points)][1] - points[i % len(points)][1])
                    denom = np.sqrt((points[(i + 1) % len(points)][0] - points[i % len(points)][0]) ** 2 + \
                                    (points[(i + 1) % len(points)][1] - points[i % len(points)][1]) ** 2) * \
                            np.sqrt((points[(i - 1) % len(points)][0] - points[i % len(points)][0]) ** 2 + \
                                    (points[(i - 1) % len(points)][1] - points[i % len(points)][1]) ** 2)
                    angle = np.around(np.arccos(nom / denom) * 180 / np.pi)
                    if angle < minangle:
                        return False

    return True

