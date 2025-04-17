import copy
import math
import random
from collections import defaultdict

import numpy as np
from scipy.spatial import Delaunay
from scipy.stats import bernoulli, truncnorm

from Basic_functions_to_work_with_buildings import nlevels, spaces_cells, do_cells_comedge, intersection_inmid, \
    edges_intersection, four_points, ext_edges_spaces, unique_v, unique_e, delete_from_list
from Constraints import C0, C1, C2, C3, C7, C4

epsilon = 10e-5


def change_sval(poly, step_size, minangle):
    if poly is None:
        return None

    # Flip happens only if randon Bernoulli variable = 1. It depends on step_size
    bern = bernoulli.rvs(size=1, p=step_size)
    if bern == 1:

        D = defaultdict()

        q = random.choice(np.unique([u[4] for u in poly]))

        list_spaces = np.unique([u[4] for u in poly])

        if q != 0:

            # l - level
            for l in range(nlevels(poly)):

                A = []
                # i - space
                for i in spaces_cells(poly):

                    if (i != 0):

                        # List of cells on level l for space i
                        K = []

                        # cells of space i
                        for j in spaces_cells(poly)[i]:
                            if j[3] == l:
                                K.append([j[0], j[1], j[2]])

                        if not K:
                            pass
                        else:
                            A.append(list([K, i]))

                D[l] = A

            if not D:
                return None

            level = random.choice([i for i in D])

            if D[level] is None:
                return None
            component = random.choice([i for i in D[level]])

            for i in component[:-1]:

                I = []
                surr_cells = []

                for j in i:
                    J = 0
                    for k in range(len(poly)):
                        set1 = set([tuple(v) for v in poly[k][:-2]])
                        set2 = set([tuple(u) for u in j])
                        if (poly[k][3] == level) and (poly[k][4] == component[-1]) and (set1 == set2):
                            J = k
                            I.append(J)

                for j in i:
                    # List of indices of near cells
                    L = []
                    for k in range(len(poly)):
                        set1 = set([tuple(v) for v in poly[k][:-2]])
                        set2 = set([tuple(u) for u in j])
                        if (poly[k][3] == level) and (poly[k][4] == component[-1]) and (set1 == set2):
                            J = k
                    for k in range(len(poly)):
                        if do_cells_comedge(poly, J, k) and (poly[k][3] == level) and (k not in I):
                            L.append(k)

                    if L not in surr_cells:
                        surr_cells.append(L)

            if not [x for y in surr_cells for x in y]:
                return None

            p = len(surr_cells)
            cell_to_col = random.choices([x for y in surr_cells for x in y], k=p)

            poly1 = copy.deepcopy(poly)

            for b in cell_to_col:
                poly1[b][4] = component[-1]

            list_spaces1 = np.unique([u[4] for u in poly1])
            if (len(list_spaces1) != len(list_spaces)):
                return None
            if (C0(poly1) == False):
                return None
            elif (C1(poly1) == False):
                return None
            elif (C3(poly1) == False):
                return None
            elif (C4(poly1) == False):
                return None
            elif (C7(poly1, minangle) == False):
                return None

        else:

            A = []
            for i in range(len(poly)):
                if poly[i][4] != 0:
                    A.append(i)

            f1 = random.choice(A)

            poly1 = copy.deepcopy(poly)

            poly1[f1][4] = 0

            list_spaces1 = np.unique([u[4] for u in poly1])
            if (len(list_spaces1) != len(list_spaces)):
                return None
            if (C0(poly1) == False):
                return None
            elif (C1(poly1) == False):
                return None
            elif (C3(poly1) == False):
                return None
            elif (C4(poly1) == False):
                return None
            elif (C7(poly1, minangle) == False):
                return None

        return poly1


def diagonal_flip(poly, step_size, minangle):
    if poly is None:
        return None
    # Flip happens only if randon Bernoulli variable = 1. It depends on step_size
    bern = bernoulli.rvs(size=1, p=step_size)
    if bern == 1:

        # A - list of convex quadrilaterals in building structure
        A = []

        # spaces_cells:
        # 2: [[[5, 0], [5, 1], [2, 0], 0, 2], [[2, 0], [2.3333333333333335, 1], [5, 1], 0, 2]]

        # i - space
        for i in spaces_cells(poly):

            # k - level
            for k in range(nlevels(poly)):

                # L - list of pairs of cells of the same sapce on the same level
                L = []

                # j and l - different indices of cells belonging to space i on level k
                for j in range(len(spaces_cells(poly)[i])):
                    for l in range(len(spaces_cells(poly)[i])):

                        if (j < l) and (spaces_cells(poly)[i][l][3] == k) and (spaces_cells(poly)[i][j][3] == k):
                            L.append([spaces_cells(poly)[i][j], spaces_cells(poly)[i][l]])

                # If there is no space i on level k
                if not L:
                    pass
                else:

                    # a - pair of cells of space i on level k
                    for a in L:

                        # t1 and t2 - first 3 coordinates of two cells
                        t1 = [a[0][0], a[0][1], a[0][2]]
                        t2 = [a[1][0], a[1][1], a[1][2]]

                        nt1 = map(tuple, t1)
                        nt2 = map(tuple, t2)

                        st1 = set(nt1)
                        st2 = set(nt2)

                        # inter - set of common vertices in two cells
                        inter = st1.intersection(st2)

                        # If two cells have common edge (2 vertices) then there are 4 vertices in total for two cells
                        if len(inter) == 2:

                            # un - all vertices
                            un = list(st1.union(st2))

                            # diff - two not common vertices
                            diff = list(st1.union(st2).difference(inter))

                            # If all 4 vertices form a convex quadrilateral -> add this pair of cells to A
                            if four_points(un[0], un[1], un[2], un[3]) == 4:
                                A.append([a, diff, list(inter), k, i])

        if not A:
            return None

        # f - random convex quadrilateral
        f = random.choice(A)

        # M - indices of cells in poly which we will delete to replace them with others
        M = []

        # p - index of cells in poly
        for p in range(len(poly)):

            # We're looking for two cells in poly coinciding with f and add their indices to M to delete later
            if (np.abs(f[0][0][0][0] - poly[p][0][0]) < epsilon) and (
                    np.abs(f[0][0][0][1] - poly[p][0][1]) < epsilon) and \
                    (np.abs(f[0][0][1][0] - poly[p][1][0]) < epsilon) and (
                    np.abs(f[0][0][1][1] - poly[p][1][1]) < epsilon) and \
                    (np.abs(f[0][0][2][0] - poly[p][2][0]) < epsilon) and (
                    np.abs(f[0][0][2][1] - poly[p][2][1]) < epsilon) and \
                    (np.abs(f[0][0][3] - poly[p][3]) < epsilon) and (np.abs(f[0][0][4] - poly[p][4]) < epsilon):
                M.append(p)
            if (np.abs(f[0][1][0][0] - poly[p][0][0]) < epsilon) and (
                    np.abs(f[0][1][0][1] - poly[p][0][1]) < epsilon) and \
                    (np.abs(f[0][1][1][0] - poly[p][1][0]) < epsilon) and (
                    np.abs(f[0][1][1][1] - poly[p][1][1]) < epsilon) and \
                    (np.abs(f[0][1][2][0] - poly[p][2][0]) < epsilon) and (
                    np.abs(f[0][1][2][1] - poly[p][2][1]) < epsilon) and \
                    (np.abs(f[0][1][3] - poly[p][3]) < epsilon) and (np.abs(f[0][1][4] - poly[p][4]) < epsilon):
                M.append(p)

        # Delete old cells
        poly = delete_from_list(poly, M)

        # Add flipped cells
        poly1 = poly + [[[f[1][0][0], f[1][0][1]], [f[1][1][0], f[1][1][1]], [f[2][0][0], f[2][0][1]], f[3], f[4]]]
        poly = poly1 + [[[f[1][0][0], f[1][0][1]], [f[1][1][0], f[1][1][1]], [f[2][1][0], f[2][1][1]], f[3], f[4]]]

        if (C7(poly, minangle) == False):
            return None
        elif (C0(poly) == False):
            return None
        elif (C1(poly) == False):
            return None


        return poly


def add_vertex_toedge(poly):
    if poly is None:
        return None
    # list of all unique edges in poly
    L = []

    # j - edge on level i
    for i in unique_e(poly):
        for j in unique_e(poly)[i]:
            L.append(j)

    # a - random edge in poly
    a = random.choice(L)

    # b - random point on edge a
    t = np.random.uniform(low=0.0, high=1.0)
    b = [a[0] + t * (a[2] - a[0]), a[1] + t * (a[3] - a[1])]

    # M - indices of cells in poly which we will delete to replace them with others
    M = []

    # i - index of cell in poly
    for i in range(len(poly)):

        # Looking for the cells containing edge a and adding new cells
        if (((np.abs(a[0] - poly[i][0][0]) < epsilon) and (np.abs(a[1] - poly[i][0][1]) < epsilon)) or \
            ((np.abs(a[0] - poly[i][1][0]) < epsilon) and (np.abs(a[1] - poly[i][1][1]) < epsilon)) or \
            ((np.abs(a[0] - poly[i][2][0]) < epsilon) and (np.abs(a[1] - poly[i][2][1]) < epsilon))) and \
                (((np.abs(a[2] - poly[i][0][0]) < epsilon) and (np.abs(a[3] - poly[i][0][1]) < epsilon)) or \
                 ((np.abs(a[2] - poly[i][1][0]) < epsilon) and (np.abs(a[3] - poly[i][1][1]) < epsilon)) or \
                 ((np.abs(a[2] - poly[i][2][0]) < epsilon) and (np.abs(a[3] - poly[i][2][1]) < epsilon))) and \
                ((np.abs(a[4] - poly[i][3]) < epsilon)):

            if ((np.abs(a[0] - poly[i][0][0]) < epsilon) and (np.abs(a[1] - poly[i][0][1]) < epsilon)):
                if ((np.abs(a[2] - poly[i][1][0]) < epsilon) and (np.abs(a[3] - poly[i][1][1]) < epsilon)):
                    poly1 = poly + [[poly[i][2], [a[0], a[1]], b, poly[i][3], poly[i][4]]]
                    poly = poly1 + [[poly[i][2], [a[2], a[3]], b, poly[i][3], poly[i][4]]]
                elif ((np.abs(a[2] - poly[i][2][0]) < epsilon) and (np.abs(a[3] - poly[i][2][1]) < epsilon)):
                    poly1 = poly + [[poly[i][1], [a[0], a[1]], b, poly[i][3], poly[i][4]]]
                    poly = poly1 + [[poly[i][1], [a[2], a[3]], b, poly[i][3], poly[i][4]]]
            elif ((np.abs(a[0] - poly[i][1][0]) < epsilon) and (np.abs(a[1] - poly[i][1][1]) < epsilon)):
                if ((np.abs(a[2] - poly[i][0][0]) < epsilon) and (np.abs(a[3] - poly[i][0][1]) < epsilon)):
                    poly1 = poly + [[poly[i][2], [a[0], a[1]], b, poly[i][3], poly[i][4]]]
                    poly = poly1 + [[poly[i][2], [a[2], a[3]], b, poly[i][3], poly[i][4]]]
                elif ((np.abs(a[2] - poly[i][2][0]) < epsilon) and (np.abs(a[3] - poly[i][2][1]) < epsilon)):
                    poly1 = poly + [[poly[i][0], [a[0], a[1]], b, poly[i][3], poly[i][4]]]
                    poly = poly1 + [[poly[i][0], [a[2], a[3]], b, poly[i][3], poly[i][4]]]
            elif ((np.abs(a[0] - poly[i][2][0]) < epsilon) and (np.abs(a[1] - poly[i][2][1]) < epsilon)):
                if ((np.abs(a[2] - poly[i][0][0]) < epsilon) and (np.abs(a[3] - poly[i][0][1]) < epsilon)):
                    poly1 = poly + [[poly[i][1], [a[0], a[1]], b, poly[i][3], poly[i][4]]]
                    poly = poly1 + [[poly[i][1], [a[2], a[3]], b, poly[i][3], poly[i][4]]]
                elif ((np.abs(a[2] - poly[i][1][0]) < epsilon) and (np.abs(a[3] - poly[i][1][1]) < epsilon)):
                    poly1 = poly + [[poly[i][0], [a[0], a[1]], b, poly[i][3], poly[i][4]]]
                    poly = poly1 + [[poly[i][0], [a[2], a[3]], b, poly[i][3], poly[i][4]]]
            #             else:
            #                 print("error")

            # Adding the index of cells containing edge e to M to delete it later
            M.append(i)

    # Deleting old cells
    poly = delete_from_list(poly, M)
    #     print(a)
    return poly


def add_vertex_inter(poly):
    if poly is None:
        return None

    # a - random cell in ply
    a = random.choice(poly)

    # p - random point inside cell
    # https://stackoverflow.com/questions/47410054/generate-random-locations-within-a-triangular-domain

    x, y = sorted([random.random(), random.random()])
    q = abs(x - y)
    s = q
    t = 0.5 * (x + y - q)
    u = 1 - 0.5 * (q + x + y)
    p = [s * a[0][0] + t * a[1][0] + u * a[2][0], s * a[0][1] + t * a[1][1] + u * a[2][1]]

    # M - indices of cells in poly which we will delete to replace them with others
    M = []

    # Looking for a cell coinciding with a and add its index to M to delete it later
    for i in range(len(poly)):
        if (np.abs(a[0][0] - poly[i][0][0]) < epsilon) and (np.abs(a[0][1] - poly[i][0][1]) < epsilon) and \
                (np.abs(a[1][0] - poly[i][1][0]) < epsilon) and (np.abs(a[1][1] - poly[i][1][1]) < epsilon) and \
                (np.abs(a[2][0] - poly[i][2][0]) < epsilon) and (np.abs(a[2][1] - poly[i][2][1]) < epsilon) and \
                (np.abs(a[3] - poly[i][3]) < epsilon) and (np.abs(a[4] - poly[i][4]) < epsilon):
            M.append(i)

    # Deleting old cell
    poly = delete_from_list(poly, M)

    # Adding new 3 cells
    # poly = poly + [a, b, c]
    # poly[-3] = [[a[0][0], a[0][1]], [a[1][0], a[1][1]], p, a[3], a[4]]
    # poly[-2] = [[a[0][0], a[0][1]], [a[1][0], a[1][1]], p, a[3], a[4]]
    # poly[-1] = [[a[0][0], a[0][1]], [a[1][0], a[1][1]], p, a[3], a[4]]
    poly1 = poly + [[[a[0][0], a[0][1]], [a[1][0], a[1][1]], p, a[3], a[4]]]
    poly2 = poly1 + [[[a[2][0], a[2][1]], [a[1][0], a[1][1]], p, a[3], a[4]]]
    poly = poly2 + [[[a[2][0], a[2][1]], [a[0][0], a[0][1]], p, a[3], a[4]]]

    return poly


def add_vertex(poly, step_size, heights, parameters, minangle):
    if poly is None:
        return None

    # Adding happens only if randon Bernoulli variable = 1. It depends on step_size
    bern = bernoulli.rvs(size=1, p=step_size)
    if bern == 1:

        adding_type = random.random()
        if adding_type < 0.5:
            poly = add_vertex_inter(poly)
        else:
            poly = add_vertex_toedge(poly)

    if poly is None:
        return None

    if (C7(poly, minangle) == False):
        return None
    elif (C0(poly) == False):
        return None
    elif (C1(poly) == False):
        return None

    return poly


def delete_vertex(poly, step_size, parameters, minangle):
    if poly is None:
        return None

    # Deleting happens only if randon Bernoulli variable = 1. It depends on step_size
    bern = bernoulli.rvs(size=1, p=step_size)
    if bern == 1:

        # List of vertices that might be deleted later
        V = []

        # i - level
        for i in unique_v(poly):

            # j - vertex on level i
            for j in unique_v(poly)[i]:

                # flag_boundary = 0, if not on the building boundary
                # flag_boundary = 1, if on the building boundary
                # flag_boundary = 2, if on the building corner
                flag_boundary = 0

                # S - list of spaces incedent to the vertex j
                S = []

                # k - cell in poly
                for k in poly:

                    # if vertex j is in the cell k
                    if ((np.abs(k[0][0] - j[0]) < epsilon) and (np.abs(k[0][1] - j[1]) < epsilon) or \
                        ((np.abs(k[1][0] - j[0]) < epsilon) and (np.abs(k[1][1] - j[1]) < epsilon)) or \
                        ((np.abs(k[2][0] - j[0]) < epsilon) and (np.abs(k[2][1] - j[1]) < epsilon))) and \
                            (np.abs(k[3] - j[2]) < epsilon):

                        # Adding space number to S
                        S.append(k[4])

                        # If on the boundary -> flag_boundary = 1
                        if ((np.abs(j[0] - parameters[0]) < epsilon) and (np.abs(j[1] - parameters[1]) > epsilon) and (
                                np.abs(j[1]) > epsilon)) or \
                                ((np.abs(j[0]) < epsilon) and (np.abs(j[1] - parameters[1]) > epsilon) and (
                                        np.abs(j[1]) > epsilon)) or \
                                ((np.abs(j[1] - parameters[1]) < epsilon) and (
                                        np.abs(j[0] - parameters[0]) > epsilon) and (np.abs(j[0]) > epsilon)) or \
                                ((np.abs(j[1]) < epsilon) and (np.abs(j[0] - parameters[0]) > epsilon) and (
                                        np.abs(j[0] - 0) > epsilon)):
                            flag_boundary = 1

                        # If on the corner -> flag_boundary = 2
                        elif ((np.abs(j[0] - parameters[0]) < epsilon) and (np.abs(j[1] - parameters[1]) < epsilon)) or \
                                ((np.abs(j[0]) < epsilon) and (np.abs(j[1] - parameters[1]) < epsilon)) or \
                                ((np.abs(j[0] - parameters[0]) < epsilon) and (np.abs(j[1]) < epsilon)) or \
                                ((np.abs(j[0]) < epsilon) and (np.abs(j[1]) < epsilon)):
                            flag_boundary = 2
                # If on the boundary and belongs to 1 space -> add to V
                if ((flag_boundary == 1) and (len(np.unique(S)) == 1)):
                    V.append([j, flag_boundary, np.unique(S)])

                # If not on the boundary and belongs max to 2 spaces -> add to V
                elif ((flag_boundary == 0) and ((len(np.unique(S)) == 2) or (len(np.unique(S)) == 1))):
                    V.append([j, flag_boundary, np.unique(S)])

        if len(V) == 0:
            return None

        # f - random vetex which is allowed to be deleted
        f = random.choice(V)

        # List of cells we're going to delete
        M = []

        # All cells that contain f are added to M and then deleted
        for i in range(len(poly)):
            if (((np.abs(poly[i][0][0] - f[0][0]) < epsilon) and (np.abs(poly[i][0][1] - f[0][1]) < epsilon)) or \
                ((np.abs(poly[i][1][0] - f[0][0]) < epsilon) and (np.abs(poly[i][1][1] - f[0][1]) < epsilon)) or \
                ((np.abs(poly[i][2][0] - f[0][0]) < epsilon) and (np.abs(poly[i][2][1] - f[0][1]) < epsilon))) and \
                    (np.abs(poly[i][3] - f[0][2]) < epsilon):
                M.append(i)

        # C - list of cells containing f
        C = []
        for i in M:
            C.append(poly[i])

        # Delete these cells
        poly = delete_from_list(poly, M)

        # D - external edges of cells in C, might be 2 list if there are 2 spaces
        D = ext_edges_spaces(C)

        # k - space number in list of external edges
        for k in D:

            # Vertices of external edges for each space
            points = []
            points.append([D[k][i][d:d + 2] for i in range(0, len(D[k])) for d in range(0, 4, 2)])
            points = np.array([a for b in points for a in b])
            g = np.unique(points, axis=0)

            # Triangulation of each set of cells
            tri = Delaunay(g)
            for h in range(len(tri.simplices)):
                # Add triangulated cells
                poly1 = poly + [[a for b in [g[tri.simplices[h]], [C[0][3]], [k]] for a in b]]
                poly = poly1

        if (C7(poly, minangle) == False):
            return None
        elif (C1(poly) == False):
            return None
        elif (C0(poly) == False):
            return None
        return poly


def alt_moving_points(poly, parameters, step_size, heights, minangle):
    if poly is None:
        return None

    # Deleting happens only if randon Bernoulli variable = 1. It depends on step_size
    bern = bernoulli.rvs(size=1, p=step_size)
    if bern == 1:

        poly1 = []

        # L - list of [vertex, area of polygon around, edges of polygon around]
        L = []

        # j - vertex on level i
        for i in unique_v(poly):
            for j in unique_v(poly)[i]:

                # A - all edges of cells containing vertex j
                A = []

                # S - area of all cells containing vetex j
                S = 0

                # k - cell in poly
                for k in range(len(poly)):

                    # If j inside cell k, and k is located on level i
                    if (np.array_equal(poly[k][0], [j[0], j[1]]) or np.array_equal(poly[k][1],[j[0], j[1]]) or np.array_equal(poly[k][2], [j[0], j[1]])) and (poly[k][3] == i):
                        A.append([x for y in [poly[k][1], poly[k][2]] for x in y])
                        A.append([x for y in [poly[k][0], poly[k][2]] for x in y])
                        A.append([x for y in [poly[k][0], poly[k][1]] for x in y])
                        a = np.sqrt((poly[k][1][0] - poly[k][0][0]) ** 2 + (poly[k][1][1] - poly[k][0][1]) ** 2)
                        b = np.sqrt((poly[k][2][0] - poly[k][1][0]) ** 2 + (poly[k][2][1] - poly[k][1][1]) ** 2)
                        c = np.sqrt((poly[k][2][0] - poly[k][0][0]) ** 2 + (poly[k][2][1] - poly[k][0][1]) ** 2)
                        s = (a + b + c) / 2

                        # S += area of cell k
                        S += np.around(np.sqrt(s * (s - a) * (s - b) * (s - c)), 2)

                ###Delete internal edges of polygon formed by cells surrounding vertex j

                # r - list of internal edges to delete later
                r = []
                # p and q - pair of different indices for edges in A
                for p in range(len(A)):
                    for q in range(len(A)):
                        if p < q:

                            # If edges p and q are the same, then add both to r to delete it later
                            if ((np.abs(A[p][0] - A[q][0]) < epsilon) and (np.abs(A[p][1] - A[q][1]) < epsilon) and \
                                (np.abs(A[p][2] - A[q][2]) < epsilon) and (np.abs(A[p][3] - A[q][3]) < epsilon)) or \
                                    ((np.abs(A[p][0] - A[q][2]) < epsilon) and (np.abs(A[p][1] - A[q][3]) < epsilon) and \
                                     (np.abs(A[p][2] - A[q][0]) < epsilon) and (np.abs(A[p][3] - A[q][1]) < epsilon)):
                                r.append(p)
                                r.append(q)

                                # Delete all internal edges from the list A, so A is the list of external edges of A
                A = delete_from_list(A, r)

                L.append([[j[0], j[1], j[2]], S, A])

        a = random.choice([x for x in L])

        # If random element a has vertex on the border but not on the corner of the building
        if ((a[0][0] > 0) and (a[0][0] < parameters[0]) and ((a[0][1] == 0) or (a[0][1] == parameters[1]))) or \
                ((a[0][1] > 0) and (a[0][1] < parameters[1]) and ((a[0][0] == 0) or (a[0][0] == parameters[0]))):

            # Find the segment of the border on which this vertex is located
            segment = []

            # Looking for external edges of the polygon which contain vertex a[0]
            # a[2] - edges of the polygon
            for j in a[2]:
                if (np.abs(j[0] - a[0][0]) < epsilon) and (np.abs(j[1] - a[0][1]) < epsilon):
                    segment.append(j[2])
                    segment.append(j[3])
                elif (np.abs(j[2] - a[0][0]) < epsilon) and (np.abs(j[3] - a[0][1]) < epsilon):
                    segment.append(j[0])
                    segment.append(j[1])

                    # If the segment is vertical
            if (np.abs(segment[0] - segment[2]) < epsilon):

                # Direction of the move towards one end of the segment or another
                direc = random.sample([segment[1], segment[3]], 1)[0]

                # Truncated distribution from point towards the end of segment
                if (direc > a[0][1]):
                    r = truncnorm.rvs(0, (direc - a[0][1]) * 3, loc=a[0][1], scale=np.abs(direc - a[0][1]) / 3, size=1)
                elif (direc < a[0][1]):
                    r = truncnorm.rvs((direc - a[0][1]) * 3, 0, loc=a[0][1], scale=np.abs(direc - a[0][1]) / 3, size=1)

                    # f - [generated point, level]
                f = [segment[0], r[0], a[0][2]]

                # If the segment is horizontal
            elif (np.abs(segment[1] - segment[3]) < epsilon):

                # Direction of the move towards one end of the segment or another
                direc = random.sample([segment[0], segment[2]], 1)[0]

                # Truncated distribution from point towards the end of segment
                if (direc > a[0][0]):
                    r = truncnorm.rvs(0, (direc - a[0][0]) * 3, loc=a[0][0], scale=np.abs(direc - a[0][0]) / 3, size=1)
                elif (direc < a[0][0]):
                    r = truncnorm.rvs((direc - a[0][0]) * 3, 0, loc=a[0][0], scale=np.abs(direc - a[0][0]) / 3, size=1)

                    # f - [generated point, level]
                f = [r[0], segment[1], a[0][2]]

            #             else:

            if 'f' in locals():
                # M - indices of cells to be deleted later
                M = []

                # poly1 - copy of poly without reference
                poly1 = copy.deepcopy(poly)

                # If cell k contains vertex -> add moved cells
                for k in range(len(poly)):
                    if (np.abs(a[0][0] - poly1[k][0][0]) < epsilon) and (
                            np.abs(a[0][1] - poly1[k][0][1]) < epsilon) and (np.abs(a[0][2] - poly1[k][3]) < epsilon):
                        poly2 = poly1 + [[[f[0], f[1]], poly1[k][1], poly1[k][2], poly1[k][3], poly1[k][4]]]
                        M.append(k)
                    elif (np.abs(a[0][0] - poly1[k][1][0]) < epsilon) and (
                            np.abs(a[0][1] - poly1[k][1][1]) < epsilon) and (np.abs(a[0][2] - poly1[k][3]) < epsilon):
                        poly2 = poly1 + [[poly1[k][0], [f[0], f[1]], poly1[k][2], poly1[k][3], poly1[k][4]]]
                        M.append(k)
                    elif (np.abs(a[0][0] - poly1[k][2][0]) < epsilon) and (
                            np.abs(a[0][1] - poly1[k][2][1]) < epsilon) and (np.abs(a[0][2] - poly1[k][3]) < epsilon):
                        poly2 = poly1 + [[poly1[k][0], poly1[k][1], [f[0], f[1]], poly1[k][3], poly1[k][4]]]
                        M.append(k)

                        # Delete old cells
                poly1 = delete_from_list(poly2, M)

                # If random element a is in the interior of the building
        elif (a[0][0] < parameters[0]) and (a[0][1] < parameters[1]) and (a[0][0] > 0) and (a[0][1] > 0):

            # Generate random angle in degrees
            angle = random.random() * 360

            # k - edge of the polygon
            for k in a[2]:

                # vec - vector from a[0] to big number with chosen angle
                vec = [a[0][0], a[0][1], a[0][0] + np.cos(angle * math.pi / 180) * 10 ** 30,
                       a[0][1] + np.sin(angle * math.pi / 180) * 10 ** 30]

                # If edge k intersects the vector
                if intersection_inmid(k, vec):
                    # Calculate the distance to intersection
                    inter = edges_intersection(k, vec)
                    length = np.sqrt((inter[0] - a[0][0]) ** 2 + (inter[0] - a[0][1]) ** 2)
                    radius = [a[0][0], a[0][1], inter[0], inter[1]]

                    # Truncated distribution from point to intersection
                    r = truncnorm.rvs(0, 3, loc=0, scale=1 / 3, size=1)[0]

                    # f - [generated point, level]
                    f = [a[0][0] + (inter[0] - a[0][0]) * r, a[0][1] + (inter[1] - a[0][1]) * r, a[0][2]]

            if 'f' in locals():

                # indices of cells to be deleted later
                M = []

                # poly1 - copy of poly without reference
                poly1 = copy.deepcopy(poly)

                # If cell k contains vertex -> add moved cells
                for k in range(len(poly)):
                    if (np.abs(a[0][0] - poly1[k][0][0]) < epsilon) and (
                            np.abs(a[0][1] - poly1[k][0][1]) < epsilon) and (np.abs(a[0][2] - poly1[k][3]) < epsilon):
                        poly2 = poly1 + [[[f[0], f[1]], poly1[k][1], poly1[k][2], poly1[k][3], poly1[k][4]]]
                        M.append(k)
                    elif (np.abs(a[0][0] - poly1[k][1][0]) < epsilon) and (
                            np.abs(a[0][1] - poly1[k][1][1]) < epsilon) and (np.abs(a[0][2] - poly1[k][3]) < epsilon):
                        poly2 = poly1 + [[poly1[k][0], [f[0], f[1]], poly1[k][2], poly1[k][3], poly1[k][4]]]
                        M.append(k)
                    elif (np.abs(a[0][0] - poly1[k][2][0]) < epsilon) and (
                            np.abs(a[0][1] - poly1[k][2][1]) < epsilon) and (np.abs(a[0][2] - poly1[k][3]) < epsilon):
                        poly2 = poly1 + [[poly1[k][0], poly1[k][1], [f[0], f[1]], poly1[k][3], poly1[k][4]]]
                        M.append(k)

                        # Delete old cells
                poly1 = delete_from_list(poly2, M)

        else:
            poly1 = poly
            # If obtained polygons are not convex or smth else is wrong with constraint 1
        if (C0(poly1) == False):
            return None
        elif (C1(poly1) == False):
            return None
        elif (C2(poly1, parameters) == False):
            return None
        elif (C3(poly1) == False):
            return None
        elif (C4(poly1) == False):
            return None
        elif (C7(poly1, minangle) == False):
            return None

        return poly1


def mutation(poly, parameters, step_size, heights, minangle):
    poly03 = diagonal_flip(poly, step_size, minangle)

    if poly03 is None:
        # print("poly03")
        poly03 = poly
    poly06 = add_vertex(poly03, step_size, heights, parameters, minangle)

    if poly06 is None:
        # print("poly06")
        poly06 = poly03
    poly1 = delete_vertex(poly06, step_size, parameters, minangle)

    if poly1 is None:
        # print("poly1")
        poly1 = poly06
    poly2 = change_sval(poly1, step_size, minangle)

    if poly2 is None:
        # print("poly2")
        poly2 = poly1

    poly3 = alt_moving_points(poly2, parameters, step_size, heights, minangle)

    if poly3 is None:
        # print("poly3")
        poly3 = poly2
    return poly3