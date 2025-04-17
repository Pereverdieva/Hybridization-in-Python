# Vizualization of one building design
import json
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import shapely
from matplotlib import pyplot as plt
from shapely.geometry import MultiPolygon

from Basic_functions_to_work_with_buildings import nlevels, nspaces, delete_from_list

epsilon = 10e-5

#Visualization of one building design by level in 2D
def vis_poly(poly):
    n = 1
    colors = ['white', 'thistle', 'cornflowerblue', "skyblue", "pink", "mediumslateblue", "lightseagreen", 'plum',
              'lightsteelblue', 'paleturquoise', 'red', 'blue', 'yellow']

    for l in range(nlevels(poly)):

        plt.figure(figsize=(5, 10))
        plt.subplot(int(str(nlevels(poly)) + str(1) + str(n)))

        m = 0

        # A = all cells on level l
        A = []
        for i in poly:
            if i[3] == l:
                m += 1
                A.append(i)

        for t in range(nspaces(poly) + 1):

            # r - list of polygons
            r = []
            for i in range(len(A)):
                if A[i][4] == t:
                    r.append(shapely.geometry.Polygon([(A[i][0][0], A[i][0][1]), (A[i][1][0], A[i][1][1]), (A[i][2][0], A[i][2][1])]))

            new_shape = MultiPolygon(r)
            for z in range(len(new_shape.geoms)):
                #                 print(i)
                res = [list(ele) for ele in list(new_shape.geoms[z].exterior.coords[:-1])]
                #                 print(l, t)
                res.append(l)
                res.append(t)
                index = 0
                #                 print(res)
                for b in range(len(poly)):
                    if (np.abs(res[0][0] - poly[b][0][0]) < epsilon) and (
                            np.abs(res[0][1] - poly[b][0][1]) < epsilon) and (
                            np.abs(res[1][0] - poly[b][1][0]) < epsilon) and (
                            np.abs(res[1][1] - poly[b][1][1]) < epsilon) and (
                            np.abs(res[2][0] - poly[b][2][0]) < epsilon) and (
                            np.abs(res[2][1] - poly[b][2][1]) < epsilon) and (
                            np.abs(res[3] - poly[b][3]) < epsilon) and (np.abs(res[4] - poly[b][4]) < epsilon):
                        index = b

                xs, ys = new_shape.geoms[z].exterior.xy
                plt.plot(xs, ys, color='k')
                plt.fill(xs, ys, facecolor=colors[t], edgecolor='black')
                plt.ylabel('Level ' + str(l))
                # with or without numbers
        #               plt.annotate(str(index), xy = [new_shape.geoms[z].centroid.x, new_shape.geoms[z].centroid.y], xytext=[new_shape.geoms[z].centroid.x, new_shape.geoms[z].centroid.y])

        plt.show()
        n += 1
    plt.show()


def sort_by_values(list1, values):
    sorted_list = []
    while (len(sorted_list) != len(list1)):
        if index_of(min(values), values) in list1:
            sorted_list.append(index_of(min(values), values))
        values[index_of(min(values), values)] = math.inf
    return sorted_list


# Function to find index of list
def index_of(a, list):
    for i in range(0, len(list)):
        if list[i] == a:
            return i
    return -1


# Sort solutions to Pareto front
def fast_non_dominated_sort(values1, values2):
    #
    S = [[] for i in range(0, len(values1))]
    front = [[]]
    n = [0 for i in range(0, len(values1))]
    rank = [0 for i in range(0, len(values1))]

    # here we form front 0, elements which don;t have elements bigger
    # p and q indices run through all len(values)
    for p in range(0, len(values1)):
        # S[p] - list of elements smaller than p
        S[p] = []
        # n[p] - number of elements bigger than p
        n[p] = 0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (
                    values1[p] >= values1[q] and values2[p] > values2[q]) or (
                    values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (
                    values1[q] >= values1[p] and values2[q] > values2[p]) or (
                    values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        # if there are no element bigger than p => rank p = 0 => front[0] append p
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)
        # print(p, values1[p], values2[p], S[p], n[p])

    i = 0  # iterator index
    while (front[i] != []):

        # Q - will be front[i]
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if (n[q] == 0):
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front.append(Q)  # front[i] = Q from now on

    del front[len(front) - 1]
    return front


# Function to calculate crowding distance
def crowding_distance(values1, values2, front):
    # print(values1, values2, front)
    distance = [0 for i in range(0, len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1, len(front) - 1):
        scale1 = max(values1) - min(values1)
        if scale1 == 0: scale1 = 1
        distance[k] = distance[k] + (values1[sorted1[k + 1]] - values1[sorted1[k - 1]]) / scale1
    for k in range(1, len(front) - 1):
        scale2 = max(values2) - min(values2)
        if scale2 == 0: scale2 = 1
        distance[k] = distance[k] + (values2[sorted2[k + 1]] - values2[sorted2[k - 1]]) / scale2
    # print(distance)
    return distance

# Class for working with json
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# Vizualization of pareto fronts in population

def pareto_fronts(objjs, it, max_gen):
    with open(objjs, 'r') as f:
        data = f.readlines()
    print(objjs, data)
    dt = data[0]
    dt = dt[1:]
    df = []

    a = defaultdict()
    for k in range(it):
        n = 0
        a[k] = ''
        for i in range(len(dt)):
            if (dt[i] == '}'):
                dt = dt[(n + 3):]
                break
            else:
                a[k] = a[k] + dt[i]
            n += 1

        a[k] = a[k] + '}'
    for k in range(it):
        df.append(pd.DataFrame.from_dict(json.loads(a[k])))

    for k in range(it):
        df[k].columns = df[k].columns.astype(int)

    fig, axs = plt.subplots(nrows=6, ncols=5, figsize=(17, 17))
    plt.subplots_adjust(hspace=0.3)
    plt.subplots_adjust(wspace=0.3)
    fig.suptitle("Pareto fronts", fontsize=15, y=0.95)
    for q in range(it):
        ax = plt.subplot(6, 5, q + 1)
        function1 = [i * (-1) for i in non_dominated([x for y in [df[q][t][0] for t in range(1, max_gen)] for x in y],
                                                     [x for y in [df[q][t][1] for t in range(1, max_gen)] for x in y])[
            0]]
        function2 = [j * (-1) for j in non_dominated([x for y in [df[q][t][0] for t in range(1, max_gen)] for x in y],
                                                     [x for y in [df[q][t][1] for t in range(1, max_gen)] for x in y])[
            1]]
        plt.xlabel('Objective 1', fontsize=7)
        plt.ylabel('Objective 2', fontsize=7)
        plt.tick_params(axis='both', which='major', labelsize=5)
        plt.tick_params(axis='both', which='minor', labelsize=5)
        plt.scatter(function1, function2, s=1)
    plt.show()


# Total pareto fronts for all solutions among different iterations and generations

def one_pareto_front(js, it, max_gen):
    with open(js, 'r') as f:
        data = f.readlines()
    dt = data[0]
    dt = dt[1:]
    df = []

    a = defaultdict()
    for k in range(it):
        n = 0
        a[k] = ''
        for i in range(len(dt)):
            if (dt[i] == '}'):
                dt = dt[(n + 3):]
                break
            else:
                a[k] = a[k] + dt[i]
            n += 1

        a[k] = a[k] + '}'
    for k in range(it):
        df.append(pd.DataFrame.from_dict(json.loads(a[k])))

    for k in range(it):
        df[k].columns = df[k].columns.astype(int)

    X = []
    Y = []
    for q in range(it):
        X.append([i * (-1) for i in non_dominated([x for y in [df[q][t][0] for t in range(1, max_gen)] for x in y],
                                                  [x for y in [df[q][t][1] for t in range(1, max_gen)] for x in y])[0]])
        Y.append([j * (-1) for j in non_dominated([x for y in [df[q][t][0] for t in range(1, max_gen)] for x in y],
                                                  [x for y in [df[q][t][1] for t in range(1, max_gen)] for x in y])[1]])
    X = [x for y in X for x in y]
    Y = [x for y in Y for x in y]
    function1 = non_dominated(X, Y)[0]
    function2 = non_dominated(X, Y)[1]
    plt.xlabel('Objective 1', fontsize=7)
    plt.ylabel('Objective 2', fontsize=7)
    plt.tick_params(axis='both', which='major', labelsize=5)
    plt.tick_params(axis='both', which='minor', labelsize=5)
    plt.scatter(function1, function2, s=1)
    plt.show()
    return non_dominated(X, Y)


# Non-dominated solutions
def non_dominated(values1, values2):
    M = []
    for q in range(0, len(values1)):
        for p in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (
                    values1[p] >= values1[q] and values2[p] > values2[q]) or (
                    values1[p] > values1[q] and values2[p] >= values2[q]):
                M.append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (
                    values1[q] >= values1[p] and values2[q] > values2[p]) or (
                    values1[q] > values1[p] and values2[q] >= values2[p]):
                M.append(p)
    return [delete_from_list(values1, M), delete_from_list(values2, M)]


# Pareto fronts: for 30 iterations and then one resulting pareto fronts for all solution among all interations
def non_domin(obj_json, it, max_gen):
    pareto_fronts(obj_json, it, max_gen)
    plt.savefig('Pareto_fronts.png')
    return one_pareto_front(obj_json, it, max_gen)