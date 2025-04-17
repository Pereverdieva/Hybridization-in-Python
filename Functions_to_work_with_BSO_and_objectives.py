import re
import subprocess
import sys, os

import numpy as np
from contextlib import contextmanager

from Basic_functions_to_work_with_buildings import nspaces, ext_edges_spaces, order
from Functions_for_optimization_and_output import vis_poly


# Ability to turn off the output of command line
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# Translation to Eindhoven: write_to_eindhoven_file(poly, height, filename)

def points_eindhov(poly, height):
    final = []
    for p in range(1, nspaces(poly) + 1):
        A = []

        for a in ext_edges_spaces(poly)[p]:
            A.append([a[0], a[1], a[4]])
            A.append([a[2], a[3], a[4]])

        # B - unique vertices of external edges of space p [1, 2, level]
        B = np.unique(A, axis=0)

        # b - list of levels in on which space p is located [1, 2, 3]
        b = []
        for k in B:
            b.append(k[2])

        # lower hight of a space
        h = 0
        for l in range(0, int(np.min(b))):
            h += height[l]

        # height of the space
        s = 0
        for l in range(int(np.min(b)), int(np.max(b)) + 1):
            s += height[l]

        # If space p is located on one level = max(b), we create a polyhhedron with coordinates [max(b), max(b) + 1]

        if max(b) == np.amin(b):

            points_low = [[i[0], i[1], h] for i in B]
            points_high = [[i[0], i[1], h + height[i[2]]] for i in B]

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

        key_func = order(origin, refvec)
        spoints_low = sorted(points_low, key=key_func)

        s0 = 0
        s1 = 0
        for k in points_high:
            s0 += k[0]
            s1 += k[1]
        origin = [s0 / len(points_high), s1 / len(points_high)]
        refvec = [0, 1]

        key_func = order(origin, refvec)
        spoints_high = sorted(points_high, key=key_func)
        # print(poly, A, 'high', spoints_high, 'low', spoints_low)
        final.append(
            [p, spoints_low[0], spoints_low[1], spoints_low[2], spoints_low[3], spoints_high[0], spoints_high[1],
             spoints_high[2], spoints_high[3]])

    return final


def write_to_eindhoven_file(poly, height, filename):
    data = points_eindhov(poly, height)
    with open(filename, 'w') as f:
        for sublist in data:
            N_value = sublist[0]  # Extract the first element (N value)
            rest = sublist[1:]  # Remaining elements in each sublist
            # Flatten the list of lists into a single string
            flattened = ',\t\t'.join([','.join(map(str, item)) for item in rest])
            # Write to file in the desired format
            f.write(f'N, {N_value},\t\t{flattened}\n')


# # Translation to Leiden: read_from_eindhoven_file(text_file)

def remove_duplicates(coords):
    # Use a list to preserve order and avoid duplicates
    seen = set()
    unique_coords = []
    for coord in coords:
        coord_tuple = tuple(coord)  # Convert to tuple to make it hashable for the set
        if coord_tuple not in seen:
            seen.add(coord_tuple)
            unique_coords.append(coord)  # Append only unique coordinates
    return unique_coords


def transform_block(data_block, group_number):
    # Find all sets of coordinates inside each block
    all_sets = re.findall(r'\{\{(.*?)\}\}', data_block)

    results = []

    # Remove the first set and process the remaining ones
    for i, row in enumerate(all_sets):
        if i == 0:
            continue  # Skip the first set

        # Split the row into individual sets of coordinates
        coords = [list(map(float, coord.split(','))) for coord in row.split('},{')]

        # Remove duplicates from the coordinates
        coords = remove_duplicates(coords)

        # Extract the third coordinate (z) values
        third_coords = [z for _, _, z in coords]

        # Find the minimum third coordinate for this fragment
        min_third = min(third_coords) if third_coords else None  # Handle case where no z values

        # Convert 3D coordinates to 2D by discarding the third (z) value
        transformed_coords = [[x, y] for x, y, z in coords]

        # Append minimal third coordinate and group number to the transformed coordinates
        if min_third is not None:
            transformed_coords.append(min_third)  # Append the minimum third coordinate
        transformed_coords.append(group_number)  # Append the group number

        # Add the transformed row to results
        results.append(transformed_coords)

    return results

def process_multiple_blocks(input_string):
    # Split the input by blocks; the pattern captures group number and corresponding data block
    blocks = re.findall(r'(\d+),\s*(\{\{.*?\}\}(?:\s*\{\{.*?\}\})*)', input_string, re.DOTALL)

    all_results = []

    for group_number, data_block in blocks:
        group_number = int(group_number)  # Convert the group number to integer
        transformed_data = transform_block(data_block, group_number)
        all_results.extend(transformed_data)  # Append results for each block

    return all_results

def transform_to_ranks(input_list):
    # Get unique values and sort them
    unique_values = sorted(set(input_list))

    # Create a mapping of value to its rank (index)
    value_to_rank = {value: index for index, value in enumerate(unique_values)}

    # Transform the input list to its ranks
    transformed = [value_to_rank[value] for value in input_list]

    return transformed

def read_from_eindhoven_file(text_file):
    # Process the input and transform the data
    # Using a context manager to read the file into a string
    file_path = text_file  # Replace with your file path

    with open(file_path, 'r') as file:
        input_string = file.read()  # Read the entire file into a string

    # Now file_content contains the text of the file
    #     print(input_string)  # Output the content of the file

    transformed_data = process_multiple_blocks(input_string)
    nodub = []
    # Display the transformed output
    flag = 0
    for item in transformed_data:
        nodub.append(item)

    # print(nodub)

    p = []
    for i in nodub:
        #     print(i)
        p.append([i[0], i[1], i[2], i[6], i[7]])

    levels = []
    for i in p:
        levels.append(i[3])

    # Transform the values
    output_values = transform_to_ranks(levels)
    #     print(output_values)

    for i in range(len(p)):
        p[i][3] = output_values[i]
    vis_poly(p)
    return p

#Input information from result file

def extract_sd_bp_from_file(file_path):
    # Jacob added this (assuming maximization?)
    if not os.path.isfile(file_path):
        return float("inf"), float("inf")

    sd_value = None
    bp_value = None

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("SDResult"):
                sd_value = float(line.split(',')[1].strip())
            elif line.startswith("BPResult"):
                bp_value = float(line.split(',')[1].strip())

    if sd_value is not None and bp_value is not None:
        # Convert SD value to integer as required
        sd_value = float(sd_value)
        return sd_value, bp_value
    else:
        raise ValueError("Could not find SDResult and BPResult in the file.")

#Objectives work COMMENT NEEDED

def obj1(P, ind, heights, it, gen, num):
    filename = f"poly{ind}_eindrepr_{it}_{gen}_{num}_results"
    result = subprocess.run(f'[ -f {filename} ] && echo 1 || echo 0', shell=True, capture_output=True, text=True)
    output = result.stdout.strip() == '1'
    if output:
        obj_sd = extract_sd_bp_from_file('poly{}_eindrepr_{}_{}_{}_results'.format(ind, it, gen, num))[0]
        return -obj_sd
    elif not output:
        write_to_eindhoven_file(P[ind], heights[ind], 'poly{}_eindrepr_{}_{}_{}'.format(ind, it, gen, num))
        with suppress_stdout():
            input_file = f"poly{ind}_eindrepr_{it}_{gen}_{num}"
            print("input_file", input_file)
            subprocess.run(['./get_results', '-i', input_file, '-r', '-o', 'result_file'], stdout=subprocess.DEVNULL)
        filename2 = f"poly{ind}_eindrepr_{it}_{gen}_{num}_results"
        result2 = subprocess.run(f'[ -f {filename2} ] && echo 1 || echo 0', shell=True, capture_output=True, text=True)
        output2 = result2.stdout.strip() == '1'
        if not output2:
            print(P[ind])


        obj_sd = extract_sd_bp_from_file('poly{}_eindrepr_{}_{}_{}_results'.format(ind, it, gen, num))[0]
        return -obj_sd
    else:
        print('error')


def obj2(P, ind, heights, it, gen, num):
    filename = f"poly{ind}_eindrepr_{it}_{gen}_{num}_results"
    result = subprocess.run(f'[ -f {filename} ] && echo 1 || echo 0', shell=True, capture_output=True, text=True)
    output = result.stdout.strip() == '1'
    # output = get_ipython().getoutput(' [ -f poly{ind}_eindrepr_{it}_{gen}_{num}_results ] && echo 1 || echo 0')
    # print(int(output[0]), int(output[0])==0, ind)
    if output:
        # print("est")
        obj_bp = extract_sd_bp_from_file('poly{}_eindrepr_{}_{}_{}_results'.format(ind, it, gen, num))[1]
        return -obj_bp
    elif not output:
        # print("net")
        write_to_eindhoven_file(P[ind], heights[ind], 'poly{}_eindrepr_{}_{}_{}'.format(ind, it, gen, num))
        with suppress_stdout():
            input_file = f"poly{ind}_eindrepr_{it}_{gen}_{num}"
            subprocess.run(['./get_results', '-i', input_file, '-r', '-o', 'result_file'], stdout=subprocess.DEVNULL)
        # output2 = get_ipython().getoutput(' [ -f poly{ind}_eindrepr_{it}_{gen}_{num}_results ] && echo 1 || echo 0')
        filename2 = f"poly{ind}_eindrepr_{it}_{gen}_{num}_results"
        result2 = subprocess.run(f'[ -f {filename2} ] && echo 1 || echo 0', shell=True, capture_output=True, text=True)
        output2 = result2.stdout.strip() == '1'
        if not output2:
            print(P[ind])
        obj_bp = extract_sd_bp_from_file('poly{}_eindrepr_{}_{}_{}_results'.format(ind, it, gen, num))[1]
        return -obj_bp
    else:
        print('error')

def pareto_solutions(objjs, soljs, heights, it, max_gen, pop_size):
    nondom = non_domin(objjs, it, max_gen)
    sols = sol(soljs, it)
    final = []
    for non in range(len(nondom[0])):
        print(non, nondom[0][non], nondom[1][non])
        ind = 0
        while ind == 0:
            for i in range(it):
                for j in range(max_gen):
                    # print(j)
                    for d in range(pop_size):
                        #                 print(obj1(sols[i][j][d], heights), obj2(sols[i][j][d], heights))
                        #                 print(sols[i][j][d]) (P, ind, heights, it, gen, num)
                        if (np.abs(-obj1(sols[i][j], d, heights, i, j, 567) - nondom[0][non]) < epsilon) and (
                                np.abs(-obj2(sols[i][j], d, heights, i, j, 567) - nondom[1][non]) < epsilon):
                            vis_poly(sols[i][j][d])
                            final.append([sols[i][j][d], d])
                            ind = 1
                            print("ind = 1")
    for i in range(len(final)):
        write_to_eindhoven_file(final[i][0], heights[final[i][1]], 'optimal_design_{}'.format(i))
    return final


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_objectives(obj_data):
    obj = []
    for it in range(len(obj_data)):
        for gen in range(len(obj_data[it])):
            for des in range(len(obj_data[it]['{}'.format(gen)][0])):
                obj.append([-obj_data[it]['{}'.format(gen)][0][des], -obj_data[it]['{}'.format(gen)][1][des]])
    return obj

def extract_solutions(sol_data):
    solut = []
    for it in range(len(sol_data)):
        for gen in range(len(sol_data[it])):
            for des in range(len(sol_data[it]['{}'.format(gen)])):
                solut.append(sol_data[it]['{}'.format(gen)][des])
    return solut

def is_non_dominated(point, others):
    ind = 0
    for other in others:
        if ((other[0] <= point[0] and other[1] < point[1]) or (other[0] < point[0] and other[1] <= point[1])):
            ind = 1
    if ind == 0:
        return True
    else:
        return False

def non_dominated_solutions(obj_file, sol_file, heights):
    obj_data = load_json(obj_file)
    sol_data = load_json(sol_file)

    objectives = extract_objectives(obj_data)
    solutions = extract_solutions(sol_data)

    non_dominated_objectives = []
    non_dominated_sols = []
    non_dominated_indices = []
    for i in range(len(objectives)):
        if is_non_dominated(objectives[i], np.delete(objectives, i, axis=0)):
            non_dominated_objectives.append(objectives[i])
            non_dominated_sols.append(solutions[i])
            non_dominated_indices.append(i)
    for i in range(len(non_dominated_sols)):
        write_to_eindhoven_file(non_dominated_sols[i], heights[non_dominated_indices[i] % (len(sol_data[0]['0']))],
                                'optimal_design_{}'.format(i))

    return [non_dominated_objectives, non_dominated_sols]