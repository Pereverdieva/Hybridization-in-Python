from collections import defaultdict
import random
import subprocess
import time
import json
import os


from Functions_for_optimization_and_output import sort_by_values, index_of, crowding_distance, fast_non_dominated_sort, \
    NpEncoder, vis_poly
from Functions_to_work_with_BSO_and_objectives import obj1, obj2, write_to_eindhoven_file
from Mutation import mutation
from Sample import heights_for_big_sample, big_sample, small_sample, heights_for_small_sample

epsilon = 10e-5


def NSGA2(P, heights, parameters, it, max_gen, step_size, minangle):
    start_time = time.time()

    pop_size = len(P)  # population size

    a_fp = open('./objectives_' + str(step_size) + '.json', 'w')
    b_fp = open('./solutions_' + str(step_size) + '.json', 'w')
    B = []
    D = []

    final_heights = []

    for iteration in range(it):

        print("iteration", iteration)

        # Main program starts here

        # Initialization
        A = defaultdict()
        C = defaultdict()
        solution = P

        original_indices = list(range(0, pop_size))
        heights_first = heights

        gen_no = 0
        while (gen_no < max_gen):

            print("GENERATION", gen_no)

            # It's just helpful to store the objective functions on each stage to the file, not very needed
            function1_values = [obj1(solution, i, heights, iteration, gen_no, 1) for i in range(0, pop_size)]
            function2_values = [obj2(solution, i, heights, iteration, gen_no, 1) for i in range(0, pop_size)]
            non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:], function2_values[:])

            # We copy the current population, then we will add mutated copies to this list and then we will select the best ones out of all of them
            solution2 = solution[:]
            heights2 = heights[:]

            # Generating new offsprings (n=pop_size -> 2*pop_size)
            while (len(solution2) != 2 * pop_size):
                s = 0
                while s == 0:
                    a1 = random.randint(0, pop_size - 1)
                    app = mutation(solution[a1], parameters, step_size, heights[a1], minangle)
                    if app is None:  # it means that mutation produces an offspring that violates constraints
                        pass
                    else:
                        solution2.append(app)
                        heights2.append(heights[a1])
                        s = 1
                        original_indices.append(original_indices[a1])
                       # print("one is found")
            function1_values2 = [obj1(solution2, i, heights2, iteration, gen_no, 2) for i in range(0, 2 * pop_size)]
            function2_values2 = [obj2(solution2, i, heights2, iteration, gen_no, 2) for i in range(0, 2 * pop_size)]

            A[gen_no] = [function1_values, function2_values]

            non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:], function2_values2[:])
            crowding_distance_values2 = []

            # for each rank we add the array of crowding distances
            for i in range(0, len(non_dominated_sorted_solution2)):
                crowding_distance_values2.append(
                    crowding_distance(function1_values2[:], function2_values2[:], non_dominated_sorted_solution2[i][:]))

            # Maximize crowding distance -> selection
            new_solution = []

            # This part is just sorting the crowding distances per front.
            # So that we add new solutions one by one to the new generation starting from front 0, biggest crowding distance. smaller, than front 1 and so on
            for i in range(0, len(non_dominated_sorted_solution2)):
                non_dominated_sorted_solution2_1 = [
                    index_of(non_dominated_sorted_solution2[i][j], non_dominated_sorted_solution2[i]) for j in
                    range(0, len(non_dominated_sorted_solution2[i]))]
                front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
                front = [non_dominated_sorted_solution2[i][front22[j]] for j in
                         range(0, len(non_dominated_sorted_solution2[i]))]
                front.reverse()
                for value in front:
                    new_solution.append(value)
                    if (len(new_solution) == pop_size):
                        break
                if (len(new_solution) == pop_size):
                    break

            # Save all the best points (n=pop_size)
            solution = [solution2[i] for i in new_solution]
            heights = [heights2[original_indices[i]] for i in new_solution]

            new_original_indices = []
            for i in new_solution:
                new_original_indices.append(original_indices[i])

            original_indices = new_original_indices
            C[gen_no] = solution
            gen_no = gen_no + 1
        B.append(A)
        D.append(C)
        final_heights_in_this_iteration = defaultdict()
        print("Original indices for visualisation", original_indices)

        for i in range(len(solution)):
            final_heights_in_this_iteration[i] = heights_first[new_original_indices[i]]
            write_to_eindhoven_file(solution[i], heights_first[new_original_indices[i]],
                                    'solution{}_{}'.format(iteration, i))
        iteration += 1
        final_heights.append(final_heights_in_this_iteration)
    json.dump(B, a_fp)
    json.dump(D, b_fp, cls=NpEncoder)
    a_fp.close()
    b_fp.close()

    print("--- %s seconds ---" % (time.time() - start_time))

    #Solutions and their objectives are in files, heights are output
    return final_heights


if __name__ == '__main__':
    #path to the directory where  you have get_results if it is not here
#    os.chdir("../../Documents/BSO/code_for_colaboration_Leiden/translate_run_visualize")
#    print(os.getcwd())
#    results = subprocess.run(["ls", "-l"], capture_output=True, text=True)
#    print(results.stdout)

    # The line below if you want to insert this parameter from keyboard
    # parameters = list(map(float,input('Please enter the maximum length and width separated by a space (meters): ').split()))
    parameters = [5000, 3000]

    #The line below if you want to insert this parameter from keyboard
    #minangle = int(input('Please enter the minimal allowable angle (degrees): '))
    minangle = 5

    it = 1  # number of iterations
    max_gen = 10  # number of generations
    step_size = 0.99  # between 0 and 1, step size of the mutation
    P = big_sample()
    heights = heights_for_big_sample()
    print(parameters, it, max_gen, step_size)

    subprocess.run(["find", "-type", "f", "-name", "poly*", "-delete"])
    subprocess.run(["find", "-type", "f", "-name", "solution*", "-delete"])
    start_time = time.time()
    NSGA2(P, heights, parameters, it, max_gen, step_size, minangle)
    print("--- %s seconds ---" % (time.time() - start_time))
