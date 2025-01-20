import copy
import random
import csv
import matplotlib.pyplot as plt
import numpy as np
import time  # Import the time module

# Constants
ax_iteration_without_improvment = 100
max_itteration_for_mval = 100
min_itteration_for_mval = 50
cadence = 5
num_cities = 0

def read_csv_to_cost_matrix(file_path):
    global num_cities
    with open(file_path, mode='r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        first_row = next(csvreader)
        num_cities = int(first_row[0])  
        next(csvreader)
        cost_matrix = np.zeros((num_cities + 1, num_cities + 1))
        for row in csvreader:
            point1_id, point2_id = map(int, row[0].split('-'))
            distance = float(row[5])
            cost_matrix[point1_id, point2_id] = distance
            cost_matrix[point2_id, point1_id] = distance
    return cost_matrix

def genarate_random_solution(base_id: int):
    global num_cities
    city_ids = list(range(1, num_cities + 1))
    if base_id in city_ids:
        city_ids.remove(base_id)
    client_random_ids = random.sample(city_ids, len(city_ids))
    path = [base_id] + client_random_ids + [base_id]
    return path

def calculate_cost(solution, cost_matrix):
    start_ids = np.array(solution[:-1])
    end_ids = np.array(solution[1:])
    return np.sum(cost_matrix[start_ids, end_ids])

def change_solution(current_solution, tabu):
    moves = []
    candidate_solution = current_solution[:]
    for _ in range(10000):
        i, j = sorted(random.sample(range(1, len(candidate_solution) - 1), 2))
        move = (i,j)
        if move not in tabu:
            candidate_solution[i:j+1] = reversed(candidate_solution[i:j+1])
            moves.append(move)
            break
    return candidate_solution, moves

def caculate_new_solution(current_solution, cost_matrix, tabu):
    MVal = {}
    current_cost = calculate_cost(current_solution, cost_matrix)
    # for _ in range(random.randint(min_itteration_for_mval, max_itteration_for_mval)):
    for _ in range(100):
        new_solution, moves = change_solution(current_solution, tabu)
        new_cost = calculate_cost(new_solution, cost_matrix)
        m_value = current_cost - new_cost
        MVal[m_value] = (new_solution, moves)
    max_m_value = max(MVal.keys())
    return MVal[max_m_value]

def tabu_search(file_path, start_id, iteration_number):
    global cadence
    cost_matrix = read_csv_to_cost_matrix(file_path)
    current_solution = genarate_random_solution(start_id)
    best_solution = copy.deepcopy(current_solution)
    best_cost = calculate_cost(best_solution, cost_matrix)
    cost_history = [best_cost]
    tabu = {}

    for _ in range(iteration_number):
        new_solution, moves = caculate_new_solution(current_solution, cost_matrix, tabu)
        new_cost = calculate_cost(new_solution, cost_matrix)
        if new_cost < best_cost:
            current_solution = copy.deepcopy(new_solution)
            best_solution = copy.deepcopy(new_solution)
            best_cost = new_cost
        cost_history.append(best_cost)
        for move in moves:
            tabu[move] = cadence

        for move_key, cadence in list(tabu.items()):
            new_cadence = cadence - 1
            if new_cadence == 0:
                del tabu[move_key]
            else:
                tabu[move_key] = new_cadence

    return best_solution, best_cost, cost_history

def plot_points(cost_matrix, start_id, path=None):
    plt.figure(figsize=(10, 10))
    
    num_points = cost_matrix.shape[0]

    # Randomly generate coordinates for each point
    coordinates = np.random.rand(num_points, 2) * 100  # Scale to visualize better

    # Plot each point
    for point_id in range(num_points):
        x, y = coordinates[point_id]
        if point_id == start_id:
            plt.scatter(x, y, color='red')
            plt.text(x, y, f'{point_id}', color='red', fontsize=12, ha='right')
        else:
            plt.scatter(x, y, color='blue')
            plt.text(x, y, f'{point_id}', color='blue', fontsize=12, ha='right')

    # Plot paths if provided
    if path:
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            plt.plot(
                [coordinates[start, 0], coordinates[end, 0]],
                [coordinates[start, 1], coordinates[end, 1]],
                'k-', lw=1
            )

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Visualization of Points and Paths')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # file_path = r'C:\Users\jakub\Visual Studio Code sem2\Supercomputer\transformed_data_large.csv'
    file_path = 'transformed_data_medium.csv'
    start_id = 1
    total_iterations = 1000

    start_time = time.time()
    best_solution, best_cost, cost_history = tabu_search(file_path, start_id, total_iterations)
    end_time = time.time()

    print(f"Execution Time: {end_time - start_time} seconds")
    print("Best Solution:", best_solution)
    print("Best Cost:", best_cost)
    cost_dictionary = read_csv_to_cost_matrix(file_path)
    if best_solution:
        # plot_points(cost_dictionary, start_id, best_solution)
        
        # Plot cost history
        plt.figure()
        plt.title('Cost history')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.plot(cost_history)
        plt.show()
