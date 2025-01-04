import ray
import copy
import random
import csv
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda, float32
import multiprocessing

# Constants
ax_iteration_without_improvment = 100
max_itteration_for_mval = 100
min_itteration_for_mval = 50
cadence = 5
num_cities = 0
tabu = {}

# Initialize Ray
ray.init(address="ray://192.168.1.81:10001")  # Connect to Ray cluster as a client

# Check CUDA availability
if not cuda.is_available():
    raise RuntimeError("CUDA is not available. Please ensure that CUDA is installed and a compatible GPU is present.")

# Optionally, set the specific GPU device to use
cuda.select_device(0)

# CUDA kernels for FFT
@cuda.jit
def fft2_cuda(f, fft_result):
    rows, cols = f.shape
    u, v = cuda.grid(2)
    if u < rows and v < cols:
        real_sum = 0.0
        imag_sum = 0.0
        for x in range(rows):
            for y in range(cols):
                angle = 2 * np.pi * (u * x / rows + v * y / cols)
                real_sum += f[x, y] * np.cos(angle)
                imag_sum += -f[x, y] * np.sin(angle)
        fft_result[u, v] = complex(real_sum, imag_sum)

@cuda.jit
def ifft2_cuda(F, inverse_result):
    rows, cols = F.shape
    x, y = cuda.grid(2)
    if x < rows and y < cols:
        real_sum = 0.0
        for u in range(rows):
            for v in range(cols):
                angle = 2 * np.pi * (u * x / rows + v * y / cols)
                real_sum += F[u, v].real * np.cos(angle) - F[u, v].imag * np.sin(angle)
        inverse_result[x, y] = real_sum / (rows * cols)

def fft2_custom(f):
    rows, cols = f.shape
    fft_result = np.zeros((rows, cols), dtype=np.complex64)
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(rows / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(cols / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    fft2_cuda[blockspergrid, threadsperblock](f, fft_result)
    cuda.synchronize()
    return fft_result

def ifft2_custom(F):
    rows, cols = F.shape
    inverse_result = np.zeros((rows, cols), dtype=np.float32)
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(rows / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(cols / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    ifft2_cuda[blockspergrid, threadsperblock](F, inverse_result)
    cuda.synchronize()
    return inverse_result

def read_csv_to_cost_dict(file_path):
    global num_cities
    cost_dict = {}
    with open(file_path, mode='r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        first_row = next(csvreader)
        num_cities = int(first_row[0])  
        next(csvreader)
        for row in csvreader:
            path = row[0]
            point1_id, point2_id = path.split('-')
            point1x, point1y = float(row[1]), float(row[2])
            point2x, point2y = float(row[3]), float(row[4])
            distance = float(row[5])
            cost_dict[(int(point1_id), int(point2_id))] = {
                'point1': {'x': point1x, 'y': point1y},
                'point2': {'x': point2x, 'y': point2y},
                'distance': distance
            }
    return cost_dict

def plot_points(cost_dict, start_id, path=None):
    plt.figure(figsize=(10, 10))
    
    for (point1_id, point2_id), data in cost_dict.items():
        point1 = data['point1']

        if point1_id == start_id :
            plt.scatter(point1['x'], point1['y'], color='red')
            plt.text(point1['x'], point1['y'], f'{point1_id}', color='red', fontsize=12, ha='right')
            continue
        
        plt.scatter(point1['x'], point1['y'], color='blue')
        plt.text(point1['x'], point1['y'], f'{point1_id}', color='blue', fontsize=12, ha='right')

    if path:
        for i in range(len(path) - 1):
            start_id = path[i]
            end_id = path[i + 1]
            
            start_point = cost_dict.get((start_id, end_id))
            if start_point:
                start_coords = start_point['point1'] 
                end_coords = start_point['point2']

                plt.plot([start_coords['x'], end_coords['x']], [start_coords['y'], end_coords['y']], 'k-', lw=1)

        
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Visualization of Points and Paths')
    plt.grid(True)
    plt.show()

def genarate_random_solution(base_id: int):
    global num_cities
    city_ids = list(range(1, num_cities+1))
    if base_id in city_ids:
        city_ids.remove(base_id)
    client_random_ids = random.sample(city_ids, len(city_ids))
    path = [base_id] + client_random_ids + [base_id]
    return path

def calculate_cost(solution, cost_dictionary):
    cost = 0
    for i in range(len(solution) - 1):
        cost += cost_dictionary[(solution[i], solution[i+1])]['distance']
    return cost

def change_solution(current_solution):
    moves = []
    candidate_solution = copy.deepcopy(current_solution)
    for _ in range(10000):
        random_node_number_1 = random.randint(1, len(candidate_solution)-2)
        random_node_number_2 = random.randint(1, len(candidate_solution)-2)
        move = (random_node_number_1, random_node_number_2)
        if move not in tabu:
            value_1 = candidate_solution[random_node_number_1]
            value_2 = candidate_solution[random_node_number_2]
            candidate_solution[random_node_number_1] = value_2
            candidate_solution[random_node_number_2] = value_1
            moves.append(move)
            break
    return candidate_solution, moves

def caculate_new_solution(current_solution, cost_dictionary):
    MVal = {}
    current_cost = calculate_cost(current_solution, cost_dictionary)
    for _ in range(random.randint(min_itteration_for_mval, max_itteration_for_mval)):
        new_solution, moves = change_solution(current_solution)
        new_cost = calculate_cost(new_solution, cost_dictionary)
        m_value = current_cost - new_cost
        MVal[m_value] = (new_solution, moves)
    max_m_value = max(MVal.keys())
    return MVal[max_m_value]

@ray.remote
def tabu_search_worker(file_path, start_id, iteration_number, seed):
    cost_dictionary = read_csv_to_cost_dict(file_path)
    random.seed(seed)
    current_solution = genarate_random_solution(start_id)
    best_solution = copy.deepcopy(current_solution)
    best_cost = calculate_cost(best_solution, cost_dictionary)
    cost_history = [best_cost]
    
    for _ in range(iteration_number):
        new_solution, moves = caculate_new_solution(current_solution, cost_dictionary)
        new_cost = calculate_cost(new_solution, cost_dictionary)
        if new_cost < best_cost:
            current_solution = copy.deepcopy(new_solution)
            best_solution = copy.deepcopy(new_solution)
            best_cost = new_cost
        cost_history.append(best_cost)
    
    return best_solution, best_cost, cost_history

def distributed_tabu_search(file_path, start_id, workers=4):
    iteration_number = 100
    tasks = [tabu_search_worker.remote(file_path, start_id, iteration_number, seed) for seed in range(workers)]
    results = ray.get(tasks)
    best_solution = min(results, key=lambda x: x[1])
    return best_solution

if __name__ == "__main__":
    file_path = r'C:\Users\jakub\Visual Studio Code sem2\Supercomputer\transformed_data_small.csv'
    start_id = 1
    cost_dictionary = read_csv_to_cost_dict(file_path)
    best_solution, best_cost, cost_history = distributed_tabu_search(file_path, start_id, workers=4)
    print("Best Solution:", best_solution)
    print("Best Cost:", best_cost)

    if best_solution:
        plot_points(cost_dictionary, start_id, best_solution)
        
        plt.figure()
        plt.title('Cost history')
        plt.xlabel('iteration')
        plt.ylabel('cost')
        plt.grid(True)
        plt.plot(cost_history)
        plt.show()