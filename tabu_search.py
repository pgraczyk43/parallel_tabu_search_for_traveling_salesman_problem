import copy
import random 
import csv
# import matplotlib.pyplot as plt

ax_iteration_without_improvment = 100
max_itteration_for_mval = 100
min_itteration_for_mval = 50
cadence = 5
num_cities = 0
tabu = {}

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


def genarate_random_solution(base_id: int) -> dict:
    global num_cities
    city_ids = list(range(1, num_cities+1))
    if base_id in city_ids:
        city_ids.remove(base_id)

    client_random_ids = random.sample(city_ids, len(city_ids))

    path = [base_id] + client_random_ids + [base_id]
    print(path)

    return path

def calculate_cost(solution: dict, cost_dictionary: dict):
    cost = 0
    for i in range(len(solution) - 1):
        cost  += cost_dictionary[(solution[i],solution[i+1])]['distance']
    return cost

def change_solution(current_soultion: dict):
    moves = []
    candidate_to_solution = copy.deepcopy(current_soultion)

    for _ in range(10000): #??
        random_node_number_1 = random.randint(1, len(candidate_to_solution)-2)
        random_node_number_2 = random.randint(1, len(candidate_to_solution)-2) # co gdy ten sam?
        move = (random_node_number_1,random_node_number_2)
        value_1 = candidate_to_solution[random_node_number_1]
        value_2 = candidate_to_solution[random_node_number_2]

        if move not in tabu:
            value_1 = candidate_to_solution[random_node_number_1]
            value_2 = candidate_to_solution[random_node_number_2]
            candidate_to_solution[random_node_number_1] = value_2
            candidate_to_solution[random_node_number_2] = value_1
            moves.append(move)
            break
    
    return candidate_to_solution, moves

def caculate_new_solution(current_soultion: dict,cost_dictionary: dict):
    MVal = {}
    current_cost = calculate_cost(current_soultion,cost_dictionary)
    itteration_for_mval = random.randint(min_itteration_for_mval,max_itteration_for_mval)
    for _ in range(itteration_for_mval):
        #strategia bez sortowania - bazowa
        new_solition, moves =  change_solution(current_soultion)

        #sortowanie po odległości
        # new_solition_unsorted, moves =  change_solution(current_soultion, clinet_weight_map, dron_capacity)
        # new_solition = sort_nodes_by_distance(1, new_solition_unsorted, cost_dictionary) 
        
        new_cost = calculate_cost(new_solition,cost_dictionary)
        m_value = current_cost - new_cost
        MVal[m_value] = (new_solition, moves)
        
        # strategia aspiracji plus
        # if(m_value > 0):
        #     MVal_2 = {}
        #     for i in range(50):
        #         new_solition2, moves =  change_solution(current_soultion, clinet_weight_map, dron_capacity)
        #         new_cost = calculate_cost(new_solition2,cost_dictionary)
        #         m_value = current_cost - new_cost
        #         MVal_2[m_value] = (new_solition2, moves)
        #         max_m_value2 = max(MVal_2.keys())
        #         # print("it:", i, m_value)
        #         # print("max: ", max_m_value2)
        #     MVal[max_m_value2] = MVal_2[max_m_value2]
        # # print("maxmax: ", max_m_value2)

    max_m_value = max(MVal.keys())
    return MVal[max_m_value]

def tabu_search(iteration_number: int, cost_dictionary: dict, base_id: int):
    global cadence
    cost_history = []

    current_solution = genarate_random_solution(base_id)

    best_solution = copy.deepcopy(current_solution)
    best_cost = calculate_cost(best_solution,cost_dictionary)
    cost_history.append(copy.deepcopy(best_cost))
    iteration_without_improvment = 0
    best_iteration = 0
    
    
    for i in range(iteration_number):
        (new_solition, moves) = caculate_new_solution(current_solution,cost_dictionary)
        new_cost = calculate_cost(new_solition,cost_dictionary)
        if  new_cost < best_cost:
            current_solution =  copy.deepcopy(new_solition)
            best_solution = copy.deepcopy(new_solition)
            best_cost = calculate_cost(best_solution,cost_dictionary)
            iteration_without_improvment = 0
            best_iteration = i
        cost_history.append(copy.deepcopy( best_cost))
        iteration_without_improvment += 1
        
        #strategia dywersyfikacji - metoda zdarzeń krytycznych
        # if iteration_without_improvment > max_iteration_without_improvment:
        #     current_solution = genarate_random_solution(drons_list, [client.id for client in client_list], base_id, clinet_weight_map, cost_dictionary)
        #     tabu.clear()
        #     iteration_without_improvment = 0
        #     print("new random solution")
        #     best_solution = copy.deepcopy(current_solution)
        #     best_cost = calculate_cost(best_solution,cost_dictionary)
        #     cost_history.append(copy.deepcopy(best_cost))

        for move in moves:
            tabu[move] = cadence

        for move_key, cadence in list(tabu.items()):
            new_cadence = cadence - 1
            if new_cadence == 0:
                del tabu[move_key]
            else:
                tabu[move_key] = new_cadence


    return best_solution, cost_history, best_iteration


if __name__ == "__main__":
    file_path = 'transformed_data_tiny.csv'
    cost_dictionary = read_csv_to_cost_dict(file_path)
    best_solution, cost_history, best_iteration = tabu_search(1000, cost_dictionary, 1) 
    
    print("best solution: " + str(best_solution))
    print("best iteration: " + str(best_iteration))
    
    # plt.title('Cost history')
    # plt.xlabel('iteration')
    # plt.ylabel('cost')
    # plt.grid(True)
    # plt.plot(cost_history)
    # plt.show()
     