import csv
import random

# Function to generate a random float in a given range
def generate_random_float(min_value, max_value):
    return random.uniform(min_value, max_value)

# Open a file to write the data
with open('bigboi.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Generate 5000 lines of data
    for _ in range(2000):
        x = generate_random_float(-10, 10)
        y = generate_random_float(-10, 20)
        writer.writerow([x, y])

print("5000 lines of data have been generated and written to large.csv")