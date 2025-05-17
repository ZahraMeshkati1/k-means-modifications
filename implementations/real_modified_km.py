import os
os.environ["OMP_NUM_THREADS"] = "1"
import time
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

folder_path = r'D:\EL studies\project\real1'

# Function to read text files and load data
def load_data_from_folder(folder_path):
    data = []
    file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            column_data = [list(map(float, line.strip().split())) for line in file]
            data.append(column_data)
    return np.hstack(data)  # Stack columns to get a 2D array

# Load the data from text files
X = load_data_from_folder(folder_path)


print(f"Data loaded successfully with shape: {X.shape}")

# Global counter for norm computations
norm_counter = 0
norm_list=[]
def compute_mean(data):
    return np.mean(data, axis=0)  # Calculate the mean of all data points

def objective_function(data, centers):
    global norm_counter
    sum_squares = 0
    for point in data:
        min_dist = np.min([np.linalg.norm(point - center) ** 2 for center in centers])
        norm_counter += len(centers)  # Increment counter by the number of centers
        sum_squares += min_dist
    #return sum_squares / len(data)
    return sum_squares 

start_time = time.time()

# Initial center and objective function
centers = np.array([compute_mean(X)])  # Initial center as the mean of all points
f1 = objective_function(X, centers)
fk_previous = 0
k = 1
obj_values = [f1]
epsilon = 0.01
iteration_times = []

while True:
    iteration_start_time = time.time()
    k += 1
    possible_centers = []
    for point in X:
        if not any((point == center).all() for center in centers):  # Checks if the current point is not already a center
            subset = [pt for pt in X if any(np.linalg.norm(pt - point) < np.linalg.norm(pt - c) for c in centers)]  # Creates a list of points closer to the current point
            norm_counter += len(centers) * len(X)  # Increment norm counter for subset
            if subset:
                possible_center = compute_mean(subset)
                aux_function_value = objective_function(X, np.vstack([centers, possible_center]))
                possible_centers.append((aux_function_value, possible_center))  # Adds a tuple to the possible_centers list

    if not possible_centers:
        break

    # Find the best center y that minimizes the auxiliary function
    aux_function_values, possible_centers = zip(*possible_centers)  # Unzips the list into two lists
    best_aux_index = np.argmin(aux_function_values) #index
    new_center_y = possible_centers[best_aux_index]
    
    centers = np.vstack([centers, new_center_y])  # Stack the new center

    # Apply simple k-means
    kmeans = KMeans(n_clusters=k, init=centers, n_init=1)
    kmeans.fit(X)
    norm_counter += kmeans.n_iter_ * len(X) * len(centers)  # Increment norm counter
    centers = kmeans.cluster_centers_
    predicted_labels = kmeans.labels_
    #fk = kmeans.inertia_
    
    fk = objective_function(X, centers)
    obj_values.append(fk)
    
    # Check for convergence
    if k != 2:
        if f1 != 0 and (fk_previous - fk) / f1 < epsilon:
            break
        else:
            fk_previous = fk
    else:
        if f1 != 0 and (f1 - fk) / f1 < epsilon:
            break
        else:
            fk_previous = fk
    norm_list.append(norm_counter)
    iteration_end_time = time.time() # End time for this iteration 
    iteration_times.append(iteration_end_time - iteration_start_time)
end_time = time.time()

# Plotting the predicted labels
plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis', s=50)
#plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('KMeans Clustering (Predicted Labels)')
plt.show()

#print("Final cluster centers:", centers)
print("Number of cluster centers:", k)
print(np.array(obj_values))
print(f"Run time: {end_time - start_time:.4f} seconds")
print(f"Iteration times: {iteration_times}")
print(f"norm counter: {norm_counter}")
print(norm_list)
