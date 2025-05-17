import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import time
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Set the path to your folder containing text files
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

K = 2

norm_counter = 0   #in 2 places:obj func and run_kmeans

def compute_mean(data):
    return np.mean(data, axis=0)

def objective_function(data, centers):
    global norm_counter
    sum_squares = 0
    for point in data:
        min_dist = np.min([np.linalg.norm(point - center) ** 2 for center in centers])
        norm_counter += len(centers)  # Increment counter by the number of centers
        sum_squares += min_dist
    return sum_squares 

def run_kmeans(X, initial_centers):
    global norm_counter
    kmeans = KMeans(n_clusters=len(initial_centers), init=np.array(initial_centers), n_init=1)
    kmeans.fit(X)
    # Add the number of norms computed during k-means fit
    norm_counter += kmeans.n_iter_ * len(X) * len(initial_centers)
    return kmeans.labels_, kmeans.inertia_, kmeans.cluster_centers_

def global_kmeans(X, K):
    centers = np.array([compute_mean(X)])  # Start with the mean of all points as the first center
    _, obj_val, centers = run_kmeans(X, centers)
    #obj_values = [objective_function(X, centers)]  #initial
    obj_values = [obj_val]
    for k in range(2, K + 1):
        min_obj_func = float('inf')
        best_new_center = None
        for point in X:
            if any((point == center).all() for center in centers):
                continue
            temp_centers = np.vstack([centers, point])
            _, obj_func, _ = run_kmeans(X, temp_centers)
            if obj_func < min_obj_func:
                min_obj_func = obj_func
                best_new_center = point
       
        centers = np.vstack([centers, best_new_center])
        predicted_labels, obj_val, centers = run_kmeans(X, centers)
        #obj_values.append(objective_function(X, centers))
        obj_values.append(obj_val)
    return predicted_labels, centers, obj_values

start_time = time.time()

predicted_labels, final_centers, obj_values = global_kmeans(X, K)

end_time = time.time()

# Plotting the results with predicted labels
plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis', s=50)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Global KMeans Clustering (Predicted Labels)')
plt.show()


#print("Final cluster centers:", final_centers)
print("Objective values:", np.array(obj_values))
print(f"Run time: {end_time - start_time:.4f} seconds")
print(f"norm counter: {norm_counter}")

'''
2:
Data loaded successfully with shape: (683, 9)
Objective values: [48443.0658858  19323.17381706]
Run time: 5.9817 seconds
norm counter: 5018001


5:
Objective values: [48443.0658858  19323.17381706 16255.51124196 14733.72633758
 13706.38594608]
Run time: 26.4509 seconds
norm counter: 56400774



10:
Objective values: [48443.0658858  19323.17381706 16255.51124196 14733.72633758
 13706.38594608 12839.07915323 12035.08244104 11341.80488333
 10732.96306595 10202.26023656]
Run time: 58.0451 seconds
norm counter: 212404804

15:
Objective values: [48443.0658858  19323.17381706 16255.51124196 14733.72633758
 13706.38594608 12839.07915323 12035.08244104 11341.80488333
 10732.96306595 10202.26023656  9839.97838568  9489.0205221
  9172.8803216   8927.19110172  8699.79903197]
Run time: 86.8762 seconds
norm counter: 402116933

20:
Data loaded successfully with shape: (683, 9)
Objective values: [48443.0658858  19323.17381706 16255.51124196 14733.72633758
 13706.38594608 12839.07915323 12035.08244104 11341.80488333
 10732.96306595 10202.26023656  9839.97838568  9489.0205221
  9172.8803216   8927.19110172  8699.79903197  8477.65832986
  8263.75965121  8049.97408843  7850.8793009   7656.54032601]
Run time: 119.8117 seconds
norm counter: 636206304


'''