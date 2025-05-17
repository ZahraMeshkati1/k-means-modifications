import os
os.environ["OMP_NUM_THREADS"] = "1"
import time
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
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



K = 2

# Global counter for norm computations
norm_counter = 0

# Compute mean
def compute_mean(data):
    return np.mean(data, axis=0)

# Objective function
def objective_function(data, centers):
    global norm_counter
    sum_squares = 0
    for point in data:
        min_dist = np.min([np.linalg.norm(point - center) ** 2 for center in centers])
        norm_counter += len(centers)  # Increment counter by the number of centers
        sum_squares += min_dist
    return sum_squares 

# Run k-means
def run_kmeans(X, initial_centers):
    global norm_counter
    kmeans = KMeans(n_clusters=len(initial_centers), init=np.array(initial_centers), n_init=1)
    kmeans.fit(X)
    norm_counter += kmeans.n_iter_ * len(X) * len(initial_centers)  # Increment the norm counter
    return kmeans.labels_, kmeans.inertia_, kmeans.cluster_centers_

# Global k-means algorithm
def global_kmeans(X, K):
    global norm_counter
    centers = np.array([compute_mean(X)])  # Start with the mean of all points as the first center
    _, obj_val , centers = run_kmeans(X, centers)
    #obj_values = [objective_function(X, centers)]
    obj_values = [obj_val]

    for k in range(2, K + 1):
        max_Bj = -float('inf')
        best_new_center = None
        for aj in X:
            if any((aj == center).all() for center in centers):
                continue
            I = [pt for pt in X if any(np.linalg.norm(pt - aj) < np.linalg.norm(pt - c) for c in centers)]
            Bj = sum(np.linalg.norm(pt - np.min([np.linalg.norm(pt - c) for c in centers])) - np.linalg.norm(pt - aj) for pt in I)
            norm_counter += len(centers) * len(I)  # Increment norm counter
            if Bj > max_Bj:
                max_Bj = Bj
                best_new_center = aj
        
        centers = np.vstack([centers, best_new_center])
        predicted_labels, obj_val, centers = run_kmeans(X, centers)

        #obj_values.append(objective_function(X, centers))
        obj_values.append(obj_val)
    return predicted_labels, centers, obj_values

# Measure run time
start_time = time.time()

predicted_labels, final_centers, obj_values = global_kmeans(X, K)

end_time = time.time()

# Plot the results with predicted labels
plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis', s=50)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Modified Global KMeans Clustering (Predicted Labels)')
plt.show()

#print("Final cluster centers:", final_centers)
print("Objective values:", np.array(obj_values))
print(f"Run time: {end_time - start_time:.4f} seconds")
print(f"norm counter: {norm_counter}")


'''
2:
Data loaded successfully with shape: (683, 9)
Objective values: [48443.0658858  19323.20489982]
Run time: 5.7261 seconds
norm counter: 207573


5:
Objective values: [48443.0658858  19323.20489982 18124.49840569 15173.22724724
 13770.55677491]
Run time: 41.0915 seconds
norm counter: 3516907

10:
Objective values: [48443.0658858  19323.20489982 18124.49840569 15173.22724724
 13770.55677491 13550.53268689 12805.6352597  12603.02730492
 12431.96375175 12352.28815687]
Run time: 133.7034 seconds
norm counter: 18056027


15:
Objective values: [48443.0658858  19323.20489982 18124.49840569 15173.22724724
 13770.55677491 13550.53268689 12805.6352597  12603.02730492
 12431.96375175 12352.28815687 12305.15174671 11910.59938558
 11092.96494301 10406.55033359  9933.40217971]
Run time: 259.5474 seconds
norm counter: 44338028


20:
Objective values: [48443.0658858  19323.20489982 18124.49840569 15173.22724724
 13770.55677491 13550.53268689 12805.6352597  12603.02730492
 12431.96375175 12352.28815687 12305.15174671 11910.59938558
 11092.96494301 10406.55033359  9933.40217971  9767.40150785
  8918.74452788  8701.57354773  8533.61923095  8465.90586099]
Run time: 430.7401 seconds
norm counter: 82687319

'''