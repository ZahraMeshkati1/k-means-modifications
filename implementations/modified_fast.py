import os
os.environ["OMP_NUM_THREADS"] = "1"
import time
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

folder_path = r'D:\EL studies\project\real2'

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

K=15


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
    _, obj_val, centers = run_kmeans(X, centers)
    #obj_values = [objective_function(X, centers)]
    obj_values = [obj_val]

    for k in range(2, K + 1):
        # Store all points with their Bj values
        bj_tuples = []
        for aj in X:
            if any((aj == center).all() for center in centers):
                continue
            I = [pt for pt in X if any(np.linalg.norm(pt - aj) < np.linalg.norm(pt - c) for c in centers)]
            Bj = sum(np.linalg.norm(pt - np.min([np.linalg.norm(pt - c) for c in centers])) - np.linalg.norm(pt - aj) for pt in I)
            norm_counter += len(centers) * len(I)  # Increment norm counter
            bj_tuples.append((Bj, aj))

        # Sort by Bj and select the top 20% of points with max Bj values
        bj_tuples.sort(reverse=True, key=lambda x: x[0])
        top_percent = bj_tuples[:max(1, int(len(bj_tuples) * 0.20))]

        # Apply k-means on the selected points to find the best new center
        min_inertia = float('inf')
        best_new_center = None
        for _, point in top_percent:
            temp_centers = np.vstack([centers, point])
            _, inertia, _ = run_kmeans(X, temp_centers)
            if inertia < min_inertia:
                min_inertia = inertia
                best_new_center = point
        
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
plt.title('Improved Global KMeans Clustering (Predicted Labels)')
plt.show()



#print("Final cluster centers:", final_centers)
print("Objective values:", np.array(obj_values))
print(f"Run time: {end_time - start_time:.4f} seconds")
print(f"Total number of norm computations: {norm_counter}")
