import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import time
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

k=20


norm_counter = 0

def run_kmeans(X, initial_centers):
    global norm_counter
    kmeans = KMeans(n_clusters=len(initial_centers), init=np.array(initial_centers), n_init=1)
    kmeans.fit(X)
    norm_counter += kmeans.n_iter_ * len(X) * len(initial_centers)

    return kmeans.labels_, kmeans.inertia_, kmeans.cluster_centers_


start_time = time.time()

centers= X[np.random.choice(X.shape[0], k, replace=False)]

predicted_labels, obj_val, centers = run_kmeans(X, centers)




end_time = time.time()


# Plotting the results with predicted labels
plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis', s=50)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Global KMeans Clustering (Predicted Labels)')
plt.show()


#print("Final cluster centers:", final_centers)
print("Objective value:", obj_val)
print(f"Run time: {end_time - start_time:.4f} seconds")
print(f"norm counter: {norm_counter}")

'''1:
Data loaded successfully with shape: (683, 9)
Objective value: 15010.381550992943
Run time: 0.0844 seconds
norm counter: 30735


2:

  '''