import os
import numpy as np

path = os.path.join('D:/Eye-Tracker/data/gan/3/facial_landmark/000001.txt')

with open(path, "r") as f:
    landmark = f.read().split('\n')

# landmark = np.array(landmark, dtype=np.float32)

print(landmark[:-1])