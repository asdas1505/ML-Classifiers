# KNN Regression

import pandas as pd
import numpy as np
import math
import operator

def Distance_Calculation(pt1, pt2): 
       
    return np.linalg.norm(pt1-pt2)

def knn_model(train_data, test_point, k): 
     
    if len(train_data) < k:
        return "ERROR: k is greater than training data"
    
    distances = {}
    sort = {}
    
    for i in range(len(train_data)):
        dist = Distance_Calculation(test_point, train_data.iloc[i])
        distances[i] = dist[0]
    sortdist = sorted(distances.items(), key=operator.itemgetter(1))
    neighbors = []
    for i in range(k):
        neighbors.append(sortdist[i][0])
    
    return sum(neighbors)/k     
        
