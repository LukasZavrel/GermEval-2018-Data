import numpy as np
def get_train_data(filename):
    X  = []
    y_task1 = []
    y_task2 = []
    
    with open(filename) as file:
        for line in file:
            tweet = line.rstrip('\n').split('\t')
            X.append(tweet[0])
            y_task1.append(tweet[1])
            y_task2.append(tweet[2])
    
    return np.asarray(X), np.asarray(y_task1), np.asarray(y_task2)
