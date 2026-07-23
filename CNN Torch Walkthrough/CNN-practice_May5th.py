input_matrix = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
]

kernel = [
    [1, 0],
    [0, -1]
]

def convolution2d(input_matrix,kernel,stride):
    n = len(input_matrix)
    k = len(kernel)

    output_size = ((n - k)//stride) + 1
    output = []

    for i in range(output_size): 
        row = []
        for j in range(output_size):
            row.append(0)
        output.append(row)

    for i in range(output_size):
        for j in range(output_size):  
            sum_val = 0

            for m in range(k):
                for n_k in range(k):
                    sum_val += input_matrix[i*stride+m][j*stride+n_k] * kernel[m][n_k]
            
            output[i][j] = sum_val

    return output

def relu_func(output):

    for i in range(len(output)):
        for j in range(len(output)):
            output[i][j] = max(0,output[i][j])
    
    return output

feature_map = [
    [1, 3, 2, 0],
    [4, 6, 5, 1],
    [7, 2, 8, 3],
    [1, 5, 2, 9]
]

def max_pool(feature_map,window_size = 2,stride = 2):

    n = len(feature_map)

    output_size = n // window_size
    output = []

    
    for i in range(output_size):
        row = []
        for j in range(output_size):

            max_val = feature_map[i * stride][j * stride]


            for m in range(window_size):

                for n_k in range(window_size):

                    val = feature_map[i* stride +m][j * stride + n_k]

                    if val > max_val:

                        max_val = val

            
            row.append(max_val)

        output.append(row)
    
    return output






def avg_pool(feature_map,pooling):
    n = len(feature_map)

    output_size = n // pooling
    output = []

    for i in range(output_size):
        row = []
        for j in range(output_size):
            sum_val = 0
        
            for m in range(pooling):
                for n_k in range(pooling):
                    sum_val += feature_map[i * pooling + m][j * pooling +n_k]
                
            
            avg_val = sum_val / (pooling * pooling)

            row.append(avg_val)
        
        output.append(row)

    return output

def global_avg_pool(feature_map):
    n = len(feature_map)

    sum_val = 0
    for i in range(n):
        for j in range(n):
            sum_val += feature_map[i][j]
        
        avg_val = sum_val /(n*n)
    
    return avg_val  

def flatten(feature_map):

    n = len(feature_map)
    output = []

    for i in range(n):
        for j in range(n):
            output.append(feature_map[i][j])
    
    return output

flattened = [2,5,1,3]
weights = [0.2,0.5,-0.1,0.8]
bias = 1

def dense_layer_cnn(flattened,weights,bias):

    if len(flattened) != len(weights):
        return None

    class_val = 0 + bias
    for i in range(len(flattened)):
        class_val += flattened[i] * weights[i]
    
    return class_val

import math

def sigmoid(logit):

    # Applies sigmoid equation to convert
    # raw score into probability between 0 and 1
    probability = 1 / (1 + math.exp(-logit))

    return probability  

import math

def softmax(logits):

    # Store exponentials of each logit
    exp_vals = []

    # Compute e^(logit) for every class score
    for logit in logits:
        exp_vals.append(math.exp(logit))

    # Sum all exponentials
    total = sum(exp_vals)

    probabilities = []

    # Normalize each exponential into probability
    for val in exp_vals:
        probabilities.append(val / total)

    return probabilities 




    







               
        


