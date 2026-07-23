image = [
 [1,2,3,4],
 [5,6,7,8],
 [9,10,11,12],
 [13,14,15,16]
]

kernel = [
 [-1,-1,-1],
 [-1, 8,-1],
 [-1,-1,-1]
]

def convolve2d(input_matrix,kernel):
    n = len(input_matrix)

    k = len(kernel)

    output_size = (n - k) + 1
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
                    sum_val += input_matrix[i+m][j+n_k] * kernel[m][n_k]
        
            output[i][j] = sum_val
    
    return output


def relu(matrix):

    rows = len(matrix)

    for i in range(rows):
        for j in range(len(matrix[i])):
            matrix[i][j] = max(0,matrix[i][j])

    return matrix

feature_map = [
 [1,5,2,4],
 [8,3,7,1],
 [2,9,6,5],
 [0,3,1,2]
]

def max_pool(feature_map,stride = 2,pooling_size = 2):

    n = len(feature_map)

    output_size = n // stride 
    output = []

    for i in range(output_size):
        row = []
        for j in range(output_size):
            row.append(0)
        
        output.append(row)

    for i in range(output_size):
        for j in range(output_size):
            max_val = feature_map[i*stride][j*stride]

            for m in range(pooling_size):
                for n_k in range(pooling_size):
                    max_val = max(max_val,feature_map[i * stride + m][j * stride + n_k])
                
            output[i][j] = max_val
    
    return output

feature_map = [
 [1,2],
 [3,4]
]

def flatten_2d(feature_map):
    rows = len(feature_map)

    output = []

    for i in range(rows):
        for j in range(len(feature_map[i])):

            output.append(feature_map[i][j])
    
    return output

inputs = [1,2]
weights = [0.5,0.3]
bias = 1

def dense_layer(inputs,weights,bias):
    dense_sum = bias 

    for i in range(len(inputs)):
        dense_sum += inputs[i]*weights[i]
    
    return dense_sum

import math

def sigmoid(x):
    class_score = 1 / (1 + math.exp(-x))

    return class_score

def softmax(logits):

    exp_values = []
    class_sum = sum(logits)

    for logit in logits:
        exp_values.append(math.exp(logit))
    
    exp_sum = sum(exp_values)

    probabilities = []

    for value in exp_values:
        probabilities.append(value/exp_sum)
    
    return probabilities








