import sys
import numpy as np 
import matplotlib 

# output = [input[0]*weight1[0] + input[1]*weight1[1] + input[2]*weight1[2] + input[3]*weight1[3] + bias1,
# 		  input[0]*weight2[0] + input[1]*weight2[1] + input[2]*weight2[2] + input[3]*weight2[3] + bias2, 
#           input[0]*weight3[0] + input[1]*weight3[1] + input[2]*weight3[2] + input[3]*weight3[3] + bias3]

inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0], 
           [0.5, -0.91, 0.26, -0.5], 
           [-0.26, -0.27, 0.17, 0.87]]
weights = np.array(weights).T
# weights =  weights.transpose()
biases = [2, 3, 0.5]
output = np.dot(inputs, np.array(weights).T) + biases   
print(output)
'''
layer_output = []
for nw, nb in zip(weights, biases):
    n_output = 0
    for n_input, weight in zip(inputs, nw):
        n_output += n_input*weight
    n_output += nb
    layer_output.append(n_output)

print(layer_output)
'''
