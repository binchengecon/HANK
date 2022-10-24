import numpy as np

# Number of parameters to estimate for
numParam = 2

# Data extraction and set up
pFile = open('price - history .txt ', 'r ')

pInter = pFile . readlines()

# Get number of terms in time series
M = len(pInter)

p = []

for i in range(0, M):
    p. append(np. float64(pInter[i]))

# If estimating 3 parameters , alpha , beta_1 , beta_2
if(numParam == 3):
    # f_M column calculation

    f_M = np. zeros(numParam)

    for t in range(2, M):
        f_M = f_M + np. array([p[t], p[t - 1] * p[t], p[t - 2] * p[t]])

    f_M = 1 / (M - 2) * f_M

    # R_M matrix calculation

    r_M = np. zeros((numParam, numParam))

    for t in range(2, M):
        r_M = r_M + np. outer(
            np. array([1, p[t - 1], p[t - 2]]),
            np. array([1, p[t - 1], p[t - 2]]). transpose())

    r_M = 1 / (M - 2) * r_M

    # Calculate optimal parameters

    theta = np. linalg .inv(r_M). dot(f_M)

    # If estimating 2 parameters , alpha and beta_1
    # Gamma set to -1 ( forces sinusoid )
if(numParam == 2):

    # f_M column calculation

    f_M = np. zeros(numParam)

    for t in range(2, M):
        f_M = f_M + np. array([p[t] + p[t - 2],
                               p[t - 1]*(p[t] + p[t - 2])])

    f_M = 1 / (M - 2) * f_M

    # R_M matrix calculation

    r_M = np. zeros((numParam))

    for t in range(2, M):
        r_M = r_M + np. outer(
            np. array([1, p[t - 1]]),
            np. array([1, p[t - 1]]). transpose())

    r_M = 1 / (M - 2) * r_M
    # Calculate optimal parameters

    theta = np. linalg .inv(r_M). dot(f_M)
