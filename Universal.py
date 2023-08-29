### IMPORTING PACKAGES & TOOLS ###
import math
import time
import os
import re
import glob
import ast
import fnmatch

import tensorflow as tf
import pandas as pd
import numpy as np
import sympy as sp
import scipy as scipy
import seaborn as sns
import matplotlib.pyplot as plt

from textwrap import wrap
import matplotlib.ticker as mticker
from tqdm import tqdm
from itertools import product
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.kernel_ridge import KernelRidge

'''
General usage functions
'''

class LogScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.log1p(X)
    
    def inverse_transform(self, X_scaled):
        return np.expm1(X_scaled)
    
    def fit_transform(self, X, y=None):
        return self.transform(X)

### COUNTER FUNCTIONS ###
def num_dict_iterations(dictionary):
    total_iter_count = 1
    for value in dictionary.values():
        if isinstance(value, list):
            total_iter_count = len(value) * total_iter_count
    return total_iter_count


### DISPLAY FUNCTIONS ###
def box_print(sentence=None, width=26):
    #print('\n')
    print('+-' + '-' * width + '-+')
    for line in wrap(sentence, width):
        print('| {0:^{1}} |'.format(line, width))
    print('+-' + '-'*(width) + '-+')
    return 

def matrix_print(sentence=None, array=[], square=True, num_rows = None):
    if sentence != None: 
        print(sentence)
    # determining number of rows
    if square == True and num_rows == None:
        num_rows = num_row_ele = math.floor(math.sqrt(len(array)))
    else: 
        num_row_ele = int(len(array)/num_rows)

    # separating array into lines
    lines = []
    for i in range(0, num_rows):
        line = []
        for j in range(0, num_row_ele):
            idx = i*num_row_ele + j
            line.append(array[idx])
        lines.append(line)

    # remaining elements
    if idx != len(array) - 1:
        rem_arr = array[idx+1:]
        if len(rem_arr) > num_row_ele/2:
            lines.append(rem_arr)
        else:

            lines[j].extend(rem_arr)

    for line in lines:
        print(line)

    return 

def start_timer(event='event', display=True):
    if display == True:
        print("\nStarting Timer for " + event)
        return time.time()
    return time.time()

def end_timer(t0, event):
    delta_t = time.time() - t0
    min_passed = int(delta_t/60)
    if min_passed >= 1:
        sec_passed = delta_t - (60 * min_passed)
        print(f'Time elapsed for {event}: {min_passed} minutes {round(sec_passed,2)} seconds')
    else:
        print(f'Time elapsed for {event}: {round(delta_t,2)} seconds')
    return

def scientific_notation(number, decimal_points=2):
    '''
    Returns string of number in scientific notation
        Example: number = 12345, decimal_points = 2 --> '1.23e+04'
    '''
    num = "{:.{}e}".format(number, decimal_points)
    return num

def return_model_name(model):
    '''
    Returns name of model instance
        MLPRegressor:   'Neural'
        SVR:            'SVR'
        KernelRidge:    'KRR'
        Other models:   'Unrecognized Model'
    '''
    if isinstance(model, MLPRegressor):
        return 'Neural'
    elif isinstance(model, SVR):
        return 'SVR'
    elif isinstance(model, KernelRidge):
        return 'KRR'
    else:
        return 'Unrecognized Model'

def scaled_results_title(scale_X=True, scale_y=True, X_scaler=StandardScaler(), y_scaler=StandardScaler()):
    if scale_X == True & scale_y == True:
        if X_scaler.__class__ is y_scaler.__class__:
            label = f'Comparison scaled with {X_scaler}'
        else: 
            label=f'Comparison with X scaled by {X_scaler} and y by {y_scaler}'

    else:
        if scale_X == True:
            label=f'Comparison with X scaled by {X_scaler}'
        else:
            label=f'Comparison with y scaled by {y_scaler}'

    return label

### X & Y DATA FUNCTIONS ###
def x_axis_function(width, step, start):
    """Returns time_axis with step size = time_step.
    Duration input must be in seconds. Best to use small time_step"""
    num_steps = int(width / step)
    x_axis = [None] * (num_steps + 1)
    i = 0
    for i in range(num_steps + 1):
        x_axis[i] = step * i + start
    return np.array(x_axis)

def x_axis_centered(middle, width, step):
    start = middle - width / 2
    return x_axis_function(width, step, start)


### POW/LOG AXIS FUNCTIONS ###
def pow_space(base, start_pow, end_pow, pow_step=1):
    ''' Return array starting at base^start_pow and ending at base^end_pow,
    with the difference between each element being base^(prev_pow + pow_step)
    '''
    pow_array = []
    pows = np.arange(start_pow, end_pow+pow_step, pow_step)
    for i in range(len(pows)):
        pow_array.append(math.pow(base, pows[i]))

    return pow_array

def get_spaced_elements(array, num_elems, low_lim=None, up_lim=None):
    bot = min(array) if low_lim == None else low_lim
    top = max(array) if up_lim == None else up_lim
    temp = np.array([x for x in array if x >= bot and x <= top])
    out = temp[np.round(np.linspace(0, len(temp) - 1, num_elems)).astype(int)]
    og_idxs = []
    for ele in out:
        og_idxs.append(int(np.where(array == ele)[0]))
    return out, og_idxs

def gen_log_space(limit, n):
    '''
    Returns Numpy array of integers which are spaced evenly on a log scale 
    '''
    limit = limit + 1
    result = [1]
    # avoiding ZeroDivisionError
    if n > 1: 
        ratio = (float(limit) / result[-1]) ** (1.0 / (n - len(result)))
    while len(result) < n:
        next_value = result[-1] * ratio
        # if next value is different from previous 
        if next_value - result[-1] >= 1:
            result.append(next_value)
        # if next value is the same as the previous incremement and rescale ratio so remaining values scale correctly
        else:
            result.append(result[-1] + 1)
            ratio = (float(limit) / result[-1]) ** (1.0 / (n - len(result)))
    # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
    return np.array(list(map(lambda x: round(x) - 1, result)), dtype=np.uint64)

def constrain_log_space(N_points, N_upper_limit, N_lower_limit):
    N_long = gen_log_space(N_upper_limit, N_points)
    # constrain N: go to lower limit
    i = 0
    counter = 0
    N_long = np.array(N_long)
    N_long = np.delete(N_long, 0)
    for i in range(0, len(N_long)):
        if N_long[i] < N_lower_limit:
            counter += 1
    N_training = np.array(N_long)
    for i in range(0, counter):
        N_training = np.delete(N_training, 0)
    return N_training

### INTERSECTION FUNCTIONS ###
def linear_line_intersect(A, B, C, D, step, display=False):
    '''
    points = [A, B, C, D]
    line1: A = (x1, y1), B = (x2, y2)
    line2: C = (x1, y1), B = (x2, y2)
    '''
    f1 = linear_line(A, B)
    f2 = linear_line(C, D)

    if display == True:
        print(f'line1: {f1} \nline2: {f2}')
        print(f'A: {A}')
        print(f'B: {B}')
        print(f'C: {C}')
        print(f'D: {D}')

    x_min = max(A[0], C[0])
    x_max = min(B[0], D[0])

    x_ax = np.arange(x_min, x_max, step)
    y1 = f1(x_ax)
    y2 = f2(x_ax)
    
    abs_diff = np.absolute(np.subtract(y1, y2))
    idx = np.where(abs_diff == min(abs_diff))

    x_intersect = float(x_ax[idx])
    y_intersect = float(f1(x_intersect))

    if display==True:
        print(f'Intersection: ({x_intersect}, {y_intersect})')
        plt.title('Finding Intersection between line 1 (A to B) and line 2 (C to D)')
        plt.plot(x_ax, y1, label='f1')
        plt.plot(x_ax, y2, label='f2')
        plt.plot([A[0], B[0]], [A[1], B[1]], 'o', label='(A,B)')
        plt.plot([C[0], D[0]], [C[1], D[1]], 'o', label='(C,D)')
        plt.plot(x_intersect, y_intersect, 'd', label='Intersection')
        plt.legend(loc='best')
        #plt.xscale('log')
        #plt.yscale('log')
        plt.show()

    return x_intersect, y_intersect

def linear_line(A, B):
    x1 = A[0]
    y1 = A[1]
    x2 = B[0]
    y2 = B[1]
    x = sp.Symbol('x')
    m = (y2-y1) / (x2-x1)
    b = y1 - m*x1
    f = m*x + b
    return sp.lambdify(x, f)

def intersect_pts(X, y1, y2, percision = 1e-4, display=False):
    """
    Find the intersection points between two functions.

    Parameters:
        func1 (function): The first function.
        func2 (function): The second function.
        nb_intervals: number of intervals to break func1 and func2 into

    Returns:
        list: List of x-values of the intersection points.
        list: List of y-values of the intersection points.
              If the functions do not intersect or there are multiple intersection points
              at the same location, two empty lists are returned.
    """
    # intersection points
    x_inter = []
    y_inter = []

    # Search each interval for overlap
    for i in range(len(X)-1):
        curr_top = 1 if y1[i] >= y2[i] else 2
        next_top = 1 if y1[i+1] >= y2[i+1] else 2
        
        if curr_top != next_top:
            print('Intersection found! Evaluating...')
            A = (X[i], y1[i])
            B = (X[i+1], y1[i+1])
            C = (X[i], y2[i])
            D = (X[i+1], y2[i+1])
            x_, y_ = linear_line_intersect(A, B, C, D, percision)
            x_inter.append(x_)
            y_inter.append(y_)

    if display == True:
        plt.title('Intersection points')
        plt.plot(X, y1, '--', marker='o', label='y1')
        plt.plot(X, y2, '--', marker='o', label='y2')
        plt.plot(x_inter, y_inter, 'd', label='found_inter')
        plt.legend(loc='best')
        #plt.xscale('log')
        #plt.yscale('log')
        plt.show()
    
    return x_inter, y_inter


