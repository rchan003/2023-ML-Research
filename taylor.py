from Universal import *

"""
1. Find min/max points as well as inflection points 
2. At each of these points find the taylor expansion (a = point)
3. Average each of these taylor expansions 
"""

def sym_taylor_function(sym_fxn, order, a):
    # a = point evaluating at
    r = sp.Symbol('r')
    sym_taylor = sym_fxn.evalf(subs={r: a})
    taylor_coeff = [sym_taylor]
    #print(sym_taylor, ", order = ", 0, "\n")
    for i in range(1, order + 1):
        sym_fxn = sp.diff(sym_fxn, r)
        #print(sym_fxn)
        coeff = sym_fxn.evalf(subs={r: a}) / math.factorial(i)
        taylor_coeff.append(coeff)
        term = coeff * sp.Pow(r - a, i)
        sym_taylor = sym_taylor + term
    lamb_taylor = sp.lambdify(r, sym_taylor)
    return sym_taylor, lamb_taylor, taylor_coeff

def taylor_eval_ranges(a, ranges, nb_pts, lamb_f, lamb_t):
    mse = []
    mae = []
    r_axes = []
    error_axes = []

    for i in tqdm(range(0, len(ranges)), initial = 0, desc ="Taylor - Evaluating Ranges"):
        r_axis = x_axis_centered(a, ranges[i], step=(ranges[i]/nb_pts)) if a-ranges[i]/2>=0 else x_axis_function(ranges[i], step=ranges[i]/nb_pts, start=0)
        y_true = lamb_f(r_axis)
        y_taylor = lamb_t(r_axis)

        r_axes.append(r_axis)
        error_axes.append(y_true-y_taylor)
        mse.append(mean_squared_error(y_true,y_taylor))
        mae.append(mean_absolute_error(y_true,y_taylor))
    
    return mae, mse, error_axes, r_axes

def taylor_eval_orders(a, o_vals, r_axis, f, lamb_f):
    mse = []
    mae = []
    error_axes = []
    lamb_taylors = []

    r = sp.Symbol('r')
    y_true = lamb_f(r_axis)

    for i in tqdm(range(0, len(o_vals)), initial = 0, desc ="Taylor - Evaluating Orders"):
        f_new = sp.sympify(f)
        lamb_t = sym_taylor_function(f_new, o_vals[i], a)[1]
        y_taylor = lamb_t(r_axis)
        
        error_axes.append(y_true - y_taylor)
        lamb_taylors.append(lamb_t)
        mse.append(mean_squared_error(y_true, y_taylor))
        mae.append(mean_absolute_error(y_true, y_taylor))

    return mse, mae, error_axes, lamb_taylors


