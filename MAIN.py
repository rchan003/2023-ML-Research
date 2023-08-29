from Universal import*
from analytic import*
from files_dfs import*
from main_fxns import*
from kernel_ridge import*
from taylor import*
from neural import*

# Importing data 
data_df = load_dataframe('formatted_data.txt')
fxns_df = load_dataframe('MORSE.txt')

# Params
N = 100
random_state = 2
range_ = 2
param_grid = {'alpha': [1e-15],  #pow_space(10, -14, 2), 
              'gamma': [1] #[1, 2, 3, 5, 6]   #pow_space(10, -4, 1)}
}


# MAE = 13458.7606 with alpha = 1e-14, gamma = 10, N_molecules = 106, N_pts = 100
# MAE = 135490.03 w/ alpha = 1e-14, gamma = 7, N_molecules = 106, N_pts = 50

# MAE = 0.000187 w/ alpha = 1e-15, gamma = 1, N_molecules = 1, N_pts = 50, N_features = 4
# MAE = 0.000224 w/ only 1 features instead
# MAE = 0.0001966 w/out scaling y 

# MAE = 57.8222 w/ LogScaler(), N_pts = 50 
all_avg_mae = []
all_avg_mpe = []
molecules = fxns_df['molec'].values.tolist()

print(len(molecules))

krr_model = KernelRidge(kernel='rbf', alpha=1e-15, gamma=1)
for molec in molecules:

    indiv_df = extract_N_points(data_df, target_col='molec', target_vals=[molec])
    avg_mae, avg_mpe = main_run_prediction(N, indiv_df, krr_model, display_plot=False, save_results=False)

    all_avg_mae.append(avg_mae)
    all_avg_mpe.append(avg_mpe)

    print(f'{molec}: {avg_mae}')

overall_avg = np.mean(all_avg_mae)
print('Overall Avg:', overall_avg)
print('percent error: ', np.mean(all_avg_mpe))

for i in range(len(molecules)): 
    print(f'{molecules[i]}, {all_avg_mae[i]}, {all_avg_mpe[i]}')


plt.bar(molecules, all_avg_mae)
plt.show()
#main_RNO_trials(fxns_df, 'fxn', krr_model, range_vals=np.arange(0.125, 4.125,0.125), o_vals=np.arange(1,50,1), trials_per_iter=1, o2n_scale=10)

#main_2d_3d_intersections(10, search_dir='/Users/rachelchan/Desktop/new_RNO/take4/', save_dir='/Users/rachelchan/Desktop/work/code_3/')
'''
# Data
#data_df = extract_N_points(data_df, N, sep_range=(0,4),  return_full_dataframe=True)
data_df = extract_N_points(data_df, N, target_col='molec', target_vals=['KBr'], sep_range=(0,4),  return_full_dataframe=True)
X = data_df[['sep [A]', 'r nat [A]', 'k force [N/m]', 'Do [kcal/mol]']].values
y = data_df[['V [kcal/mol]']].values

sc_X = StandardScaler()
X_sc = sc_X.fit_transform(X)

sc_y = StandardScaler()
y_sc = sc_y.fit_transform(y)

# Data setup
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(X_sc, y_sc, test_size=0.2, random_state=22)

y_train_sc = np.reshape(y_train_sc, (-1))
y_test_sc = np.reshape(y_test_sc, (-1))
y_test = np.reshape(y_test, (-1))
y_train = np.reshape(y_train, (-1))

# Perform gridsearch
cv = KFold(n_splits=5, random_state=1, shuffle=True) 
grid_search = GridSearchCV(krr_model, param_grid, cv=cv, scoring='neg_mean_absolute_error')
grid_search.fit(X_train_sc, y_train_sc)

# Get the best model and best hyperparameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
n_splits = grid_search.n_splits_

# Predict y & reshape 
y_pred_sc = best_model.predict(X_test_sc)
y_pred = sc_y.inverse_transform(y_pred_sc)

print(y_test)
print(y_pred_sc)
print(y_pred)

# Calculate the Mean Squared Error (MSE) for evaluation
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)

rmse_sc = mean_squared_error(y_test_sc, y_pred_sc, squared=False)
mae_sc = mean_absolute_error(y_test_sc, y_pred_sc)

# Display Results 
box_print(f'Hypertuning results', 26)
print("Best hyperparameters:", best_params)
print("Best model:", best_model)
print("Number Splits: ", n_splits)
print("RMSE on the test set:", rmse)
print("MAE on the test set:", mae)

print("Scaled RMSE: ", rmse_sc)
print("Scaled MAE: ", mae_sc)


# Running Prediction
#range_vals = np.arange(0.5, 4, 0.5)
#o_vals = np.arange(1, 50, 1)

#main_2d_3d_intersections(5, search_dir='/Users/rachelchan/Desktop/new_RNO/take2/', save_dir='/Users/rachelchan/Desktop/work/code_3/')

#main_RNO_trials(fxns_df, 'fxn', krr_model, range_vals, o_vals, sep_start = 0.001, trials_per_iter = 1, o2n_scale=5, save_folder='/Users/rachelchan/Desktop/new_RNO/')

# do another trial where expansion point is centered on axis 

'''