from Universal import*

X_scaler = StandardScaler()
y_scaler = LogScaler()
print(X_scaler)
print(y_scaler)

'''
sc_names = LabelEncoder()
encoded_names = np.repeat(sc_names.fit_transform(molecules), N)


# Formatting data 
X_df = data_df.iloc[:, 2:6]
X = X_df.values
#X = np.reshape(data_df['sep [A]'].values.tolist(), (-1,1))
y = data_df['V [kcal/mol]'].values

y_test_true = train_test_split(X, y, test_size=0.2, random_state=random_state)[3]


sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = np.ravel(sc_y.fit_transform(np.reshape(y, (-1,1))))

# Hypertuning: 
param_grid = {
    'kernel': ['rbf'],
    'C': [1e4, 1e5],
    'gamma': [3, 5, 10],
    'epsilon': [1e-4, 1e-5] #pow_space(10, -3, 2)
}

model = SVR()
svr_model = hypertuning(model, X, y, molecules, param_grid, display_params=True)[1]
#svr_model = SVR(**param_grid)

param_grid = {
    'hidden_layer_sizes': [(16,), (516,), (256,)],
    'alpha': [1e-4, 1e-3, 1e-2],
    'learning_rate_init': [1e-4, 1e-3],
    'activation': ['logistic'],
    'solver': ['adam'],
    'max_iter': [300]
}

neural_model = neural_hypertuning(X, y, param_grid, splits=1)[0]

# Splitting data into test & train 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
encoded_names_test = train_test_split(encoded_names, test_size=0.2, random_state=random_state)[1]

# Preforming regression 
t0 = time.time()
svr_model.fit(X_train, y_train)
svr_fit_time = time.time() - t0
y_pred = svr_model.predict(X_test)

t0 = time.time()
neural_model.fit(X_train, y_train)
neural_fit_time = time.time() - t0



# Scaled Results
sep_scaled = np.transpose(X)[0]
sep_scaled_test = np.transpose(X_test)[0]
scaled_mae = mean_absolute_error(y_test, y_pred)

# Unscaled Results
y_pred_unscaled = sc_y.inverse_transform(y_pred)

unscaled_mae = mean_absolute_error(y_test_true, y_pred_unscaled)

# Error results 
print('Scaled MAE: ', scaled_mae)
print('True MAE (original y_test & unscaled y_pred): ', unscaled_mae)


### 3D PLOTTING ###
fig = plt.figure()
ax = fig.add_subplot(projection='3d')  # (x: sep, y: encoded name, z: predicted energy)
fig.suptitle('SVR results')

ax.plot(sep_scaled_test, encoded_names_test, y_pred_unscaled, '.', alpha=0.2, color='blue', label='Prediction')
ax.plot(sep_scaled_test, encoded_names_test, y_test_true, '.', alpha = 0.2, color='red', label='Reference')


ax.set_xlabel('Separation [A]')
ax.set_ylabel('Encoded Molecule')
ax.set_zlabel('Potential [V]')
ax.legend(loc='best')

plt.show()


### PLOTTING ###
fig1 = plt.figure(figsize=(12, 9))
ax1 = fig1.add_subplot(221)      
ax2 = fig1.add_subplot(222)     
ax3 = fig1.add_subplot(212)        
fig1.subplots_adjust(left=0.08, right=0.92, bottom=0.09, top=0.875,wspace=0.3, hspace=0.4)

# Setting titles
fig1.suptitle(f"SVR Results for N={N} Data Points per Molecule for {len(molecules)} Molecules", fontsize=14)
ax1.title.set_text('Scaled Results')
ax2.title.set_text('Unscaled Results')
ax3.title.set_text('Unscaled MAE per Molecule')

# plotting values 
indiv_mae = []

N_test = int(len(y_test) / len(molecules))

y_test = np.reshape(y_test, (len(molecules), N_test))
y_pred = np.reshape(y_pred, (len(molecules), N_test))
y_test_true = np.reshape(y_test_true, (len(molecules), N_test))
y_pred_unscaled = np.reshape(y_pred_unscaled, (len(molecules), N_test))

for i in range(len(molecules)):
    linreg1 = scipy.stats.linregress(y_pred[i], y_test[i])
    ax1.plot(np.linspace(min(y_pred[i]), max(y_pred[i])), linreg1.intercept + linreg1.slope*np.linspace(min(y_test[i]), max(y_test[i])), label=f'{molecules[i]} r2={round(linreg1.rvalue,3)}')

    linreg2 = scipy.stats.linregress(y_pred_unscaled[i], y_test_true[i])
    ax2.plot(np.linspace(min(y_pred_unscaled[i]), max(y_pred_unscaled[i])), linreg2.intercept + linreg2.slope*np.linspace(min(y_test_true[i]), max(y_test_true[i])), label=f'{molecules[i]} r2={round(linreg2.rvalue,3)}')

    ax1.plot(y_pred[i], y_test[i], '.')
    ax2.plot(y_pred_unscaled[i], y_test_true[i], '.')


    indiv_mae.append(mean_absolute_error(y_test_true[i], y_pred_unscaled[i]))

labels = [f'{molecules[i]}\n{round(indiv_mae[i],3)}' for i in range(len(molecules))]
ax3.bar(labels, indiv_mae)


# setting axes labels
ax1.set_xlabel('Prediction')
ax1.set_ylabel('Expected')
ax1.legend(loc='best')

ax2.set_xlabel('Predicted Energy')
ax2.set_ylabel('Expected Energy')
ax2.legend(loc='best')

ax3.set_xlabel('Molecules')
ax3.set_ylabel('MAE [kcal/mol]')


plt.show()
'''
'''
param_grid = {
    'hidden_layer_sizes': [(16,), (516,), (256,)],
    'alpha': [1e-4, 1e-3, 1e-2],
    'learning_rate_init': [1e-4, 1e-3],
    'activation': ['logistic'],
    'solver': ['adam'],
    'max_iter': [300]
}

N = 20
save_folder = f'/Users/rachelchan/Desktop/neural_hypertune/N={N}/'

# general hypertuning 
X, y = extract_N_points(N, data_df, molecules, target_col='molec', target_vals=['AlBr'], Xy_format=True)
#best_model, best_params, best_mae, best_rmse = neural_hypertuning(X, y, param_grid, splits=1)



print(X)
print(y)

alphas = gammas = pow_space(10, -20, 1, 1)
param_grid = {
    'gamma': gammas,
    'alpha': alphas
}

model = KernelRidge(kernel='rbf')

best_params, best_model, rmse, mae = hypertuning(model, X, y, molecules, param_grid, test_size=0.2, random_state=22, n_splits=5, scoring='neg_mean_absolute_error', display_params=True, display_results=True)
'''


'''
# Example data
feature_matrix1 = np.array([[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]])

feature_matrix2 = np.array([[10, 20, 30],
                            [40, 50, 60],
                            [70, 80, 90]])

target_matrix = np.array([[100, 200],
                          [300, 400],
                          [500, 600]])

# Combine feature matrices row-wise
combined_feature_matrix = np.concatenate((feature_matrix1, feature_matrix2), axis=1)

# Reshape the data to match the desired structure
num_samples = combined_feature_matrix.shape[0]
num_features = combined_feature_matrix.shape[1]
num_targets = target_matrix.shape[1]

# Create an array for each target's features
target_features = np.tile(combined_feature_matrix, (num_targets, 1)).reshape(num_samples, num_targets, num_features)

print('Combined feature matrix: ', combined_feature_matrix)
print('Num samples, features, targets: ', num_samples, num_features, num_targets)
print('Target features: ', target_features)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(target_features, target_matrix, test_size=0.2, random_state=42)

# Reshape the data to match the input format for the model
X_train = X_train.reshape(-1, num_targets * num_features)
X_test = X_test.reshape(-1, num_targets * num_features)
y_train = y_train.reshape(-1, num_targets)
y_test = y_test.reshape(-1, num_targets)

# Create and train a machine learning model (e.g., Linear Regression)
model = KernelRidge(kernel='rbf', alpha=1e-5, gamma=5)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model (you can use different evaluation metrics based on your problem)
print(y_pred)
print(y_test)

mse = np.mean((y_pred - y_test) ** 2)
print("Mean Squared Error:", mse)
'''

'''
# Example data
feature_matrix1 = np.array([[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]])

feature_matrix2 = np.array([[10, 20, 30],
                            [40, 50, 60],
                            [70, 80, 90]])

target_matrix = np.array([[100, 200],
                          [300, 400],
                          [500, 600]])

# Combine feature matrices row-wise
combined_feature_matrix = np.concatenate((feature_matrix1, feature_matrix2), axis=1)

print(combined_feature_matrix)

# Reshape the data to match the desired structure
num_samples = combined_feature_matrix.shape[0]
num_features = combined_feature_matrix.shape[1]
num_targets = target_matrix.shape[1]

# Create an array for each target's features
target_features = np.tile(combined_feature_matrix, (num_targets, 1)).reshape(num_samples, num_targets, num_features)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(target_features, target_matrix, test_size=0.2, random_state=42)

# Reshape the data to match the input format for the model
X_train = X_train.reshape(-1, num_features)
X_test = X_test.reshape(-1, num_features)
y_train = y_train.reshape(-1, num_targets)
y_test = y_test.reshape(-1, num_targets)

# Create and train a machine learning model (e.g., Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model (you can use different evaluation metrics based on your problem)
mse = np.mean((y_pred - y_test) ** 2)
print("Mean Squared Error:", mse)
'''