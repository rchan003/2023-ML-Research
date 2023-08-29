from Universal import*
from analytic import*
from kernel_ridge import*
from files_dfs import*

def neural_hypertuning(X, y, param_grid, splits = 5, test_size=0.2, random_state=22):
    box_print('Performing Neural Hypervalidation')
    print(f'Shape of X: {X.shape}')

    num_iter = num_dict_iterations(param_grid)
    combinations = product(*param_grid.values())

    best_mae = float('inf')
    best_rmse = float('inf')
    best_params = {}

    counter = 1
    for combination in combinations:
        params = dict(zip(param_grid.keys(), combination))
        model = MLPRegressor(**params)

        box_print(f'Iteration {counter}/{num_iter}')
        print(params)
        counter += 1

        if splits == 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            rmse_score = mean_squared_error(y_test, y_pred, squared=False)
            mae_score = mean_absolute_error(y_test, y_pred)

            print('Current MAE', mae_score)

            if mae_score < best_mae:
                best_mae = mae_score
                best_rmse = rmse_score
                best_params = params

        else:
            mae_scores = []
            rmse_scores = []
            kf = KFold(n_splits=splits)
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
        
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                rmse_scores.append(mean_squared_error(y_test, y_pred, squared=False))
                mae_scores.append(mean_absolute_error(y_test, y_pred))
            
            avg_mae = np.mean(mae_scores)
            avg_rmse = np.mean(rmse_scores)
            
            print('Current MAE', avg_mae)

            if avg_mae < best_mae:
                best_mae = avg_mae
                best_rmse = avg_rmse
                best_params = params

    box_print(f'! RESULTS ! \nN = {len(y)} Iterations = {num_iter}')
    print("Best Hyperparameters:", best_params)
    print("Best Mean Absolute Error:", best_mae)

    best_model = MLPRegressor(**best_params)
    return best_model, best_params, best_mae, best_rmse


def extract_V_max(df_full, molecules, V_col_name = 'V [kcal/mol]', molec_col_name = 'molec', save_as_txt=False):
    V_maxs = []
    for molecule in molecules:
        df_specific = df_full[df_full[molec_col_name] == molecule]
        V_vals = df_specific[V_col_name]
        V_maxs.append(max(V_vals))
        
    V_df = pd.DataFrame(dict(zip(molecules, V_maxs)))
    if save_as_txt == True:
        V_df.to_csv('max_potentials.txt', sep='\t', index=False, encoding='utf-8')

    return V_df


def neural_molecule_hypertune(data_df, molecules, general_model, param_grid, N, splits=1, test_size = 0.2, random_state = 22, save_fig=False, save_folder='/Users/rachelchan/Desktop/neural_hypertune/', save_df = False):
    models = []
    best_maes = []
    best_rmses = []
    test_maes = []
    test_rmses = []
    mae_gens = []
    rmse_gens = []

    num_trials = len(next(os.walk(save_folder))[1]) + 1
    os.makedirs(save_folder + f'trail{num_trials}')
    save_folder = save_folder + f'trail{num_trials}/'
    
    for molecule in molecules:
        X, y = extract_N_points(N, data_df, target_col='molec', target_vals=[molecule])

        best_params, best_mae, best_rmse = neural_hypertuning(X, y, param_grid=param_grid, splits=splits, test_size=test_size, random_state=random_state)[1:]
        
        model = MLPRegressor(**best_params)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, splits=splits)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        general_model.fit(X_train, y_train)
        y_pred_gen = general_model.predict(X_test)
        mae_gen = mean_absolute_error(y_test, y_pred_gen)
        rmse_gen = mean_squared_error(y_test, y_pred_gen, squared=False)

        models.append(model)
        best_maes.append(best_mae)
        best_rmses.append(best_rmse)
        test_maes.append(mae)
        test_rmses.append(rmse)
        mae_gens.append(mae_gen)
        rmse_gens.append(rmse_gen)

        if save_fig == True:    
            fig = plt.figure(figsize=(9, 5))
            ax = fig.add_subplot()

            fig.suptitle(f'Comparing the Best Model and the General Model for {molecule} with N={N}')
            ax.text(0.95, 0.5, f'Specfic Model:\n{model}\n\nGeneral Model:\n{general_model}', transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', fontsize=8)
            ax.plot(X, y, '--', label='Reference')
            ax.plot(X_test, y_pred_gen, 'o', fillstyle='full', color='black',label=f'General Model: \nMAE = {round(mae_gen,3)}, RMSE = {round(rmse_gen,3)}')
            ax.plot(X_test, y_pred, '*', fillstyle='full', color='red', label=f'Specific Model: \nMAE = {round(mae,3)}, RMSE = {round(rmse,3)}')
            ax.legend(loc='best')
            ax.set_xlabel('Sep [A]')
            ax.set_ylabel('V [kcal/mol]')
            
            plt.savefig(save_folder+f'{molecule}')

    keys = ['molecule', 'model', 'mae', 'rmse', 'test mae', 'test rmse', 'general model', 'gen mae', 'gen rmse']
    values = [molecules, models, best_maes, best_rmses, test_maes, test_rmses, general_model, mae_gens, rmse_gens]

    ht_df = pd.DataFrame(dict(zip(keys, values)))
    if save_df == True:
        ht_df.to_csv(save_folder+f'N={N}.txt', sep='\t', index=False, encoding='utf-8')

    return ht_df


def neural_param_grid(N=None, tester='general'):

    param_grid_indiv = {
        'hidden_layer_sizes': [(64,), (32,), (16,)],
        'alpha': [1e-3, 1e-2, 1e-1],
        'learning_rate_init': [1e-4, 1e-2, 1, 2],
        'batch_size': [2, 10, 50],
        'activation': ['relu'],
        'solver': ['lbfgs', 'adam'],
        'max_iter': [200, 250, 300],
    }

    general_best_params_N50 = {
        'hidden_layer_sizes': (1024,), 
        'alpha': 0.01, 
        'learning_rate_init': 1, 
        'batch_size': 100, 
        'activation': 'relu', 
        'solver': 'lbfgs', 
        'max_iter': 250
    }

    param_grid_big = {
        'hidden_layer_sizes': [(1024,), (512,), (256,)],
        'alpha': [1e-3, 1e-2, 1e-1],
        'learning_rate_init': [1e-3, 1e-2, 1],
        'batch_size': [10, 100, 1000],
        'activation': ['relu'],
        'solver': ['lbfgs', 'adam'],
        'max_iter': [200, 250, 300],
    }
    
    if tester == 'general':
        return param_grid_big
    elif tester == 'individual':
        return param_grid_indiv
    
    if N == 50:
        return general_best_params_N50

    return

