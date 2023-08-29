from Universal import*
from files_dfs import*
from analytic import*
from taylor import*
from kernel_ridge import*
from neural import extract_N_points


def main_run_prediction(N, full_data, model, sep_range = (0.001,2), scale_X = True, scale_y = True, X_scaler = StandardScaler(), y_scaler=StandardScaler(), test_size = 0.2, random_state=2, display_plot=True, save_results=False, X_cols = ['sep [A]', 'r nat [A]', 'k force [N/m]', 'Do [kcal/mol]'], y_cols = ['V [kcal/mol]'], base_dir='/Users/rachelchan/Desktop/pred_results', molec_col = 'molec'):
    # Extracting Data 
    data = extract_N_points(full_data, N, sep_range=sep_range, return_full_dataframe=True)
    df_cols = full_data.columns.tolist()
    molecules = np.unique(data[molec_col].values.tolist())

    # Creating a scaled version
    data_sc = data.copy()
    data_sc[X_cols] = X_scaler.fit_transform(data[X_cols].values)
    data_sc[y_cols] = y_scaler.fit_transform(data[y_cols].values)

    # Splitting Data 
    data_train, data_test, data_sc_train, data_sc_test = train_test_split(data.values, data_sc.values, test_size=test_size, random_state=random_state)
    train = pd.DataFrame(data_train, columns=df_cols)
    test = pd.DataFrame(data_test, columns=df_cols)
    train_sc = pd.DataFrame(data_sc_train, columns=df_cols)
    test_sc = pd.DataFrame(data_sc_test, columns=df_cols)

    # Extracting X & y 
    X_train = train_sc[X_cols].values if scale_X else train[X_cols].values
    y_train = train_sc[y_cols].values.tolist() if scale_y else train[y_cols].values.tolist()
    X_test = test_sc[X_cols].values if scale_X else test[X_cols].values
    y_test = test_sc[y_cols].values.tolist() if scale_y else test[y_cols].values.tolist()

    # Preforming Fit 
    t0 = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    t_fit_pred = time.time() - t0

    # creating save directory
    model_name = return_model_name(model)
    save_folder = create_path(base_dir+f'/N={N}/{model_name}')

    r_range = sep_range[1] - sep_range[0]
    
    # Scaled & Unscaled Results
    box_print(f'Results for {model_name}')
    print(f'Model Parameters: {model.get_params()}')
    
    if scale_y:
        y_pred_sc = y_pred
        y_test_sc = y_test
        mae_sc = mean_absolute_error(y_test_sc, y_pred_sc)

        y_pred = y_scaler.inverse_transform(y_pred)
        y_test = np.reshape(test[y_cols].values.tolist(), (-1))

        print(y_pred.shape)
        print(y_test.shape)

        mae = mean_absolute_error(y_test, y_pred)
        print(f'MAE (avg of all molecules): {mae}\nScaled: {mae_sc}')

    else: # scale_y == False
        mae = mean_absolute_error(y_test, y_pred)
        print(f'MAE (avg of all molecules): {mae,3}')


    ### 3D PLOT ###
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(projection='3d')  # (x: sep, y: encoded name, z: predicted energy)
    ###ax.view_init(60, 35)

    fig.suptitle('Error of Each Testing Point')
    ax.set_xlabel('Separation [A]')
    ax.set_ylabel('Encoded Molecule')
    ax.set_zlabel('Potential [V]')

    
    y_err = np.subtract(np.reshape(y_pred, (-1)), np.reshape(y_test, (-1)))
    y_test_dict = {
        'molec': np.reshape(test['molec'].values.tolist(), (-1)), 
        'encoded name': np.reshape(LabelEncoder().fit_transform(test['molec'].values.tolist()), (-1)),
        'sep [A]':np.reshape(test['sep [A]'].values.tolist(), (-1)), 
        'y_err': np.reshape(y_err, (-1)), 
        'y_pred': np.reshape(y_pred, (-1)), 
        'y_test': np.reshape(y_test, (-1)),
        'score': np.reshape(test['Score (max V [kcal/mol])'].values.tolist(), (-1))
        }
    #print(y_test_dict)
    
    #print(y_test_dict)
    #for key, value in y_test_dict.items():
    #    entry_length = len(value)
    #    print(f"Key: {key}, Entry Length: {entry_length}")

    y_test_df = pd.DataFrame(y_test_dict)
    if scale_y:
        y_test_df['y_err_sc'] = np.subtract(np.reshape(y_pred_sc, (-1)), np.reshape(y_test_sc, (-1)))
        y_test_df['y_pred_sc'] = np.reshape(y_pred_sc, (-1))
        y_test_df['y_test_sc'] = np.reshape(y_test_sc, (-1))


    over, under = y_test_df[y_test_df['y_err'] > 0], y_test_df[y_test_df['y_err'] <= 0]

   #ax.plot(over['sep [A]'].values.tolist(), over['encoded name'].values.tolist(), over['y_pred'].values.tolist(), 'd', label='Predicted Value')
    #ax.plot(over['sep [A]'].values.tolist(), over['encoded name'].values.tolist(), over['y_test'].values.tolist(), 'd', label='Actual Value')
    def log_tick_formatter(val, pos=None):
        return f"$10^{{{int(val)}}}$"

    # You still have to take log10(Z) but thats just one operation
    ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    #ax.plot_trisurf(y_test_df['sep [A]'].values.tolist(), y_test_df['encoded name'].values.tolist(), np.log10(y_test_df['y_err'].values.tolist()), color='black')

    ax.plot(over['sep [A]'].values.tolist(), over['encoded name'].values.tolist(), np.log10(over['y_err'].values.tolist()), '.', alpha=0.3, color='blue', label='Over Estimated')
    ax.plot(under['sep [A]'].values.tolist(), under['encoded name'].values.tolist(), np.log10(np.absolute(under['y_err'].values.tolist())), '.', alpha = 0.3, color='red', label='Under Estimated')

    ax.legend(loc='best')

    if save_results == True:
        fig.savefig(save_folder+'3D_plot.png')
        plt.close(fig)


    ### 2D PLOTTING ###
    fig1 = plt.figure(figsize=(12, 9))
    fig1.subplots_adjust(left=0.08, right=0.92, bottom=0.09, top=0.875,wspace=0.3, hspace=0.4)
    
    if scale_X == True or scale_X == True:
        ax1 = fig1.add_subplot(221)   
        ax2 = fig1.add_subplot(222)     
        ax3 = fig1.add_subplot(212)   

        ax1_title = scaled_results_title(scale_X, scale_y, X_scaler, y_scaler)

        ax1.title.set_text(ax1_title)
        ax1.set_xlabel('Prediction')
        ax1.set_ylabel('Expected')
    
    else:
        ax2 = fig1.add_subplot()
        ax3 = fig1.add_sublot()     

    # Setting titles
    fig1.suptitle(f"{model_name} Accuracy with N={N} Data Points per Molecule and range={r_range} for {len(molecules)} Molecules")
    ax2.title.set_text('Predicted Energy vs Actual Energy')
    ax3.title.set_text(f'MAE per Molecule (Average = {round(mae,3)})')

    # plotting values 
    indiv_mae = []
    indiv_mpe = []
    r2_sc = []
    r2 = []

    unique_test_molec = np.unique(y_test_df['molec'].values.tolist())
    unique_encode = np.unique(y_test_df['encoded name'].values.tolist())

    for molec in unique_test_molec:
        df_indiv = y_test_df[y_test_df['molec']==molec].sort_values(by='sep [A]')

        #print(df_indiv)
        if scale_y == True:
            molec_pred_sc = df_indiv['y_pred_sc'].values.tolist()
            molec_test_sc = df_indiv['y_test_sc'].values.tolist()
            
            linreg1 = scipy.stats.linregress(molec_pred_sc, molec_test_sc)
            x_sc = np.linspace(min(molec_pred_sc), max(molec_pred_sc))

            ax1.plot(x_sc, linreg1.intercept + linreg1.slope*x_sc)
            r2_sc.append(round(linreg1.rvalue, 3))
            
            
            if molec == unique_test_molec[-1]:
                ax1.plot(molec_pred_sc, molec_test_sc, '.', label=f'Average r2 = {round(np.mean(r2_sc),3)}')
            else:
                ax1.plot(molec_pred_sc, molec_test_sc, '.')
                

        molec_pred = df_indiv['y_pred'].values.tolist()
        molec_test = df_indiv['y_test'].values.tolist()
        linreg2 = scipy.stats.linregress(molec_pred, molec_test)
        x_ = np.linspace(min(molec_pred), max(molec_pred))

        ax2.plot(x_, linreg2.intercept + linreg2.slope*x_)
        r2.append(round(linreg2.rvalue, 3))

        if molec == unique_test_molec[-1]:
            ax2.plot(molec_pred, molec_test, '.', label=f'Average r2 = {round(np.mean(r2),3)}')
            
        else:
            ax2.plot(molec_pred, molec_test, '.')

        indiv_mae.append(mean_absolute_error(molec_test, molec_pred))
        indiv_mpe.append(mean_absolute_percentage_error(molec_test, molec_pred))

    ax2.text(0.95, 0.05, f'Average r2 = {round(np.mean(r2), 6)}', ha='right', va='bottom', transform=ax.transAxes, fontsize=12, color='gray', bbox=dict(facecolor='white', alpha=0.5))

    ax3.bar(unique_encode, indiv_mae)
    # Rotation of the bars names
    plt.xticks(unique_encode, unique_test_molec, rotation=90, fontsize=5)


    # setting axes labels
    if scale_X or scale_y == True:
        ax1.legend(loc='best')
        ax1.text(0.95, 0.05, f'Average r2 = {round(np.mean(r2_sc), 6)}', ha='right', va='bottom', transform=ax.transAxes, fontsize=12, color='gray', bbox=dict(facecolor='white', alpha=0.5))

    ax2.set_xlabel('Predicted Energy [kcal/mol]')
    ax2.set_ylabel('Expected Energy [kcal/mol]')
    ax2.legend(loc='best')

    ax3.set_xlabel('Molecules')
    ax3.set_ylabel('MAE [kcal/mol]')
    ax3.set_yscale('log')

    if display_plot == True:
        plt.show()
    if save_results == True:
        fig1.savefig(save_folder+'2D_plot'+'.png')
        plt.close(fig1)

        columns = ['model', 'fit+predict time [s]', 'N', 'range', 'overall mae']
        save_df = pd.DataFrame(list(zip([model], [t_fit_pred], [N], [r_range], [mae])), columns =columns)
        save_df.to_csv(save_folder+'Summary_Results.txt', sep='\t', encoding='utf-8', index=False)

        y_test_df.to_csv(save_folder+'test_results.txt', sep='\t', encoding='utf-8', index=False)

    plt.close(fig1)
    plt.close(fig)

    return np.mean(indiv_mae), np.mean(indiv_mpe)


def main_hypervalid(data_df, molecules, ht_splits = 5, cv_splits = 5, N_tests = np.arange(10,101,5), method='KRR', display=False):
    '''
    Currently only works for KRR & SVR
    '''
    box_print('Running main_hypervalid, please wait will take a while...')

    if method == 'KRR':
        model_gauss=GaussianKernelRidgeRegression()
        model_laplace=LaplacianKernelRidgeRegression()
        model_poly=KernelRidge(kernel='polynomial')
        model_rbf=KernelRidge(kernel='rbf')

    if method == 'SVR':
        model_gauss=GaussianKernelSVR()
        model_laplace=LaplacianKernelSVR()
        model_poly=SVR(kernel='polynomial')
        model_rbf=SVR(kernel='rbf')

    # hyperparams
    sigmas = np.arange(1,51,1)
    degrees = np.arange(0,15,1)
    alphas = gammas = pow_space(10, -15, 1, 1)
    coef0s = np.arange(0,25,0.5)

    # defining models
    gau_model = []
    lap_model = []
    pol_model = []
    rbf_model = []

    gau_params = []
    lap_params = []
    pol_params = []
    rbf_params = []

    gau_cv = []
    lap_cv = []
    pol_cv = []
    rbf_cv = []

    gau_mae = []
    lap_mae = []
    pol_mae = []
    rbf_mae = []

    gau_rmse = []
    lap_rmse = []
    pol_rmse = []
    rbf_rmse = []

    for nb_x_vals in N_tests:
        box_print(f'!\nN = {nb_x_vals}\n!')
        X, y = extract_N_points(nb_x_vals, data_df) ##################################

        # Gauss Hypertuning + CV
        param_grid = {'sigma': sigmas, 'alpha': alphas}
        best_params, best_model, rmse, mae = hypertuning(model_gauss, X, y, molecules=molecules, param_grid=param_grid, n_splits=ht_splits, display_params=display)
        scores = cross_valid(best_model, X, y, n_splits=cv_splits)
        model = 'Gaussian'


        gau_model.append(model)
        gau_cv.append(abs(scores))
        gau_mae.append(mae)
        gau_rmse.append(rmse)
        gau_params.append(best_params)

        # Laplace Hypertuning + CV
        param_grid = {'gamma': gammas, 'alpha': alphas}
        best_params, best_model, rmse, mae = hypertuning(model_laplace, X, y, molecules=molecules, param_grid=param_grid, n_splits=ht_splits, display_params=display)
        scores = cross_valid(best_model, X, y, n_splits=cv_splits)
        model = 'Laplacian'

        lap_model.append(model)
        lap_cv.append(abs(scores))
        lap_mae.append(mae)
        lap_rmse.append(rmse)
        lap_params.append(best_params)

        # Polynomial Hypertuning + CV
        param_grid = {'degree': degrees, 'coef0': coef0s}
        best_params, best_model, rmse, mae = hypertuning(model_poly, X, y, molecules=molecules, param_grid=param_grid, n_splits=ht_splits, display_params=display)
        scores = cross_valid(best_model, X, y, n_splits=cv_splits)
        model = 'Polynomial'

        pol_model.append(model)
        pol_cv.append(abs(scores))
        pol_mae.append(mae)
        pol_rmse.append(rmse)
        pol_params.append(best_params)

        # RBF Hypertuning + CV
        param_grid = {'gamma': gammas, 'alpha': alphas}
        best_params, best_model, rmse, mae = hypertuning(model_rbf, X, y, molecules=molecules, param_grid=param_grid, n_splits=ht_splits, display_params=display)
        scores = cross_valid(best_model, X, y, n_splits=cv_splits)
        model = 'RBF'

        rbf_model.append(model)
        rbf_cv.append(abs(scores))
        rbf_mae.append(mae)
        rbf_rmse.append(rmse)
        rbf_params.append(best_params)


    # saving results in df
    df_res = pd.DataFrame(list(zip(gau_model, lap_model, pol_model, rbf_model, N_tests, gau_mae, lap_mae, pol_mae, rbf_mae)), columns =['gauss model', 'laplace model', 'poly model', 'rbf model', 'N_vals', 'gauss mae', 'laplace mae', 'poly mae', 'rbf mae'])
    df_res.to_csv('main_hypervalid_results.txt',sep='\t', index=False, encoding='utf-8')

    # plotting 
    fig = plt.figure(figsize=(12, 9))
    fig.suptitle('Paramters tested during Hypervalidation')
    ax1 = fig.add_subplot(212)   
    ax2 = fig.add_subplot(221)      
    ax3 = fig.add_subplot(222)     

    ax1.plot(N_tests, gau_cv, marker = 'o', label='Gaussian')
    ax1.plot(N_tests, lap_cv, marker = 'o', label='Laplacian')
    ax1.plot(N_tests, pol_cv, marker = 'o', label='Polynomial')
    ax1.plot(N_tests, rbf_cv, marker = 'o', label='RBF')

    ax1.set_xlabel(f'N Data Points per Molecule, with {len(molecules)} Total Molecules')
    ax1.set_ylabel('MAE')
    ax1.set_title(f'K-Fold Cross Validation MAE Score for Each Kernel using Hypertuned Parameters from GridSearchCV \nhypertuning splits = {ht_splits}, cross validation splits = {cv_splits}')
    ax1.set_yscale('log')
    ax1.legend(loc='best')

    ax2.set_yticks([-1,0,1])
    ax2.set_yticklabels(['sigma', 'degree', 'coef0'])
    ax2.plot(sigmas, len(sigmas)*[-1], '.', label=f'# Values Tested: {len(sigmas)}')
    ax2.plot(degrees, len(degrees)*[0], '.', label=f'# Values Tested: {len(degrees)}')
    ax2.plot(coef0s, len(coef0s)*[1], '.', label=f'# Values Tested: {len(coef0s)}')
    ax2.legend(loc='best')

    ax3.set_yticks([0,1])
    ax3.set_yticklabels(['alpha', 'gamma'])
    ax3.plot(alphas, len(alphas)*[0], '.', label=f'# Values Tested: {len(alphas)}')
    ax3.plot(gammas, len(gammas)*[1], '.', label=f'# Values Tested: {len(gammas)}')
    ax3.set_xscale('log')
    ax3.legend(loc='best')

    plt.show()

    return 


def main_eval_traintest_split(model, fxns_df, emp_name, X_min=0, X_max=3, N_vals=np.arange(5,250,5), train_size_vals=np.arange(0.05, 1.00, 0.05)):
    # Setup
    fin_best_size = [] 

    # Initiate figure 
    fig, (ax1,ax2) = plt.subplots(2)
    fig.suptitle('Comparing Accuracy of Best Model for Different Training/Test Splits')
    ax1.set_xlabel('Train Size')
    ax1.set_ylabel('MAE Score')
    ax1.set_yscale('log')

    for N in N_vals:
        # Setup 
        X_list = np.linspace(X_min,X_max,N)
        X, y = create_Xy_matrix(X_list, fxns_df, emp_name)
        scores = []

        for size in train_size_vals:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= size, random_state=None)     
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            scores.append(mean_absolute_error(y_test, y_pred))

        size_min = train_size_vals[np.where(scores == min(scores))[0]]
        fin_best_size.append(size_min)
        ax1.plot(train_size_vals, scores)

    # Plotting 
    ax2.set_xlabel('N Value')
    ax2.set_ylabel('Best Size')
    ax2.plot(N_vals, fin_best_size, 'o', label='Best Size using MAE Scoring')
    ax2.plot(N_vals, [np.mean(fin_best_size)]*len(N_vals), label=f'Mean Best Size = {np.mean(fin_best_size)}')
    ax2.legend(loc='best')

    plt.show()
    return


def main_RNO_trials(fxns_df, fxn_col, model, range_vals, o_vals, sep_start = 0.001, trials_per_iter = 3, o2n_scale=5, save_folder='/Users/rachelchan/Desktop/new_RNO/'):
    '''
    Creates & saves intersection points as txt file & resulting plot for each combination of range, N, and O 
    Note: Assigned range_vals using following format before 
        range_vals = range(0, 4+0.05, 0.05)
        range_vals[0] = 0.01
    '''
    # loading data
    molecules = fxns_df['molec'].values.tolist()
    functions = fxns_df[fxn_col].values.tolist()
    r_nat_vals = fxns_df['r_nat [A]'].values.tolist()

    # Setup
    N_vals = o_vals * o2n_scale

    # creating save folder 
    num_files = len(next(os.walk(save_folder))[1]) + 1
    print(f'Number folders in R_trials: {num_files-1}')
    temp0 = save_folder+f'take{num_files}'
    os.makedirs(temp0)
    new_save_folder = temp0 + '/'

    for range_ in range_vals:
        temp = new_save_folder+f'R_width_{range_}'
        os.makedirs(temp)
        save_folder2 = temp + '/'

        for i in range(trials_per_iter):
            print('\n')
            box_print(f'TRIAL #{i+1}\n Range = {range_}')

            # creating folder for saving
            num_files2 = len(next(os.walk(save_folder2))[1]) + 1
            print(f'Number folders in image folder: {num_files2-1}')
            temp2 = save_folder2+f'trial_{num_files2}'
            os.makedirs(temp2)
            trial_path = temp2 + '/'

            # Setup
            sep_axis = np.linspace(sep_start, range_, max(N_vals))
            intersect_x_mae = []
            intersect_y_mae = []
            intersect_x_mse = []
            intersect_y_mse = []

            for i in range(len(molecules)):
                # function parameters
                molec = molecules[i]
                f = sp.sympify(functions[i])
                r_nat = r_nat_vals[i]
                box_print(f'Molecule: {molec} ({i+1}/{len(molecules)})', 26)

                # Sympy Functions
                r = sp.Symbol('r')
                lamb_f = sp.lambdify(r, f)

                # evaluating orders (taylor)
                a = r_nat
                t_mse, t_mae, t_error_axes, lamb_taylors = taylor_eval_orders(a, o_vals, sep_axis, f, lamb_f)

                # evaluating data sizes (gauss)
                ml_mse, ml_mae, ml_error_axes, r_test_axes, y_pred_axes = krr_eval_N_vals(model, sep_start, range_, N_vals, lamb_f, random_state=None)

                # Intersection Points
                x_mae, y_mae = intersect_pts(o_vals, t_mae, ml_mae)
                x_mse, y_mse = intersect_pts(o_vals, t_mse, ml_mse)
                intersect_x_mae.append(x_mae)
                intersect_y_mae.append(y_mae)
                intersect_x_mse.append(x_mse)
                intersect_y_mse.append(y_mse)

                ### PLOTTING ###
                fig = plt.figure(figsize=(12, 9))
                ax1 = fig.add_subplot(221)      
                ax2 = fig.add_subplot(222)     
                ax3 = fig.add_subplot(212)        
                fig.subplots_adjust(left=0.08, right=0.92, bottom=0.06, top=0.875,wspace=0.3, hspace=0.4)

                # Setting titles
                fig.suptitle(f"ML and Taylor Prediction Accuracy of Interatomic Potential vs Prediction Range for {molec} \n {model}", fontsize=14)
                ax1.title.set_text('Interatomic Potential vs Atom Separation')
                ax2.title.set_text('ΔE = Reference - Prediction vs Atom Separation')
                ax3.title.set_text('Error vs N Data Points and Order')

                # ax1: top left 
                ax1.set_xlabel("r [Å]")
                ax1.set_ylabel("E [kcal/mol]")
                ax1.plot(sep_axis, lamb_f(sep_axis), label="Reference")
                ax1.plot(sep_axis, lamb_taylors[-1](sep_axis),'--' , label=f'Taylor, order = {o_vals[-1]}')
                ax1.plot(r_test_axes[-1], y_pred_axes[-1], '.', label=f'KRR, N = {N_vals[-1]}')
                ax1.plot(a, lamb_f(a), 'd', label = f'Expansion pt = r_eq')
                ax1.legend(loc='best')

                # ax2: top right 
                ax2.set_xlabel("r [Å]")
                ax2.set_ylabel("E [kcal/mol]")
                o_idxs = get_spaced_elements(o_vals, 3, low_lim=15)[1]
                n_idxs= get_spaced_elements(N_vals, 3, low_lim=30, up_lim=100)[1]
                for idx in o_idxs:
                    ax2.plot(sep_axis, t_error_axes[idx], label=f'Order = {o_vals[idx]}')
                for idx in n_idxs[::-1]:
                    ax2.plot(r_test_axes[idx], ml_error_axes[idx], '.', label=f'N = {N_vals[idx]}')
                ax2.legend(loc='best')

                # ax3: bottom left
                ax3.set_xlabel("Order")
                ax3.set_ylabel("E [kcal/mol]")
                ax3.set_xscale('log')
                ax3.set_yscale('log')
                ax3.plot(o_vals, t_mse, '--' ,label="Taylor MSE", color='C0')
                ax3.plot(o_vals, t_mae, label="Taylor MAE", color='C0')
                
                mae_textstr = 'MAE:'
                for i in range(len(x_mae)):
                    mae_textstr = mae_textstr + f' ({round(x_mae[i],2)}, {round(y_mae[i],2)})'
                mse_textstr = 'MSE:'
                for i in range(len(x_mse)):
                    mse_textstr = mse_textstr + f' ({round(x_mse[i],2)}, {round(y_mse[i],2)})'

                textstr = f'Intersections\nFormat: (Order, Error) or ({1/o2n_scale}N, Error) \n' + mae_textstr + '\n' + mse_textstr
                ax3.text(0.05, 0.2, textstr, transform=ax3.transAxes, verticalalignment='top', fontsize=8)
                ax33 = ax3.twiny()
                ax33.set_xscale('log')
                ax33.set_xlabel('N Data Points')
                ax33.plot(N_vals, ml_mse, '--',label="KRR MSE", color='C3')
                ax33.plot(N_vals, ml_mae, label="KRR MAE", color='C3')
                ax33.legend(loc='lower right')
                ax3.legend(loc='upper right')

                # Save Figure
                fig.savefig(trial_path+molec+'.png')
                plt.close(fig)

            # Intersect dataframe 
            intersect_df = pd.DataFrame(list(zip(molecules,intersect_x_mae,intersect_y_mae, intersect_x_mse, intersect_y_mse)), columns =['molec', 'x mae', 'y mae', 'x mse', 'y mse'])

            # parameters dataframe 
            trial_params_df = pd.DataFrame(list(zip([model], [o_vals], [N_vals],[sep_axis], [o2n_scale], [molecules], [range_])), columns =['model', 'o_vals', 'N_vals', 'sep_axis', 'O to N scale', 'molecules', 'range'])

            # saving results 
            intersect_df.to_csv(trial_path+f'intersect.txt', sep='\t', index=False, encoding='utf-8')
            trial_params_df.to_csv(trial_path+f'params.txt', sep='\t', index=False, encoding='utf-8')

    return 


def main_2d_3d_intersections(scale, search_dir, save_dir):
    # paths
    num_files = len(os.listdir(save_dir))
    save_dir = save_dir+f'take{num_files}/'

    def extract_float_from_folder_names(search_dir, save_dir, folder_pattern=re.compile(r'R_width_([0-9]\.[0-9][0-9][0-9])'), target_file_name = 'intersect', save_as_txt=False):
        float_numbers = []
        dfs = []

        for folder_name in os.listdir(search_dir):
            match = folder_pattern.search(folder_name)
            if match:
                float_number = float(match.group(1))
                float_numbers.append(float_number)

                search_sub_dir = search_dir+folder_name+'/'
                print('\n\n',folder_name)
                save_temp = save_dir+f'width_{float_number}/'
                if save_as_txt == True:
                    os.makedirs(save_temp)
                dfs.append(df_from_dir(search_sub_dir, save_dir=save_temp, file_name = target_file_name, save_as_txt = save_as_txt)[0])

        return float_numbers, dfs

    def normalize_hues(vals_1D, scale='log', palette='PiYG'):
        if scale == 'log':
            max_pow = int(math.log10(max(vals_1D)))
            min_pow = int(math.log10(min(vals_1D)))
            powers = list(range(min_pow-1, max_pow+2)) # to account for rounding

            ranges = []
            for i in range(len(powers)-1):
                r = [math.pow(10,powers[i]), math.pow(10,powers[i+1])]
                ranges.append(r)
        hues = sns.color_palette(palette, n_colors=len(ranges))
        return hues[::-1], ranges

    def assign_hue(pt_val, ranges, hues):
        for i in range(len(ranges)):
            r = ranges[i]
            low_lim_inclusive = r[0]
            up_lim = r[1]
            if pt_val >= low_lim_inclusive and pt_val < up_lim:
                return hues[i]
        
        return (0,0,0)

    # USING
    r_width_vals, dfs = extract_float_from_folder_names(search_dir, save_dir)

    print(r_width_vals)
    print(dfs)

    paired_data = zip(r_width_vals, dfs)
    sorted_pairs = sorted(paired_data, key=lambda pair: pair[0])
    sorted_float_list, sorted_data_frame_list = zip(*sorted_pairs)

    r_axis = sorted_float_list
    df_list = sorted_data_frame_list

    # looping
    n_axis = []
    mae_axis = []
    r_axis_long = []

    mae_tot = []

    for i in range(len(r_axis)):
        r = r_axis[i]
        df = df_list[i]

        # Sample data for the 2D plot
        x_mae = np.array(df['x mae'].values)
        y_mae = np.array(df['y mae'].values)

        x_mse = df['x mse'].values.tolist()
        y_mse = df['y mse'].values.tolist()

        def extract(X, Y):
            x_ = [ast.literal_eval(x) for x in X]
            y_ = [ast.literal_eval(y) for y in Y]
            x_new = []
            y_new = []
            for i in range(len(x_)):
                x = x_[i]
                y = y_[i]
                if type(x) == list:
                    for i in range(len(x)):
                        x_new.append(x[i])
                else:
                    x_new.append(x)
                if type(y) == list:
                    for i in range(len(y)):
                        y_new.append(y[i])
                else:
                    y_new.append(y)
            return x_new, y_new

        xa, ya = extract(x_mae, y_mae)
        xs, ys = extract(x_mse, y_mse)

        n_axis.extend(xa)
        mae_axis.extend(ya)
        r_axis_long.extend([r]*len(xa))

    # 3d plot
    hues, ranges = normalize_hues(mae_axis)
    print('hue ranges: ', ranges)

    fig = plt.figure()
    plt.title(f'Intersection Points between ML and Pertubation Accuracy')
    ax = fig.add_subplot(projection='3d')  
    for i in range(len(r_axis_long)):
        n = n_axis[i]
        r = r_axis_long[i]
        mae = mae_axis[i]
        color_rbg = assign_hue(mae, ranges, hues)
        ax.plot(n, r, mae, '*', color=color_rbg)

    # Set labels for each axis
    ax.set_ylabel('Prediction Range [Å]')
    ax.set_xlabel(f'N Derivatives\nN Data Points * 1/{scale}')
    ax.set_zlabel('MAE [kcal/mol]')

    plt.show()


    # 2d plot  
    fig2, ax2 = plt.subplots()
    plt.title(f'Intersection Points between ML and Pertubation Accuracy')
    ax2.set_xlabel(f'N Derivatives')
    ax2.set_ylabel('Range [Å]')

    for i in range(len(r_axis_long)):
        n = n_axis[i]
        r = r_axis_long[i]
        mae = mae_axis[i]
        color_rbg = assign_hue(mae, ranges, hues)
        ax2.plot(n, r, '*', color=color_rbg)

    x_ticks = np.arange(0,30,5)
    x_ticks[0] = 1
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([f'{num}' for num in x_ticks])

    ax3 = ax2.twiny()
    x_ticks = np.arange(0,30*5, 5*5)
    x_ticks[0] = 5
    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels([f'{num}' for num in x_ticks])
    ax3.set_xlabel('N Data Points')
    plt.show()

    # plotting color pallet 
    sns.palplot(hues)
    plt.show()

    return
