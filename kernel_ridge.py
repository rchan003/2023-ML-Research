from Universal import *
from neural import extract_N_points

class GaussianKernelRidgeRegression(BaseEstimator, RegressorMixin):
    def __init__(self, sigma=1.0, alpha=0.1):
        """
        Initialize the GaussianKernelRidgeRegression model.
        
        Parameters:
            sigma (float): Standard deviation of the Gaussian kernel.
            alpha (float): Regularization parameter for Ridge Regression.
        """
        self.sigma = sigma
        self.alpha = alpha
        self.X_train = None
        self.y_train = None
        self.alpha_y = None

    def _gaussian_kernel(self, X1, X2):
        """
        Compute the Gaussian kernel matrix for Kernel Ridge Regression.
        
        Parameters:
            X1 (numpy array): Input matrix of shape (n1, d) representing the first set of samples.
            X2 (numpy array): Input matrix of shape (n2, d) representing the second set of samples.
            
        Returns:
            K (numpy array): Kernel matrix of shape (n1, n2).
        """
        n1, _ = X1.shape
        n2, _ = X2.shape
        K = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                diff = X1[i] - X2[j]
                K[i, j] = np.exp(-np.dot(diff, diff) / (2 * self.sigma**2))

        return K

    def fit(self, X_train, y_train):
        """
        Fit the Kernel Ridge Regression model using the Gaussian kernel.
        
        Parameters:
            X_train (numpy array): Training input matrix of shape (n_train, d).
            y_train (numpy array): Training target matrix of shape (n_train, num_targets).
        """
        self.X_train = X_train
        self.y_train = y_train

        K_train = self._gaussian_kernel(X_train, X_train)
        n_train = X_train.shape[0]
        I = np.eye(n_train)
        alpha_inv = np.linalg.inv(K_train + self.alpha * n_train * I)
        self.alpha_y = alpha_inv.dot(y_train)

    def predict(self, X_test):
        """
        Make predictions using the fitted model.
        
        Parameters:
            X_test (numpy array): Test input matrix of shape (n_test, d).
            
        Returns:
            y_pred (numpy array): Predicted target matrix for the test samples.
        """
        K_test = self._gaussian_kernel(X_test, self.X_train)
        y_pred = K_test.dot(self.alpha_y)
        return y_pred

    def get_coefficients(self):
        """
        Get the learned coefficients of the model.
        
        Returns:
            alpha_y (numpy array): Learned coefficients.
        """
        return self.alpha_y
    
    def get_params(self, deep: bool = True) -> dict:
        return super().get_params(deep)


class LaplacianKernelRidgeRegression(BaseEstimator, RegressorMixin):
    def __init__(self, gamma=1.0, alpha=1.0):
        """
        Initialize the LaplacianKernelRidgeRegression model.

        Parameters:
            gamma (float): Scale parameter of the Laplacian kernel.
            alpha (float): Regularization parameter for Ridge Regression.
        """
        self.gamma = gamma
        self.alpha = alpha
        self.X_train = None
        self.y_train = None
        self.alpha_y = None

    def _laplacian_kernel(self, X1, X2):
        """
        Compute the Laplacian kernel matrix for Kernel Ridge Regression.

        Parameters:
            X1 (numpy array): Input matrix of shape (n1, d) representing the first set of samples.
            X2 (numpy array): Input matrix of shape (n2, d) representing the second set of samples.

        Returns:
            K (numpy array): Kernel matrix of shape (n1, n2).
        """
        n1, _ = X1.shape
        n2, _ = X2.shape
        K = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                diff = X1[i] - X2[j]
                K[i, j] = np.exp(-self.gamma * np.sum(np.abs(diff)))

        return K

    def fit(self, X_train, y_train):
        """
        Fit the Kernel Ridge Regression model using the Laplacian kernel.

        Parameters:
            X_train (numpy array): Training input matrix of shape (n_train, d).
            y_train (numpy array): Training target matrix of shape (n_train, num_targets).
        """
        self.X_train = X_train
        self.y_train = y_train

        K_train = self._laplacian_kernel(X_train, X_train)
        n_train = X_train.shape[0]
        I = np.eye(n_train)
        alpha_inv = np.linalg.inv(K_train + self.alpha * n_train * I)
        self.alpha_y = alpha_inv.dot(y_train)

    def predict(self, X_test):
        """
        Make predictions using the fitted model.

        Parameters:
            X_test (numpy array): Test input matrix of shape (n_test, d).

        Returns:
            y_pred (numpy array): Predicted target matrix for the test samples.
        """
        K_test = self._laplacian_kernel(X_test, self.X_train)
        y_pred = K_test.dot(self.alpha_y)
        return y_pred

    def get_coefficients(self):
        """
        Get the learned coefficients of the model.

        Returns:
            alpha_y (numpy array): Learned coefficients.
        """
        return self.alpha_y
    
    def get_params(self, deep: bool = True) -> dict:
        return super().get_params(deep)


class GaussianKernelSVR(BaseEstimator, RegressorMixin):
    def __init__(self, sigma=1.0, C=1.0, epsilon=0.1, kernel='rbf'):
        """
        Initialize the GaussianKernelSVR model.
        
        Parameters:
            sigma (float): Standard deviation of the Gaussian kernel.
            C (float): Regularization parameter for SVR.
            epsilon (float): Epsilon-tube parameter for SVR.
            kernel (str): Kernel type for SVR.
        """
        self.sigma = sigma
        self.C = C
        self.epsilon = epsilon
        self.X_train = None
        self.y_train = None
        self.model = None

    def _gaussian_kernel(self, X1, X2):
        """
        Compute the Gaussian kernel matrix.
        
        Parameters:
            X1 (numpy array): Input matrix of shape (n1, d) representing the first set of samples.
            X2 (numpy array): Input matrix of shape (n2, d) representing the second set of samples.
            
        Returns:
            K (numpy array): Kernel matrix of shape (n1, n2).
        """
        n1, _ = X1.shape
        n2, _ = X2.shape
        K = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                diff = X1[i] - X2[j]
                K[i, j] = np.exp(-np.dot(diff, diff) / (2 * self.sigma**2))

        return K

    def fit(self, X_train, y_train):
        """
        Fit the GaussianKernelSVR model using the Gaussian kernel.
        
        Parameters:
            X_train (numpy array): Training input matrix of shape (n_train, d).
            y_train (numpy array): Training target matrix of shape (n_train, num_targets).
        """
        self.X_train = X_train
        self.y_train = y_train

        K_train = self._gaussian_kernel(X_train, X_train)
        self.model = SVR(C=self.C, epsilon=self.epsilon, kernel=self.kernel)
        self.model.fit(K_train, y_train)

    def predict(self, X_test):
        """
        Make predictions using the fitted model.
        
        Parameters:
            X_test (numpy array): Test input matrix of shape (n_test, d).
            
        Returns:
            y_pred (numpy array): Predicted target matrix for the test samples.
        """
        K_test = self._gaussian_kernel(X_test, self.X_train)
        y_pred = self.model.predict(K_test)
        return y_pred
    
    def get_params(self, deep: bool = True) -> dict:
        return super().get_params(deep)


class LaplacianKernelSVR(BaseEstimator, RegressorMixin):
    def __init__(self, gamma=1.0, C=1.0, epsilon=0.1):
        """
        Initialize the LaplacianKernelSVR model.
        
        Parameters:
            gamma (float): Scale parameter of the Laplacian kernel.
            C (float): Regularization parameter for SVR.
            epsilon (float): Epsilon-tube parameter for SVR.
        """
        self.gamma = gamma
        self.C = C
        self.epsilon = epsilon
        self.X_train = None
        self.y_train = None
        self.model = None

    def _laplacian_kernel(self, X1, X2):
        """
        Compute the Laplacian kernel matrix.
        
        Parameters:
            X1 (numpy array): Input matrix of shape (n1, d) representing the first set of samples.
            X2 (numpy array): Input matrix of shape (n2, d) representing the second set of samples.
            
        Returns:
            K (numpy array): Kernel matrix of shape (n1, n2).
        """
        n1, _ = X1.shape
        n2, _ = X2.shape
        K = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                diff = X1[i] - X2[j]
                K[i, j] = np.exp(-self.gamma * np.sum(np.abs(diff)))

        return K

    def fit(self, X_train, y_train):
        """
        Fit the LaplacianKernelSVR model using the Laplacian kernel.
        
        Parameters:
            X_train (numpy array): Training input matrix of shape (n_train, d).
            y_train (numpy array): Training target matrix of shape (n_train, num_targets).
        """
        self.X_train = X_train
        self.y_train = y_train

        K_train = self._laplacian_kernel(X_train, X_train)
        self.model = SVR(C=self.C, epsilon=self.epsilon, kernel='precomputed')
        self.model.fit(K_train, y_train)

    def predict(self, X_test):
        """
        Make predictions using the fitted model.
        
        Parameters:
            X_test (numpy array): Test input matrix of shape (n_test, d).
            
        Returns:
            y_pred (numpy array): Predicted target matrix for the test samples.
        """
        K_test = self._laplacian_kernel(X_test, self.X_train)
        y_pred = self.model.predict(K_test)
        return y_pred
    
    def get_params(self, deep: bool = True) -> dict:
        return super().get_params(deep)


### this is a repeat function, generalize the neural one 
def krr_molecule_hypertune(data_df, molecules, general_model, param_grid, N, save_fig=False, save_folder='/Users/rachelchan/Desktop/krr/', save_df = False):
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

        best_params, model, best_rmse, best_mae = hypertuning(model, X, y, len(molecules), param_grid, display_results=False)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
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


def hypertuning(model, X_matrix, y_matrix, molecules, param_grid, test_size=0.2, random_state=22, n_splits=5, scoring='neg_mean_absolute_error', display_params=False, display_results=True):
    '''
    param_grid = dictionary of form: {'param1': [], 'param2': [], ... }
    '''
    if display_params == True:
        box_print(f'Performing Hypertuning on {model} using GridSearchCV', 35)
        print(f'N Molecules:    {len(molecules)}')
        print(f'N Data Points:  {X_matrix.shape[0]*X_matrix.shape[1]}')
        print(f'N Train Points: {int(X_matrix.shape[0]*X_matrix.shape[1]*(1-test_size))}')
        for key, value in param_grid.items():
                print(f'{key} Values: ', value)
    t0 = start_timer('Hypertuning', display=False)
    
    # Data setup
    X_train, X_test, Y_train, Y_test = train_test_split(X_matrix, y_matrix, test_size=test_size, random_state=random_state)

    # Perform gridsearch
    cv = KFold(n_splits=n_splits, random_state=1, shuffle=True) 
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring)
    grid_search.fit(X_train, Y_train)

    # Get the best model and best hyperparameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    n_splits = grid_search.n_splits_

    # Predict y & reshape 
    Y_pred = best_model.predict(X_test)

    # Calculate the Mean Squared Error (MSE) for evaluation
    rmse = mean_squared_error(Y_test, Y_pred, squared=False)
    mae = mean_absolute_error(Y_test, Y_pred)

    if display_results == True:
        box_print(f'Hypertuning results', 26)
        end_timer(t0, 'Hypertuning')
        print("Best hyperparameters:", best_params)
        print("Best model:", best_model)
        print("Number Splits: ", n_splits)
        print("RMSE on the test set:", rmse)
        print("MAE on the test set:", mae)

    return best_params, best_model, rmse, mae

def cross_valid(model, X, y, scoring='neg_mean_absolute_error', n_splits=5, n_jobs=-1, display=True):
    # computing
    cv = KFold(n_splits=n_splits, random_state=1, shuffle=True)
    scores = cross_val_score(model, X, y, scoring=scoring, cv=cv, n_jobs=n_jobs)

    # display results
    if display == True:
        box_print('Cross Validation Results', 26)
        print("Scoring Method: ", scoring)
        print("Cross Validation Scores: \n", scores)
        print("Average CV Score: ", scores.mean())
        print("Number of CV Scores used in Average: ", len(scores))
    return scores.mean()


def krr_eval_ranges(ranges, lamb_morse, model, n_training, center, random_state=22, train_size=0.8):
    mse = []
    mae = []
    error_axes = []
    r_test_axes = []
    y_pred_axes = []

    for i in range(0,len(ranges)):
        if center - ranges[i]/2 < 0:
            np.delete(ranges,i)  

    for i in tqdm(range(0, len(ranges)), initial = 0, desc ="KRR - Range Training"):
        # setup
        step = ranges[i] / (n_training/train_size)
        r_axis = x_axis_centered(center, ranges[i], step)
        y_true = lamb_morse(r_axis)

        # splitting & fitting
        X_train, X_test, y_train = train_test_split(r_axis.reshape(-1,1), y_true, random_state = random_state, train_size = train_size)[:-1]
        model.fit(X_train, y_train)

        # results 
        X_flat = X_test.reshape(-1)
        y_pred = model.predict(X_test)
        y_true = lamb_morse(X_flat)

        # saving results
        r_test_axes.append(X_flat)
        y_pred_axes.append(y_pred)
        error_axes.append(y_true - y_pred)
        mse.append(mean_squared_error(y_true, y_pred))
        mae.append(mean_absolute_error(y_true, y_pred))

    return mse, mae, error_axes, r_test_axes, y_pred_axes


def krr_eval_N_vals(model, r_start, r_stop, n_axis, lamb_f, train_size = 0.8, random_state=None):
    '''
    Returns 1D arrays (msqe, mae, nb_points_array) and matrix (error_axes)
    Note: only used for 1 molecule at a time, determined by lamb_f, which is molecule specific 
    '''
    mse = []
    mae = []
    error_axes = []
    x_test_axes = []
    y_pred_axes = []
    for i in tqdm(range(0, len(n_axis)), initial = 0, desc ="KRR - Evaluating various N_training"):
        # Creating data 
        X_lin = np.linspace(r_start, r_stop, n_axis[i])
        X = X_lin.reshape(-1,1)
        y = lamb_f(X_lin)

        # splitting & fitting 
        X_train, X_test, y_train = train_test_split(X, y, random_state = random_state, train_size = train_size)[:-1]
        model.fit(X_train, y_train)
        
        # results 
        X_flat = X_test.reshape(-1)
        y_pred = model.predict(X_test)
        y_true = lamb_f(X_flat)

        # saving trial results 
        error_axes.append(y_true-y_pred)
        x_test_axes.append(X_test)
        y_pred_axes.append(y_pred)
        mse.append(mean_squared_error(y_true, y_pred))
        mae.append(mean_absolute_error(y_true, y_pred))

    return mse, mae, error_axes, x_test_axes, y_pred_axes