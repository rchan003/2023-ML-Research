from Universal import*
# k_force data: https://www.sciencedirect.com/science/article/abs/pii/S1386142515302808?via%3Dihub

### DATA CREATION FUNCTIONS ###
def truncate_y(df, max_y='largest', min_y=-1, remove_full_entry=False, y_col='V [kcal/mol]', entry_col = 'molec'):
    '''
    Returns dataframe truncated st values in y_col E(min_y, max_y)
    
    Parameters: 
        max_y (str or float/int): (default = 'largest')
            'largest': upper limit = largest value in y_col

        min_y (float/int): (default = -1)
            
        remove_full_entry (bool): (default = False)
            True: removes entire entry (ie molecule) if any values in y_col DNE(min_y, max_y)
            False: only removes rows where values in y_col DNE(min_y, max_y)

        y_col (str): (default = 'V [kcal/mol]')
            column to truncate

        entry_col (str): (default = 'molec')
            column containing names of the unique entries
    '''
    # finds largest y value
    if type(max_y) == str:
        max_y = np.max(df[y_col].values.tolist())
        #print(f'Max value from {y_col}: {max_y}')
        
    # truncating y values
    df_trunc = df[(df[y_col] > min_y) & (df[y_col] < max_y)]
    df_removed = df[(df[y_col] <= min_y) | (df[y_col] >= max_y)]
    entries_removed = df_removed[entry_col].values.tolist()

    # removes full entries
    if remove_full_entry == True:
        df_trunc = df_trunc[~df_trunc[entry_col].isin(entries_removed)] 

    # resetting indices
    df_trunc = df_trunc.reset_index(drop=True)
    df_removed = df_removed.reset_index(drop=True)

    return df_trunc, df_removed, entries_removed

def format_entries(df, entry_col='molec', comparison_col='V [kcal/mol]', sort_by='max', ascending=False, add_scores_col=True, display_final_order=False, sort_dict={'max': np.max, 'min': np.min, 'mean': np.mean}):
    '''
    Returns dataframe with each molecule sorted based off an assigned score

    Parameters:
        df (pd.DataFrame):
            dataframe of unique entires that contain 1+ datapoint(s)  
            
        entry_col (str): (default = 'molec')
            column with names of entries to score 

        comparison_col (str): (default = 'V [kcal/mol]')
            column with values to use for scoring

        sort_by (str): (default = 'max')
            method used to assign a score to an entry, with options given by sort_dict
            default options:
                'max': entry score =  max(vals in comparison_col)
                'min': entry score =  min(vals in comparison_col)
                'mean': entry score = mean(vals in comparison_col)

        ascending (bool): (default = False)
            False: Sorts entries by score in descending order 
            True: Sorts entries by score in ascending order

        add_scores_col (bool): (default = True)
            True: Creates new column 'Score (<sort_by> <comparison_vol>)' & adds adds in score for every row corrosponding to entry score
            False: No new column created, sorted dataframe returned

        display_final_order (bool): (default = False)
            True: Prints entries in new order 
            False: Nothing printed 

        sort_dict (dict): (default = {'max': np.max, 'min': np.min, 'mean': np.mean})
            names & functions availiable for sort_by parameter
                Keys: callable name for sort_by
                Values: functions to assign score using values in comparison_col
    '''
    # setting sorting funciton 
    if sort_by in sort_dict:
        score_function = sort_dict[sort_by]
        score_col_name = f'Score ({sort_by} {comparison_col})'
    else:
        print('Unrecognized sorting function\nReturning original dataframe')
        return df 

    # scoring each entry 
    entrys = np.unique(df[entry_col].values.tolist())
    dfs = []
    scores = []

    for entry in entrys:
        df_entry = df[(df[entry_col] ==entry)]
        values = df_entry[comparison_col].values.tolist()
        score = score_function(values)
        scores.append(score)

        if add_scores_col == True:
            scores_long = [score]*len(values)
            df_entry[score_col_name] = scores_long

        dfs.append(df_entry)


    # sorting each dataframe based on score 
    ordered_dfs = score_sort_list_dataframes(scores, dfs, ascending=ascending)
    df_new = pd.concat(ordered_dfs, axis=0, join='outer')
    df_new = df_new.reset_index(drop=True)
    entry_sorted = np.unique(df[entry_col].values.tolist())

    # diplaying order
    if display_final_order == True:
        matrix_print('Final entry order:', entry_sorted, num_rows=len(entry_sorted))

    return df_new

def score_sort_list_dataframes(scores, dfs, ascending=False):
    '''
    Returns list of dataframes sorted by parallel list of scores

    Parameters:
        scores (list): scores with each element corrosponding to same index element in dfs
            elements: str (alphabetical order) or float/in (numerical order)
            
        dfs (list): elements of type pd.DataFrame

        ascending (bool): (default = False)
            True: sorts dfs in ascending order based on scores 
            False: sorts dfs in descending order based on scores  
    '''
    # checking length
    if len(scores) != len(dfs):
        print('Need to pass same number of scores and dataframes\nReturning original DataFrame list')
        return dfs
    
    # finding sorted indicies
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=not ascending)
    sorted_scores = [scores[i] for i in sorted_indices]

    # sorting the dataframes 
    dfs_sorted = [dfs[i] for i in sorted_indices]

    return dfs_sorted
    

def create_Xy_matrix(X, y, num_molecs=106, N_per_molec=1000):
    '''
    Returns matrix of X & y data 
    '''
    # Setup
    X = np.reshape(X, (num_molecs, N_per_molec))
    y = np.reshape(y, (num_molecs, N_per_molec))
    return X, y

def extract_N_points(df_full, N = 'all', return_full_dataframe=True, sep_range=(0.001, 4), target_col='', target_vals=[], scaled=False, X_col='sep [A]', molec_col='molec', y_col='V [kcal/mol]'):
    '''
    Returns truncated dataframe where each molecule has N data points within sep_range 
    
    Parameters: 
        N (str or int): (default = 'all')
            'all' (str): all datapoints extracted
            type(int): N evenly spaced datapoints extracted for each molecule
        
        target_<col/vals>: (default = no targeting)
            If not default: returns entries where target_col entry is in target_vals
                target_col (str): Name of column to search, 
                target_vals (list): List of values to find in target_col
            
        return_full_dataframe (bool): (default = True)
            True: returns points in a data frame
            False: returns tuple of X & y matrices)
            
        scaled (bool): (default = False)
            True: scales X data with StandardScaler()
            False: does not scale X with StandardScaler()
            
        molec_col (str): (default = 'molec')
            name of column containing empirical formulas
            
        X_col (str): (default = 'sep [A]')
            name of column containing X data 
            
        y_col (str): (default = 'V [kcal/mol]')
            name of column containing y data 
    '''
    molecules = list(set(df_full[molec_col].values.tolist()))

    if N != 'all':
        x_axis = df_full[df_full[molec_col] == molecules[0]][X_col]
        x_vals = get_spaced_elements(x_axis, num_elems=N, low_lim=sep_range[0], up_lim=sep_range[1])[0]
        Ndf = df_full[df_full[X_col].isin(x_vals)]
    else:
        Ndf = df_full


    if target_col != '':
        df = Ndf[Ndf[target_col].isin(target_vals)]
    else:
        df = Ndf

    if return_full_dataframe == True:
        return df

    new_molecs = df[molec_col].values.tolist()
    match_with_duplicates = [value for value in molecules if value in new_molecs]
    num_molecs = len(match_with_duplicates)

    X_flat = np.ravel(df[X_col].values.tolist())
    y_flat = np.ravel(df[y_col].values.tolist())

    X = np.reshape(X_flat, (num_molecs, N))
    y = np.reshape(y_flat, (num_molecs, N))

    if scaled==True:
        scaler = StandardScaler()
        scaled_X = scaler.fit_transform(X)
        X = scaled_X
        
    return X, y


#### Loading from files 

def create_path(directory, base_name='trial', count_files=True, extension='/'):
    '''
    Returns path of file/folder created in directory
    
    Parameters:
        directory (str): 
            path of directory to create path, if directory DNE then it is created

        base (str): (default = 'trial')
            base name to give to new file/folder
            ex) '/path/given/by/directory/trial/'
        
        count_files (bool): (default = True)
            True: counts number of files in directory & adds number to base_name 
            ex) '/path/given/by/directory/trial_3/'
            
        extension (str)" (default = '/')
            extention to add to new path, if '/' then creates a folder 
    '''
    # removes '/' from end of directory path
    if directory.endswith('/'):
        directory = directory[:-1]

    # creates path if it DNE
    if not os.path.exists(directory):
        os.makedirs(directory)

    # new path
    new_path = directory+f'/{base_name}'

    # counts number of files in directory
    if count_files:
        file_count = len(next(os.walk(directory))[1]) + 1
        new_path = new_path + f'_{file_count}'

    # creating path
    os.makedirs(new_path)
    new_path = new_path + '/'

    return new_path


def deep_find_files(start_dir, name='', extension='.txt'):
    '''
    Returns list of all files in start_dir & its subdirectories named <name> with extention <extension>

    Parameters:
        start_dir (str): path of directory to search for files 

        name (str): name of target files

        extension (str): (default = '.txt')
            extension to target files 
    '''
    
    matched_files = []
    for root, _, files in os.walk(start_dir):
        for filename in fnmatch.filter(files, f"{name}*{extension}"):
            matched_files.append(os.path.join(root, filename))
    return matched_files


def load_dataframe(folder_path='', file_name='', file_path='', sep='\t', head_lines=None, custom_headers=None, custom_dtypes=None):
    ''' 
    Returns pd.DataFrame created from file given by a path

    Note: can pass file path either directly or indirectly using following params
        direct: <file_path>
        indirect: <folder_path> + <file_name>

    Parameters:
        head_lines (None or int): (default = None)
            number of empty/useless lines to remove at start of file, excluding line containing the column headers
            
        custom_headers (None or list of str): (default = None)
            None: automatically assigns column names using first row of DataFrame
            list of str: manually assigns column names 
            
        sep (str): (default = '/t')
            file delimiter

        custom_dtypes (None or list of types): (default = None)
            None: automatically assigns data type of columns
            list of tyeps: manually assigns data types of columns
    '''
    if file_path == '':
        data = pd.read_csv(folder_path+file_name, header=head_lines, sep=sep)
    else:
        data = pd.read_csv(file_path, header=head_lines, sep=sep)

    # naming columns 
    if custom_headers != None:
        data.columns = custom_headers
        headers = custom_headers
    elif custom_headers == None:
        headers = (data.iloc[0]).values.tolist()
        data.columns = headers
        data = data[1:]

    # if all dtypes specified 
    if type(custom_dtypes) == list:
        for i in range(len(headers)):
            data[headers[i]] = data[headers[i]].values.astype(custom_dtypes[i])
    # if only 1 dtype given (ie make all str)
    elif type(custom_dtypes) != list and custom_dtypes != None:
        data[headers] = data[headers].values.astype(custom_dtypes)
    # if nothing given then make all possible columns float64
    else:
        data = data.apply(pd.to_numeric, errors="ignore")

    return data 


def df_from_dir(search_dir, save_dir='Desktop', file_name = '*', file_ext='.txt', deep=True, merge = True, save_as_txt = False, display=False):
    '''
    Converts files in directory into Pandas dataframes

    Parameters:
        search_dir (string):    Path to directory to search
        file_name (string):     Name of files to find. Default to '*' for all files
        file_ext (string):      Extension of files to find. Default to '.txt'
        deep (boolean):         Search dir recursively if True or not if False. Default to True
        merge (boolean):        Merge all dfs into one df if True. Default to True
        save_as_txt (boolean):  Save dfs as .txt file in search_dir
    
    Returns:
        dfs (list or df)
        num_files
    '''
    # Paths to files
    if deep == True:
        files = deep_find_files(search_dir, file_name)
    else:
        files = [search_dir+f for f in glob.glob(f'{file_name}.txt')]


    if display == True: matrix_print(f'{len(files)} Files Found', [os.path.basename(file) for file in files])

    # Converting files into dfs
    dfs = [load_dataframe(file_path=file) for file in files]

    if len(dfs)==1 and merge==False: return dfs[0], 1

    # Merging or unpacking
    df = dfs if merge==False else merge_dfs(dfs)
    if save_as_txt == False: return df, len(files)

    # Saving
    if save_dir == '':
        save_dir = search_dir+f'saved_dfs/'
        num_files = len(os.listdir(save_dir)) + 1
        save_dir = save_dir+f'run{num_files}/'
        os.makedirs(save_dir)

        if merge==True: 
            save_dir = search_dir+f'merged_dfs/'
            num_files = len(os.listdir(save_dir)) + 1
            df.to_csv(save_dir+f'extracted_df{num_files}.txt', sep='\t', index=False, encoding='utf-8')
            return df, len(files)
        else:
            save_dir = search_dir+f'unmerged_dfs/'
            num_files = len(os.listdir(save_dir)) + 1
            save_dir = save_dir+f'run{num_files}/'
            os.makedirs(save_dir)
            for i in range(len(df)):
                df[i].to_csv(save_dir+f'extracted_df{i+1}.txt', sep='\t', index=False, encoding='utf-8')

    else:
        num_files = len(os.listdir(save_dir)) + 1
        if merge==True: 
            df.to_csv(save_dir+f'extracted_df{num_files}.txt', sep='\t', index=False, encoding='utf-8')
            return df, len(files)
        else:
            for i in range(len(df)):
                df[i].to_csv(save_dir+f'extracted_df{i+1}.txt', sep='\t', index=False, encoding='utf-8')

    return df, len(files)


def merge_dfs(dfs, group_rows = False, group_cols = ['molec', 'molecules']):
    '''
    Merge list dataframes into one dataframe, automatically combining columns with common names
    Note: currently assume all dataframes have same columns and rows 

    Parameters:
        dfs (pd.DataFrame list):    List of pandas dataframes. Columns within each df should be unique.
        group_rows (boolean):        Combines rows with duplicate values in comb_idx.
        group_cols (str list):        List of possible column names to combine along. 
    
    Returns:
        df (pd.DataFrame):          Merged dataframe
    '''
    # Return if 1 df and not combining rows
    if group_rows == False and len(dfs) == 1: return dfs[0]

    # Merging
    df = pd.concat(dfs, axis=0, join='outer')
    if group_rows == False: return df

    # Grouping --> NEED TO FIX
    group_cols = list(set(group_cols) & set(df.columns.values.tolist()))[0]

    return df


def create_Xy_df(fxns_df, X_list):
    '''
    Returns Dataframe with column[0] = X data
    Remaining columns = y(X) for specific molecule 
    '''
    molecules = fxns_df['molec'].tolist()
    functions = fxns_df['sym fxn'].tolist()

    headers = ['X'] + molecules
    lists = [X_list]

    r = sp.Symbol('r')
    for i in range(len(molecules)):
        f = sp.lambdify(r,functions[i],'numpy')
        lists.append(f(X_list))

    xy_df = pd.DataFrame(list(zip(*lists)), columns = headers) 
    return xy_df
