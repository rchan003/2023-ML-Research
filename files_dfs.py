from Universal import*
# k_force data: https://www.sciencedirect.com/science/article/abs/pii/S1386142515302808?via%3Dihub

### DATA CREATION FUNCTIONS ###
def truncate_y(df, max_y='largest', min_y=-1, remove_full_entry=False, y_col='V [kcal/mol]', entry_col = 'encoded name'):
    '''
    Returns dataframe and where y column is truncated where min_y < y < max_y
    if remove_full_entry == True then that molecules entire entry is deleted
    '''
    if type(max_y) == str:
        max_y = np.max(df[y_col].values.tolist())
        #print(f'Max value from {y_col}: {max_y}')
        
    #dropping values
    df_trunc = df[(df[y_col] > min_y) & (df[y_col] < max_y)]
    df_removed = df[(df[y_col] <= min_y) | (df[y_col] >= max_y)]

    entries_removed = df_removed[entry_col].values.tolist()
    
    if remove_full_entry == True:
        df_trunc = df_trunc[~df_trunc[entry_col].isin(entries_removed)] 

    df_trunc = df_trunc.reset_index(drop=True)
    df_removed = df_removed.reset_index(drop=True)

    return df_trunc, df_removed, entries_removed

def format_entries(df, entry_col='molec', comparison_col='V [kcal/mol]', sort_by='max', ascending=False, add_scores_col=True, display_final_order=False, sort_dict={'max': np.max, 'min': np.min, 'mean': np.mean}):
    '''
    sort_by: 'max', 'min', 'mean'
        (takes values from the comparison col specific the specific entry )
    order: 'descending', 'ascending'
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

    # final steps
    if display_final_order == True:
        matrix_print('Final entry order:', entry_sorted, num_rows=len(entry_sorted))

    return df_new

def score_sort_list_dataframes(scores, dfs, ascending=False):
    # Use sorted() with enumerate() to get sorted indices
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=not ascending)
    sorted_scores = [scores[i] for i in sorted_indices]

    # now sorting the dataframes 
    dfs_sorted = [dfs[i] for i in sorted_indices]

    return dfs_sorted

def bucket_score_morse(fxns_df):
    molecules = fxns_df['molec'].values.tolist()
    a0 = fxns_df['ao'].values.tolist()
    a1 = fxns_df['a1'].values.tolist()
    a2 = fxns_df['a2'].values.tolist()

    # Define bucket boundaries (increased the outer edges)
    a0_edges = [1e1, 1e4, 1e6, 1e8, 1e11]
    a1_edges = [-6e-6, -5e-4, -5e-3, -5e-2, -5]
    a2_edges = [0, -0.5, -1, -3, -7]

    # Determine bucket indices for each value
    a0_indices = np.digitize(a0, a0_edges, right=True)
    a1_indices = np.digitize(a1, a1_edges, right=True)
    a2_indices = np.digitize(a2, a2_edges, right=True)

    scores = []
    for molec, idx0, idx1, idx2 in zip(molecules, a0_indices, a1_indices, a2_indices):
        print(f"{molec} is in Bucket {idx0, idx1, idx2}")
        scores.append(idx0+idx1+idx2)

    return scores 

def create_Xy_matrix(X, y, num_molecs=106, N_per_molec=1000):
    # Setup
    X = np.reshape(X, (num_molecs, N_per_molec))
    y = np.reshape(y, (num_molecs, N_per_molec))
    return X, y

def extract_N_points(df_full, N = 'all', return_full_dataframe=True, sep_range=(0.001, 4), target_col='', target_vals=[], scaled=False, X_col='sep [A]', molec_col='molec', y_col='V [kcal/mol]'):
    '''
    returns dataframe where target col == target val, only for N values of sep [A] within sep_range
    basically slices dataframe twice (once to reduce sep and again to reduce just to target
    if target not updated then returns dataframe for 
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
    Creates desired path if it doesn't exist and returns string of path (format: directory + final_name + extension)
    Count_files
        True: final_name = base_name + f'_{number files in directory}'
        False: final_name = base_name
    '''
    if directory.endswith('/'):
        directory = directory[:-1]

    if not os.path.exists(directory):
        os.makedirs(directory)

    if os.path.isdir(directory):
        file_count = sum(1 for item in os.listdir(directory) if os.path.isfile(os.path.join(directory, item)))
        print(f"Number of files in the directory: {file_count}")

    file_count = len(next(os.walk(directory))[1]) + 1
    save_folder = directory+f'/{base_name}_{file_count}'
    os.makedirs(save_folder)
    save_folder = save_folder + '/'

    return save_folder


def deep_find_files(start_dir, name_prefix=''):
    matched_files = []
    for root, _, files in os.walk(start_dir):
        for filename in fnmatch.filter(files, f"{name_prefix}*.txt"):
            matched_files.append(os.path.join(root, filename))
    return matched_files


def load_dataframe(folder_path='', file_name='', file_path='', head_lines=None, headers=None, sep='\t', dtypes=None):
    ''' Returns pandas dataframe from file
    head_lines = number of lines for header (typically None)
    is_range = None: return specified values for conditions 
    is_range != None && conditions == [a,b]: return values within range [a,b] inclusive 
    headers == list of col names: assumes first row of dataframe are NOT column names
    headers == None: assumes first row of dataframe are col names 
    cols_desired & conditions MUST be lists
    '''
    if file_path == '':
        data = pd.read_csv(folder_path+file_name, header=head_lines, sep=sep)
    else:
        data = pd.read_csv(file_path, header=head_lines, sep=sep)

    # naming columns 
    if headers != None:
        data.columns = headers
    elif headers == None:
        headers = (data.iloc[0]).values.tolist()
        data.columns = headers
        data = data[1:]

    # if all dtypes specified 
    if type(dtypes) == list:
        for i in range(len(headers)):
            data[headers[i]] = data[headers[i]].values.astype(dtypes[i])
    # if only 1 dtype given (ie make all str)
    elif type(dtypes) != list and dtypes != None:
        data[headers] = data[headers].values.astype(dtypes)
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


def merge_dfs(dfs, group_rows = False, group_col = ['molec', 'molecules']):
    '''
    Merge list dataframes into one dataframe, automatically combining columns with common names
    Note: currently assume all dataframes have same columns and rows 

    Parameters:
        dfs (pd.DataFrame list):    List of pandas dataframes. Columns within each df should be unique.
        comb_rows (boolean):        Combines rows with duplicate values in comb_idx.
        comb_idx (str list):        List of possible column names to combine along. 
    
    Returns:
        df (pd.DataFrame):          Merged dataframe
    '''
    # Return if 1 df and not combining rows
    if group_rows == False and len(dfs) == 1: return dfs[0]

    # Merging
    df = pd.concat(dfs, axis=0, join='outer')
    if group_rows == False: return df

    # Grouping !!!!!NOT DONE
    group_col = list(set(group_col) & set(df.columns.values.tolist()))[0]

    return df


def create_Xy_df(fxns_df, X_list):
    '''
    Returns Dataframe with index & column[0] = X 
    Remaining columns = Y(x) for specific molecule 
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
