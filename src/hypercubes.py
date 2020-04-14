import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from functools import reduce


def define_cubes_in_df(df, ref_vars, target_var, bins=20,
                       cube_specs=None, cube_ids=None,
                       order_cubes=True, drop_stats=False,
                       reset_idx=False, assign_nearest_cube=False,
                       dat_types=None):
    """ Add cube_id, to dataset and return as well median target var and count of
    cube occurrences. Binning specs used in cube definition are also returned
    Parameters
    ----------
    df : dataframe. Main dataset containing reference variables
    ref_vars: list. List of reference variable(s) to use.
    target_var : str. Name of column with target.
    bins : int. Number of bins to cut reference variable in to. It can also
    be a list or array with bin intervals or a dictionary with the bins per
    reference variable. Default : 20.
    cube_specs : dict. Dictionary with specs for binning. Dictionary
    should have one entry per reference variable with keys 'values' (bin
    intervals) and 'labels' (bin labels). Default : None.
    cube_ids : dataframe. Dataframe with one column with bin labels per
    reference variable and a column specifying the ID of that hypercube.
    order_cubes : bool. Whether to sort cubes by median target_var.
    Default : True
    drop_stats : bool. Whether to drop cube stats. Default : False.
    assign_nearest_cube : bool. Whether to find nearest cube for entries in the
    dataset with no cube assigned. Needs cube_ids input. Default : False
    dat_types : list. List of data types for each variable. Default : None

    Returns
    ------
    dataframe : input dataset with extra column for cube_id
    cube_specs : dictionary with specs on cubes
    """

    if reset_idx:
        idx_cols = df.index.names
        df.reset_index(inplace=True)
    if cube_specs is None:
        # define hypercube binning
        _, cube_specs, bins = define_binning(df, ref_vars, bins)
    else:
        bins = None

    # loop through reference variables and apply binning
    cube_vars = []
    cube_specs = cube_specs or {}
    for f in ref_vars:
        binning = bin_variable(
            df[f],
            bins=bins[f] if bins is not None else None,
            bins_dict=cube_specs[f] if cube_specs is not None else None,
            ret_bins=True)
        cube_specs[f] = binning[1]
        df['cube_%s' % f] = binning[0].values
        cube_vars.append('cube_%s' % f)

    # get bin stats
    cubes = df.groupby(cube_vars)[target_var]. \
        agg(['median', 'count']). \
        reset_index()
    if cube_ids is None:
        # order cubes and label
        if order_cubes:
            cubes.sort_values('median', ascending=False, inplace=True)
        cubes['cube_id'] = np.arange(len(cubes))
    else:
        renamer = {v: 'cube_%s' % v for v in cube_ids.columns
                   if v != 'cube_id'}
        cubes = cubes.merge(cube_ids.rename(columns=renamer), on=cube_vars)

    # add to main dataset
    df = df.drop(['median', 'count', 'cube_id'], axis=1, errors='ignore'). \
        reset_index(). \
        merge(cubes, on=cube_vars, how='left'). \
        drop(cube_vars, axis=1). \
        set_index('index')
    if (cube_ids is not None) & (assign_nearest_cube is True):
        if df.cube_id.isnull().sum() > 0:
            df = get_nearest_cube(cube_ids, df, ref_vars, dat_types)
            df = df.drop(['median', 'count'], axis=1). \
                merge(cubes, on='cube_id', how='left')
    if drop_stats:
        df.drop(['median', 'count'], axis=1, inplace=True)

    if reset_idx:
        df.set_index(idx_cols, inplace=True)

    return df, cube_specs


def extract_residuals(df, cubes, cube_specs, cube_ids, ref_vars, target_vars, dat_types, col_index, res_prefix='res',
                      assign_nearest_cube=True):
    """
    Given the df and the hypercubes, it computes for each row of the df the residual (namely the distance
    from the expected output)
    This residuals are provided as new dataframe (df_res)
    This function is just a wrapper for define_cubes_in_df where some input variables are already set

    :param df: dataframe. Main dataset containing reference variables
    :param cubes: dataframe. Hypercubes statistical summary.
    :param cube_specs: dictionary with specs on cubes
    :param cube_ids: dataframe. Dataframe with one column with bin labels per
    :param ref_vars: list. List of reference variable(s) to use.
    :param target_vars: list. List of target variable(s) to use.
    :param dat_types: array. Type of ref_vars (e.g. ['numerical', 'angular',..])
    :param col_index: str. Name of the column that we want to promote as index
    :param res_prefix: str. Prefix to be used to indicate the column with residuals
    :param assign_nearest_cube: bool. Whether to find nearest cube for entries in the
    dataset with no cube assigned. Needs cube_ids input. Default : True
    :return df_res: dataframe. Main dataset + residual
    """
    df_res, _ = define_cubes_in_df(df.copy(), ref_vars, target_vars[0], drop_stats=True, reset_idx=True,
                                   cube_specs=cube_specs, cube_ids=cube_ids, assign_nearest_cube=assign_nearest_cube,
                                   dat_types=dat_types)

    df_res = df_res.reset_index().merge(cubes.drop(ref_vars, axis=1), on='cube_id', how='left').set_index(col_index)

    for t in target_vars:
        cube_target_var = 'cube_%s' % t
        df_res[res_prefix + '_%s' % t] = df_res[t] - df_res[cube_target_var]

        df_res = df_res.drop(cube_target_var, 1)

    return df_res


def extract_cubes(df, ref_vars, target_vars, cube_specs, lower_bnd, upper_bnd, cube_ids, nbins):
    """
    Given a dataframe df, extract the hypecubes based on ref_var (e.g. wind speed, wind direction) and target var
    (e.g. active power).
    This function is just a wrapper for hypercubes.hypercube_analysis where some input variables are already set


    :param df: dataframe. Main dataset containing target and reference variables
    :param ref_vars: list. List of reference variable(s) to use.
    :param target_vars:  str. Name of column with target.
    :param cube_specs: dict. Dictionary with specs for binning
    :param lower_bnd: number. lower bound of the percentile
    :param upper_bnd: number. upper bound of the percentile
    :param cube_ids: dataframe. Dataframe with one column with bin labels per reference variable
           and a column specifying the ID of that hypercube.
    :param nbins:  int. Number of bins to cut reference variable in to.
            It can also be a list or array with bin intervals or a dictionary with the bins per
    reference variable.

    Returns
    ------
    :return: cubes : dataframe. Hypercubes statistical summary.
    :return: cube_specs : dictionary with specs on cubes
    :return: cube_ids: dataframe. Dataframe with one column with bin labels per
    :return: cubes_reduced : dataframe. Same as cubes but reports only columns that refer to ref_var,
    cube_id, target median
    """

    def lower_bound(x):
        return np.nanpercentile(x, lower_bnd)

    def upper_bound(x):
        return np.nanpercentile(x, upper_bnd)

    cubes_reduced = []

    for t in target_vars:
        cubes, cube_specs = hypercube_analysis(df, ref_vars, t, cube_specs=cube_specs, cube_ids=cube_ids, bins=nbins,
                                               lower_bound=lower_bound, upper_bound=upper_bound)
        if cube_ids is None:
            cube_ids = cubes[np.append(ref_vars, 'cube_id')].drop_duplicates()

        cubes_reduced.append(
            cubes.rename(columns={'median': 'cube_%s' % t})[np.append(np.append(ref_vars, 'cube_id'), 'cube_%s' % t)])

    cubes_reduced = reduce(lambda x, y: x.merge(y, on=list(np.append(ref_vars, 'cube_id'))), cubes_reduced)

    return cubes_reduced, cube_specs, cube_ids, cubes


def define_binning(df, ref_vars, bins):
    """ Apply binning across a set of variables
    Parameters
    ----------
    df : dataframe. Main dataset containing reference variables
    ref_vars: list. List of reference variable(s) to use.
    bins : int. Number of bins to cut reference variable in to. It can also
    be a list or array with bin intervals or a dictionary with the bins per
    reference variable. Default : 20.

    Returns
    ------
    binning : list. list with objects to be passed to pd.DataFrame.groupby. One
    object per reference variable
    cube_specs : dictionary with specs on cubes
    """

    if type(bins) != dict:
        bins = {f: bins for f in ref_vars}
    bins = {f: bin_variable(df[f], bins=bins[f], ret_bins=True)
            for f in ref_vars}
    binning = [bins[b][0] for b in bins.keys()]
    cube_specs = {b: bins[b][1] for b in bins.keys()}

    return binning, cube_specs, bins

def bin_variable(x, bins=20, bins_dict=None, ret_bins=False):
    """ Bin pandas series into equally spaced n_bins
    Parameters
    ----------
    x : pandas series.
    bins : int. Number of bins to segment x into. It can also be a list or an
    array of bin intervals. Default : 20
    bins_dict : dict. Dictionary with bin values and labels. Default : None.
    ret_bins : bool. Return bin values and labels? Default : False.

    Returns
    ------
    new binned pandas series
    bins: dict. Bin values and labels
    """
    if bins_dict is None:
        if (isinstance(bins, list)) | (isinstance(bins, np.ndarray)):
            x_bin = bins
        else:
            # define bins
            x_bin = np.linspace(x.min(), x.max(), bins)

        # define labels
        x_bin_labels = [np.round(np.mean([x0, x_bin[k + 1]]), 2)
                        for k, x0 in enumerate(x_bin)
                        if k < (len(x_bin) - 1)]
    else:
        x_bin = bins_dict['values']
        x_bin_labels = bins_dict['labels']

    # bin series
    x_cut = pd.cut(x, x_bin, labels=x_bin_labels, include_lowest=True)
    x_cut = x_cut.astype(np.float64)

    if ret_bins:
        return x_cut, {'values': x_bin, 'labels': x_bin_labels}
    else:
        return x_cut
    
    
def hypercube_analysis(df, ref_vars, target,
                       cube_specs=None,
                       cube_ids=None,
                       bins=20,
                       lower_bound=None,
                       upper_bound=None):
    """ perform hypercube analysis by binning dataset into its reference variables
    and deriving standard statistics on each bin.
    Parameters
    ----------
    df : dataframe. Main dataset containing target and reference variables
    ref_vars: list. List of reference variable(s) to use.
    target : str. Name of column with target.
    cube_specs : dict. Dictionary with specs for binning. Dictionary
    should have one entry per reference variable with keys 'values' (bin
    intervals) and 'labels' (bin labels). Default : None.
    cube_ids : dataframe. Dataframe with one column with bin labels per
    reference variable and a column specifying the ID of that hypercube.
    bins : int. Number of bins to cut reference variable in to. It can also
    be a list or array with bin intervals or a dictionary with the bins per
    reference variable. Default : 20.
    lower_bound, upper_bound : method. function to determin the lower and upper
    bound of a cube. Default is None (will use default boxplot_outliers bounds)

    Returns
    ------
    cubes : dataframe. Hypercubes statistical summary.
    cube_specs : dictionary with specs on cubes
    """

    # get binning
    if cube_specs is None:
        binning, cube_specs, _ = define_binning(df, ref_vars, bins)
    else:
        binning = [bin_variable(df[b], bins_dict=cube_specs[b])
                   for b in cube_specs.keys()]
        bin_keys = list(cube_specs.keys())

    # apply binning to fleet summary frame
    if lower_bound is None:
        def lower_bound(x):
            return boxplot_outliers(x)[1][0]
    if upper_bound is None:
        def upper_bound(x):
            return boxplot_outliers(x)[1][1]
    # get stats per cube
    cubes = df.groupby(binning)[target]. \
        agg(['median', 'mean', 'sum', 'count', 'min', 'max',
             lower_bound, upper_bound]). \
        reset_index(). \
        sort_values('mean', ascending=False)

    # label cubes
    if cube_ids is not None:
        cubes = cubes.merge(cube_ids, on=list(ref_vars))
    else:
        cubes['cube_id'] = np.arange(len(cubes))

    return cubes, cube_specs
