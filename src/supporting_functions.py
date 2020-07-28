import operator
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt

from sklearn import metrics
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances


def convert_to_u_v(ang_var, speed=1):
    """
    Convert a angular var and scale variable to u and v variables, which take the circular nature of the variable into
    account.

    Parameters
    ----------
    ang_var: numpy array or pandas series containing the angular variable
    scale_var: numpy array or pandas series of the same length containing the speed (non-angular) variable


    Returns
    -------
    u and v arrays (or Series)
    """

    # Transform from degrees to radians:
    factor = np.pi / 180


    u = np.sin(ang_var * factor) * scale_var
    v = np.cos(ang_var * factor) * scale_var

    return(u, v)


def plot_kmeans_validation(df, k_max=9, sample_size=1 ,CONNECTIVITY=True, SILHOUETTE=True, RETURN=False):
    """
    This function plots multiple cluster validation metrics, based on kmeans clustering.

    :df:            Dataframe   This is the data to cluster.
    :k_max:         Int         Integer with the max amount of clusters to test
    :sample_size:   Float       Is a value between 0 and 1, the size of the sample to do the validation on for the
                                computative expensive measures.
    :CONNECTIVITY   Boolean     Whether or not to calculate the (computationally expensive) connectivity
    :SILHOUETTE     Boolean     Whether or not to calculate the (computationally expensive) silhouette
    :RETURN=False   Boolean     Whether or not to return the values

    """
    ss = []  # sum of squared distances of samples to their closest cluster center (used for elbow method)
    silhouette = []
    connectivity = []
    calinski_harabasz = []
    davies_bouldin = []

    # Use only part of the data for the validation, since some metrics take quite some time to calculate
    if sample_size!=1:
        random_indices = np.random.choice(np.arange(len(df)), int(len(df) * sample_size), replace=False)
        df_sampled = df.iloc[random_indices]
    else:
        df_sampled = df

    for j in range(1, k_max):
        # Do the clustering
        kmeans = KMeans(n_clusters=j, init='k-means++', max_iter=500, n_init=1, random_state=0)
        clustering = kmeans.fit_predict(df)
        if sample_size != 1:
            clustering_sampled = clustering[random_indices]
        else:
            clustering_sampled = clustering

        # Calculate the validation measures
        ss.append(kmeans.inertia_)
        if CONNECTIVITY:
            connectivity.append(connectivity_metric(     df_sampled, clustering_sampled, neighbourhood=3))

        if (j != 1):
            if SILHOUETTE:
                silhouette.append(metrics.silhouette_score(          df_sampled, clustering_sampled))
            calinski_harabasz.append(metrics.calinski_harabasz_score(df,         clustering))
            davies_bouldin.append(metrics.davies_bouldin_score(      df,         clustering))

    f, axes = plt.subplots(nrows=1, ncols=3+CONNECTIVITY+SILHOUETTE,
                           figsize=(4*(3+CONNECTIVITY+SILHOUETTE), 5))

    # Plot elbow method
    ax = sns.lineplot(range(1, k_max), ss, markers=True, ax=axes[0])
    ax.set(xlabel='Number of clusters', ylabel='SS')
    ax.set_title('Elbow method');

    # Plot connectivity method
    if CONNECTIVITY:
        ax = sns.lineplot(range(1, k_max), connectivity, markers=True, ax=axes[1])
        ax.set(xlabel='Number of clusters', ylabel='Connectivity')
        ax.set_title('Connectivity method');

    # Plot the silhouette measures
    if SILHOUETTE:
        ax = sns.barplot(x=list(range(2, k_max)), y=silhouette, ax=axes[1+CONNECTIVITY], palette="Blues_d")
        ax.set(xlabel='Number of clusters', ylabel='Silhouette score')
        ax.set_title('Silhouette analysis');

    # Plot the calinski_harabasz measures
    ax = sns.barplot(x=list(range(2, k_max)), y=calinski_harabasz, ax=axes[1+CONNECTIVITY+SILHOUETTE], palette="Blues_d")
    ax.set(xlabel='Number of clusters', ylabel='Calinski harabasz score')
    ax.set_title('Calinski harabasz analysis');

    # Plot the calinski_harabasz measures
    ax = sns.barplot(x=list(range(2, k_max)), y=davies_bouldin, ax=axes[2+CONNECTIVITY+SILHOUETTE], palette="Blues_d")
    ax.set(xlabel='Number of clusters', ylabel='Davies Bouldin score')
    ax.set_title('Davies Bouldin analysis');

    plt.tight_layout()
    plt.show()

    if RETURN:
        return(ss, silhouette, connectivity, calinski_harabasz, davies_bouldin)
    

def do_individual_clustering(turbines, dict_turbines_norm, n_clusters, operational_features, dict_turbines, colorpal):
    rows = len(turbines)
    f, axes = plt.subplots(nrows = rows, ncols = 3, figsize=(20, 5*rows))

    color_counter = 0

    for i, turbine in enumerate(turbines):
        df_One_norm = dict_turbines_norm[turbine].copy()

        # Do clustering:
        kmeans = KMeans(n_clusters[i], init='k-means++', max_iter=1000, n_init=1, random_state=0)
        labels = kmeans.fit_predict(df_One_norm.loc[:, operational_features]) # Only use the operational features in the MAP step

        # Add labels to original and normalized dataset:
        dict_turbines[turbine].loc[:,'Label']  = labels
        dict_turbines_norm[turbine].loc[:,'Label'] = labels


        # Visualize
        cpal = colorpal[color_counter:color_counter+n_clusters[i]]
        color_counter += n_clusters[i]

        # 1. Plot power curve
        df_sample = dict_turbines[turbine].sample(int(len(labels)/5)) #sample of 20%
        ax = sns.scatterplot(x='Ws_avg', y='P_avg', data=df_sample, hue='Label', s=2, legend='full',linewidth=0, alpha = 0.2, palette=cpal, ax=axes[i][0])
        ax.set(xlabel='Wind speed [m/s]', ylabel='Active Power [MW]')
        ax.set_title('Power curve (sample of 20%)');

        # 2. Plot torque curve
        ax = sns.scatterplot(x='Ds_avg', y='Rm_avg', data=df_sample, hue='Label', s=2, legend='full',linewidth=0, alpha = 0.2, palette=cpal, ax=axes[i][1])
        ax.set(xlabel='Generator_speed [rpm]', ylabel='Torque [Nm]')
        ax.set_title('Torque curve (sample of 20%)');

        # 3. Plot size of clusters
        dict_counter = Counter(labels)
        dict_counter = sorted(dict_counter.items(), key=operator.itemgetter(0)) # sorts the dict by key
        keys =   [ i[0] for i in dict_counter ]
        values = [ i[1] for i in dict_counter ]
        ax = sns.barplot(x=keys, y=list(values/np.sum(values)), palette=cpal, ax=axes[i][2])
        ax.set(xlabel='Label', ylabel='Relative count')
        ax.set_title('Amount of points per Label')

    plt.tight_layout()
    
    return (dict_turbines_norm)

def visualize_individual_clustering_per_clusters(colorpal, n_clusters, dict_turbines, turbines):
    color_counter = 0

    for i, turbine in enumerate(turbines):
    
        cpal = colorpal[color_counter:color_counter+n_clusters[i]]
        


        df_sample = dict_turbines[turbine]

        # 1. Plot power curve
        g1 = sns.FacetGrid(df_sample, col="Label", height=2.5, aspect = 1.25, hue='Label', palette=cpal)
        g1.map(plt.scatter, "Ws_avg", "P_avg", s=1,linewidth=0, alpha = 1)
        g1.set(xlim=(0, 20), ylim=(-0.100, 2.100))
        g1.set_axis_labels("Wind speed [m/s]","Active power [MW]")
        for j in range(0, n_clusters[i]):
            g1.axes[0,j].set_title("Operating mode " + str(color_counter+j+1) + " (Turbine " + str(i+1) + ")")
        g1.fig.suptitle('Power curve', y=1.02); 

        # 2. Plot torque curve
        g2 = sns.FacetGrid(df_sample, col="Label", height=2.5, aspect = 1.25, hue='Label', palette=cpal)
        g2.map(plt.scatter, "Ds_avg", "Rm_avg", s=1,linewidth=0, alpha = 1)
        g2.set_axis_labels("Torque [Nm]", "Generator speed [rpm]")
        for j in range(0, n_clusters[i]):
            g2.axes[0,j].set_title("Operating mode " + str(color_counter+j+1) + " (Turbine " + str(i+1) + ")")
        g2.fig.suptitle('Torque curve', y=1.02);

        # 3. Plot size of clusters
        dict_counter = Counter(dict_turbines[turbine]['Label'])
        dict_counter = sorted(dict_counter.items(), key=operator.itemgetter(0)) # sorts the dict by key
        keys =   [ i[0] for i in dict_counter ]
        values = [ i[1] for i in dict_counter ]
        g3 = sns.FacetGrid(df_sample, height=2.5, aspect = 1.25*n_clusters[i])
        ax = sns.barplot(x=keys, y=list(values/np.sum(values)), palette=cpal)
        ax.set(xlabel='Label', ylabel='Relative count')
        ax.set_title('Amount of points per Label')

        color_counter += n_clusters[i]

def create_table_endogenous_view(turbines, dict_turbines, operational_features):
    table = pd.DataFrame()
    counter = 1

    for turbine_i, turbine in enumerate(turbines):
        df_Turbine = dict_turbines[turbine]

        for cluster in range(0, len(df_Turbine['Label'].unique())):
            df_Cluster = df_Turbine[df_Turbine['Label']==cluster]
            min_max_values = {' ':'Turbine '+str(turbine_i+1), '  ':'OM'+str(counter)}

            for parameter in df_Cluster.columns: 
                if parameter in operational_features:
                    min_max_values[parameter+'_A'] = str(int(np.round(np.min(df_Cluster[parameter]))))
                    min_max_values[parameter+'_B'] = str(int(np.round(np.max(df_Cluster[parameter]))))

            table = table.append(min_max_values, ignore_index=True, sort=False)
            counter += 1
    table.set_index([' ', '  '], inplace=True)
    table = table.T

    # Add columns for index
    parameters_col = []
    min_max_col = []
    for parameter in dict_turbines[turbines[0]].columns:
        if parameter in operational_features:
            parameters_col.append(parameter)
            parameters_col.append(parameter)
            min_max_col.append('min')
            min_max_col.append('max')
    table['  '] = parameters_col
    table[' '] = min_max_col
    table.set_index(['  ', ' '], inplace=True)
    
    return(table)

def remove_noise(dict_turbines_uncleaned, hypercubes, remove_negative=True, min_cube_count=5, max_pos_deviation=400, max_neg_deviation=-400):
    """
    There are 4 methods we use to filter points:
        - We remove points which have a negative active power or negative wind speed
        - We remove points which have an uncommon wind speed-active power relation (by sparse cubes)
        - We remove points which deviate to much from the expected active power on the pos. side (hypercube approach)
        - We remove points which deviate to much from the expected active power on the neg. side (hypercube approach)

    :param dict_turbines_uncleaned:
    :param hypercubes:
    :param remove_negative:
    :param min_cube_count:
    :param max_pos_deviation:
    :param max_neg_deviation:
    :return:
    """
    dict_turbines = {}

    for turbine_name, turbine_df in dict_turbines_uncleaned.items():

        df = turbine_df
        # 1) Remove negative active power values and negative windspeed values:
        if remove_negative:
            df = df[(df['P_avg'] > 0) & (df['Ws_avg'] > 0)]

        # Construct hypercubes and calculate residuals
        df.set_index('Date_time', drop=False, inplace=True)
        df.rename(columns={'Date_time': 'time'}, inplace=True)
        ref_vars = ['Ws_avg', 'Wa_avg']
        dat_types = ['numerical', 'angular']
        target_var = ['P_avg']
        _, cube_specs, _ = hypercubes.define_binning(df, ref_vars, {'Ws_avg': 150, 'Wa_avg': 40})
        cubes, _, cube_ids, cubes0 = hypercubes.extract_cubes(df, ref_vars, target_var, cube_specs, 40, 60,
                                                              cube_ids=None, nbins=None)

        # 2) Filter cubes with too less points
        onePointCube = cubes0[cubes0['count'] <= min_cube_count].cube_id.values  # we might lose some points which are by accident under balanced
        cubes = cubes[~cubes.cube_id.isin(onePointCube)]
        cube_ids = cube_ids[~cube_ids.cube_id.isin(onePointCube)]

        # 3) Remove values which deviate too much from the median of its hypercube:
        df = hypercubes.extract_residuals(df.copy(), cubes, cube_specs, cube_ids,
                                          # For each time stamp we enrich the df with: residual information (w.r.t. reference cube)
                                          ref_vars, target_var, dat_types, col_index=['Date_time'],
                                          assign_nearest_cube=False)
        df = df[df['res_P_avg'] < max_pos_deviation]
        df = df[df['res_P_avg'] > max_neg_deviation]

        dict_turbines[turbine_name] = df

    uncleaned_df = dict_turbines_uncleaned['R80711'].reset_index()
    removed_points = uncleaned_df[~uncleaned_df['Date_time'].isin(dict_turbines['R80711']['time'])]

    fig = plt.figure(figsize=(10, 5))
    sns.set(font_scale=1.8)
    ax = sns.scatterplot(x='Ws_avg', y='P_avg', data=dict_turbines['R80711'], alpha=1, s=1, linewidth=0)
    sns.scatterplot(x='Ws_avg', y='P_avg', data=removed_points, alpha=1, s=5, linewidth=0)
    fig.legend(labels=['Clean data', 'Noise'], loc='upper right', bbox_to_anchor=(0.8, 0.9))
    ax.set(xlabel='Wind speed [m/s]', ylabel='Active power [MW]')

    return (dict_turbines)


def plot_kmeans_validation(df, k_max=9, sample_size=1 ,CONNECTIVITY=True, SILHOUETTE=True, RETURN=False):
    """
    This function plots multiple cluster validation metrics, based on kmeans clustering.

    :df:            Dataframe   This is the data to cluster.
    :k_max:         Int         Integer with the max amount of clusters to test
    :sample_size:   Float       Is a value between 0 and 1, the size of the sample to do the validation on for the
                                computative expensive measures.
    :CONNECTIVITY   Boolean     Whether or not to calculate the (computationally expensive) connectivity
    :SILHOUETTE     Boolean     Whether or not to calculate the (computationally expensive) silhouette
    :RETURN=False   Boolean     Whether or not to return the values

    """
    ss = []  # sum of squared distances of samples to their closest cluster center (used for elbow method)
    silhouette = []
    connectivity = []
    calinski_harabasz = []
    davies_bouldin = []

    # Use only part of the data for the validation, since some metrics take quite some time to calculate
    if sample_size!=1:
        random_indices = np.random.choice(np.arange(len(df)), int(len(df) * sample_size), replace=False)
        df_sampled = df.iloc[random_indices]
    else:
        df_sampled = df

    for j in range(1, k_max):
        # Do the clustering
        kmeans = KMeans(n_clusters=j, init='k-means++', max_iter=1000, n_init=1, random_state=0)
        clustering = kmeans.fit_predict(df)
        if sample_size != 1:
            clustering_sampled = clustering[random_indices]
        else:
            clustering_sampled = clustering

        # Calculate the validation measures
        ss.append(kmeans.inertia_)
        if CONNECTIVITY:
            connectivity.append(connectivity_metric(     df_sampled, clustering_sampled, neighbourhood=3))

        if (j != 1):
            if SILHOUETTE:
                silhouette.append(metrics.silhouette_score(          df_sampled, clustering_sampled))
            calinski_harabasz.append(metrics.calinski_harabasz_score(df,         clustering))
            davies_bouldin.append(metrics.davies_bouldin_score(      df,         clustering))

    f, axes = plt.subplots(nrows=1, ncols=3+CONNECTIVITY+SILHOUETTE,
                           figsize=(4*(3+CONNECTIVITY+SILHOUETTE), 5))

    # Plot elbow method
    ax = sns.lineplot(range(1, k_max), ss, markers=True, ax=axes[0])
    ax.set(xlabel='Number of clusters', ylabel='SS')
    ax.set_title('Elbow method');

    # Plot connectivity method
    if CONNECTIVITY:
        ax = sns.lineplot(range(1, k_max), connectivity, markers=True, ax=axes[1])
        ax.set(xlabel='Number of clusters', ylabel='Connectivity')
        ax.set_title('Connectivity method');

    # Plot the silhouette measures
    if SILHOUETTE:
        ax = sns.barplot(x=list(range(2, k_max)), y=silhouette, ax=axes[1+CONNECTIVITY], palette="Blues_d")
        ax.set(xlabel='Number of clusters', ylabel='Silhouette score')
        ax.set_title('Silhouette analysis');

    # Plot the calinski_harabasz measures
    ax = sns.barplot(x=list(range(2, k_max)), y=calinski_harabasz, ax=axes[1+CONNECTIVITY+SILHOUETTE], palette="Blues_d")
    ax.set(xlabel='Number of clusters', ylabel='Calinski harabasz score')
    ax.set_title('Calinski harabasz analysis');

    # Plot the calinski_harabasz measures
    ax = sns.barplot(x=list(range(2, k_max)), y=davies_bouldin, ax=axes[2+CONNECTIVITY+SILHOUETTE], palette="Blues_d")
    ax.set(xlabel='Number of clusters', ylabel='Davies Bouldin score')
    ax.set_title('Davies Bouldin analysis');

    plt.tight_layout()
    plt.show()

    if RETURN:
        return(ss, silhouette, connectivity, calinski_harabasz, davies_bouldin)
    

def connectivity_metric(X, labels, neighbourhood=10, metric='euclidean'):
    """
    Connectivity captures the degree to which genes are connected within a cluster by keeping track of whether
    neighboring points are put into the same cluster. It looks for each point to the points in the neighbourhood, and
    adds a penalty for each point which is not in its neighbourhood.

    source: https://cran.microsoft.com/web/packages/clValid/vignettes/clValid.pdf

    :param X:           Dataframe   Each row contains a new datapoint, the columns are the different features.
    :param labels:      Array       Containing for each row, the assigned classification label.
    :param neighbourhood:Integer    Defines the amount of neighbours to use in this method.
    :param metric:      String      The metric to use when calculating distance between instances in a feature array.
                                    If metric is a string, it must be one of the options allowed by
                                    scipy.spatial.distance.pdist for its metric parameter, or a metric listed in
                                    pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
    :return:            Integer     The connectivity score.
    """

    # Make sure that the index start from 0 (in case of sampling e.g.):
    X = X.reset_index(drop=True)

    # Calculate distance matrix:
    distance_matrix = pairwise_distances(X, metric=metric)

    score = 0
    penalty = 1/neighbourhood

    for index, observation in X.iterrows():
        sorted_indexes = np.argsort(distance_matrix[index])

        current_label = labels[index]
        for n in range(1, neighbourhood+1):
            neighbour_label = labels[sorted_indexes[n]]
            if neighbour_label != current_label:
                score += penalty

    return(score)


def make_cube_ids(cube_specs, ref_vars):
    cube_ids = get_all_combinations(cube_specs, ref_vars)
    cube_ids['cube_id']=cube_ids.index 
    return(cube_ids)


def get_all_combinations(cube_specs, ref_vars, df=pd.DataFrame(), ref_counter=0, current_dict={}):
    
    if len(ref_vars) > ref_counter:
        for label_value in cube_specs[ref_vars[ref_counter]]['labels']:
            current_dict[ref_vars[ref_counter]]=label_value
            
            df = get_all_combinations(cube_specs, ref_vars, df, ref_counter+1, current_dict)
    else: 
        df = df.append(current_dict, ignore_index=True)
    return(df)
    
    
def visualize_sparse_hypercubes(MIN_POINTS_IN_CUBE, turbines, dict_turbines, n_clusters):
    df_cube_counts = pd.DataFrame()
    counter=1
    for turbine_i, turbine in enumerate(turbines):
        df_One = dict_turbines[turbine]

        for cluster_i in range(0, n_clusters[turbine_i]):
            df_cluster_i = df_One[df_One['Label']==cluster_i]
            # print('\nturbine '+ turbine + ', cluster ' + str(cluster_i) + ':')
            df_grouped_cluster_i = df_cluster_i.groupby('cube_id')['time'].count()

            n_cubes = len(df_grouped_cluster_i)
            used_cubes = df_grouped_cluster_i[df_grouped_cluster_i>=MIN_POINTS_IN_CUBE]
            n_used_cubes = len(used_cubes)

            n_points = len(df_cluster_i)
            n_used_points = sum(df_grouped_cluster_i[df_grouped_cluster_i>=MIN_POINTS_IN_CUBE])

            label = turbine+"_"+str(cluster_i)
            label_nice = 'turb.'+str(turbine_i+1)+'_clus.'+str(cluster_i+1)
            label_operating_mode = 'OM '+str(counter)
            counter += 1


            df_cube_counts = df_cube_counts.append({'n_used_cubes':n_used_cubes, 'n_nonused_cubes':n_cubes-n_used_cubes,
                                                   'n_used_points':n_used_points, 'n_nonused_points':n_points-n_used_points,
                                                    'cluster': label, 'label_nice':label_nice, 'label_operating_mode':label_operating_mode}, ignore_index=True)

    cluster_labels = df_cube_counts['cluster']
    cluster_labels_nice = df_cube_counts['label_nice']

    # 1) Plot amount of hypercubes per cluster
    g1 = df_cube_counts.set_index('label_operating_mode')[['n_used_cubes','n_nonused_cubes']].plot(kind='bar', width=0.9, stacked=True, color=['darkblue','lightgrey'], figsize=(5,1.5), linewidth=0)
    g1.set(xlabel='', ylabel='Amount of hypercubes')
    for index, row in df_cube_counts.iterrows():
        n_cubes = row.n_used_cubes+row.n_nonused_cubes
        g1.text(row.name, row.n_used_cubes+5 , str(round(row.n_used_cubes/n_cubes*100,1))+'%', color='darkblue', ha="center", fontsize=6)
    plt.legend(['n used hypercubes','n removed hypercubes'], loc='upper center', bbox_to_anchor=(0.5, 1.3),
              fancybox=True, shadow=True, ncol=2)
    plt.xticks(rotation=75)

    # 2) Plot amount of points per cluster
    g2 = df_cube_counts.set_index('label_operating_mode')[['n_used_points','n_nonused_points']].plot.bar(width=0.9, stacked=True, color=['darkblue','lightgrey'], figsize=(5,1.5), linewidth=0)
    g2.set(xlabel='', ylabel='Amount of points')
    for index, row in df_cube_counts.iterrows():
        n_points = row.n_used_points+row.n_nonused_points
        g2.text(row.name, n_points+5000 , str(round(row.n_used_points/n_points*100,1))+'%', color='darkblue', ha="center", fontsize=6)
    plt.legend(['n used points','n removed points'], loc='upper center', bbox_to_anchor=(0.5, 1.32),
              fancybox=True, shadow=True, ncol=2)
    plt.xticks(rotation=75);
    
    return(df_cube_counts, cluster_labels_nice)


def calculate_KDE_per_hypercube(turbines, n_clusters, dict_turbines, MIN_POINTS_IN_CUBE):
    KDEs = []
    points_per_cube = []

    for i, turbine in enumerate(turbines):
        df_One = dict_turbines[turbine].copy()
        turbine_KDEs = []

        for cluster in range(0, n_clusters[i]):
            df_label = df_One[df_One['Label']==cluster]
            cluster_KDEs = []
            cluster_points_per_cube = []

            for cube_id in df_label['cube_id'].unique():
                df_hypercube = df_label[df_label['cube_id']==cube_id]
                n_points_in_cube = len(df_hypercube)

                if n_points_in_cube > MIN_POINTS_IN_CUBE: 
                    kde = st.gaussian_kde(df_hypercube['P_avg'], bw_method='silverman')

                    cluster_KDEs.append(kde)
                    cluster_points_per_cube.append(n_points_in_cube)
            turbine_KDEs.append(cluster_KDEs)
            points_per_cube.append(cluster_points_per_cube)
        KDEs.append(turbine_KDEs)
        
    return(KDEs, points_per_cube)


def construct_engdogenous_profiles(turbines, n_clusters, points_per_cube, KDEs, colorpal):
    n_rows = int(np.ceil(np.sum(n_clusters)))
    fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(15*2,5*n_rows))

    group_counter = 0
    x = np.linspace(0, 2.500, 1000)
    sum_densities = [0]*sum(n_clusters)

    for turbine_i, turbine in enumerate(turbines):

        for cluster_i in range(0, n_clusters[turbine_i]):
            KDE_sum = [0]*len(x)
            n_points_in_cluster = sum(points_per_cube[group_counter])

            for cube_i, kde in enumerate(KDEs[turbine_i][cluster_i]):
                y = kde(x)

                # Plot all KDEs separately
                row = int(np.floor(group_counter/2))
                col = group_counter%2
                axes[group_counter, 0].plot(x, y)
                axes[group_counter, 0].set_title("Individual probability distributions of OM "+str(group_counter+1)+" (turbine "+str(turbine_i+1)+' )', size='xx-large',  backgroundcolor=colorpal[group_counter])
                axes[group_counter, 0].set_ylim(0, 35)
                axes[group_counter, 0].set_ylabel('Density')
                axes[group_counter, 0].set_xlabel('Active Power [MW]')

                # Calculate the normalized sum per cluster/operative mode
                y_norm = y * points_per_cube[group_counter][cube_i] / n_points_in_cluster
                KDE_sum  = [a + b for a, b in zip(KDE_sum, y_norm)]

            sum_densities[group_counter] = KDE_sum

            # Plot the weighted sum
            axes[group_counter, 1].plot(x, KDE_sum)
            axes[group_counter, 1].set_title("Mixture probability distribution of OM "+str(group_counter+1)+" (turbine "+str(turbine_i+1)+' )', size='xx-large', backgroundcolor=colorpal[group_counter])
            axes[group_counter, 1].set_ylim(0, 9)
            axes[group_counter, 1].set_ylabel('Density')
            axes[group_counter, 1].set_xlabel('Active Power [MW]')

            group_counter+=1

    sum_densities = pd.DataFrame.from_records(sum_densities)
    plt.tight_layout();
    
    return(sum_densities)

def plot_power_and_torque(points, colorpal):
    # 1. Plot power curve
    g1 = sns.FacetGrid(points, col="FW_Operating_mode", height=3, aspect = 1.5, hue='Operating_mode', palette=colorpal)
    g1.map(plt.scatter, "Ws_avg", "P_avg", s=2,linewidth=0, alpha = 0.01)
    g1.set(xlim=(0, 20), ylim=(-0.100, 2.100))
    g1.set_axis_labels("Wind speed [m/s]", "Active power [MW]")
    g1.set_titles("Operating mode {col_name}")
    g1.fig.suptitle('Power curve', y=1.02)
    g1.add_legend()
    for lh in g1._legend.legendHandles: # This makes the legend visible
        lh.set_alpha(1)
        lh._sizes = [50] 
    plt.tight_layout()

    # 2. Plot torque curve
    g2 = sns.FacetGrid(points, col="FW_Operating_mode", height=3, aspect = 1.5, hue='Operating_mode', palette=colorpal)
    g2.map(plt.scatter, "Ds_avg", "Rm_avg", s=2,linewidth=0, alpha = 0.03)
    g2.set_axis_labels("Torque [Nm]", "Generator speed [rpm]")
    g2.set_titles("Operating mode {col_name}")
    g2.fig.suptitle('Torque curve', y=1.02)
    g2.add_legend();
    for lh in g2._legend.legendHandles: # This makes the legend visible
        lh.set_alpha(1)
        lh._sizes = [50] 
        
def view_endogenous_profiles(turbines, n_clusters, points_per_cube, KDEs, colorpal, sum_densities):
    group_counter = 0
    x = np.linspace(0, 2.500, 1000)

    for turbine_i, turbine in enumerate(turbines):
        n_cols = int(n_clusters[turbine_i])
        fig, axes = plt.subplots(nrows=2, ncols=n_cols, figsize=(4*n_cols,2*2))#(3.5*n_cols,2.5*2))

        for cluster_i in range(0, n_clusters[turbine_i]):
            KDE_sum = [0]*len(x)

            for cube_i, kde in enumerate(KDEs[turbine_i][cluster_i]):
                y = kde(x)

                # Plot all KDEs separately
                axes[0, cluster_i].plot(x, y)
                axes[0, cluster_i].set_title("Individual probability distributions of OM "+str(group_counter+1)+" (turbine "+str(turbine_i+1)+' )',  backgroundcolor=colorpal[group_counter])
                axes[0, cluster_i].set_ylim(0, 35)
                axes[0, cluster_i].set_ylabel('Density')
                axes[0, cluster_i].set_xlabel('Active Power [MW]')

            # Plot the weighted sum
            axes[1, cluster_i].plot(x, sum_densities.iloc[group_counter])
            axes[1, cluster_i].set_title("Mixture probability distribution of OM "+str(group_counter+1)+" (turbine "+str(turbine_i+1)+' )', backgroundcolor=colorpal[group_counter])
            axes[1, cluster_i].set_ylim(0, 9)
            axes[1, cluster_i].set_ylabel('Density')
            axes[1, cluster_i].set_xlabel('Active Power [MW]')

            group_counter+=1

        plt.tight_layout();
    

        
def construct_fleetwide_performance_profiles(n_reduced_clusters, final_labels, sum_densities, df_cube_counts, points_per_cube, colorpal, compact=False):
    fig, axes = plt.subplots(nrows=2, 
                         ncols=n_reduced_clusters, 
                         figsize=(5*n_reduced_clusters,2.5*2));
    x = np.linspace(0, 2.500, 1000)

    for j, group in enumerate(np.unique(final_labels)): # For each cluster
        n_points_in_group = 0
        group_profile = [0]*1000
        for i, label in enumerate(final_labels):
            y = sum_densities.loc[i]
            if label is str(group):

                # Caculate group profiles
                group_profile = [a + b*sum(points_per_cube[i]) for a, b in zip(group_profile, y)]
                n_points_in_group += sum(points_per_cube[i])

        # plot group profiles
        if compact: 
            row = 0
            label = 'OM ' + group
            kwargs = {'alpha':0.5, 'color':'grey', 'linewidth':10}
        else: 
            axes[1, j].set_title("Fleet-wide performance profile", size='large')
            axes[1, j].set_ylim(-0.001, 9)
            axes[1, j].set_xlim(0, 2.300)
            axes[1, j].set_ylabel('Density')
            axes[1, j].set_xlabel('Active Power [MW]')
            row = 1
            label = ''
            kwargs = {}
        group_profile = [x/n_points_in_group for x in group_profile]
        
        axes[row, j].plot(x, group_profile, **kwargs, label=label)
        # help(axes[row, j].plot)
        axes[row, j].text(0.5, -4, 'Fleet-wide operating mode ' + group, size='xx-large')
        # axes[1, j].text(300, -0.006, 'Fleet-wide operating mode '+group, size='xx-large')

        for i, label in enumerate(final_labels):
            if label is str(group):
                y = sum_densities.loc[i]

                # Plot clustered proficles
                axes[0, j].plot(x, y, color=colorpal[i], label=df_cube_counts['label_operating_mode'][i])
                axes[0, j].set_title("Individual performance profiles", size='large')# OM "+OM_dict[group], size='large')
                axes[0, j].set_ylim(-0.001, 9)
                axes[0, j].set_xlim(0, 2.300)
                axes[0, j].set_ylabel('Density')
                axes[0, j].set_xlabel('Active Power [MW]')
                axes[0, j].legend()

    plt.tight_layout()

