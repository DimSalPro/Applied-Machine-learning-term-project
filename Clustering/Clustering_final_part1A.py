import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn import preprocessing
from sklearn import metrics
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
import modifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn import tree

sns.set()


# -------------------------------------------------------------------------- #
# Functions
# -------------------------------------------------------------------------- #

# # dor every column, checks the variance and if it is lower than the threshold, it drops it
def variance(data_frame, a=0.8):
    for col in data_frame.columns:
        if data_frame[col].var() <= (a * (1 - a)):
            print()
            print('Feature with Variance under 0.016 : ', col)
            data_frame.drop(col, axis=1, inplace=True)

    return data_frame


# # for every pair of column, it checks the correlation and if it is more than 95% it drops one column of the pair
# # the one with the least variance
def correlation(data_frame, b=0.95):
    for col1 in data_frame.columns:
        for col2 in data_frame.columns:
            try:
                if col1 != col2 and abs(data_frame[col1].corr(data_frame[col2])) > b:
                    print('\nFeatures : ', col2, ' and ', col1, ' have more than 95% correlation')
                    if data_frame[col1].var() < data_frame[col2].var():
                        data_frame.drop(col1, axis=1, inplace=True)
                        print('Feature :', col1, ' has been dropped as it has lower variance')
                    else:
                        data_frame.drop(col2, axis=1, inplace=True)
                        print('Feature :', col2, ' has been dropped as it has lower variance')

            except:
                continue

    return data_frame


# # uses the yellow brick library to provide a plot for the cluster numbers with the best silhouette and inertia score
# # k represents the range of clusters to search for the best parameters
def best_kmeans_clusters(data_input):
    model = KMeans()

    visualizer = KElbowVisualizer(model, k=(2, 10), metric='silhouette', timings=False)
    visualizer.fit(data_input)  # Fit the data to the visualizer
    visualizer.show()

    visualizer = KElbowVisualizer(model, k=(2, 10), metric='distortion', timings=False)
    visualizer.fit(data_input)  # Fit the data to the visualizer
    visualizer.show()


# # implements KMeans clustering and outputs the dataframe with an extra column which has the labels and an array
# # with the labels
def kmeans(data_frame, data_input, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=1)
    kmeans.fit(data_input)
    y_means = kmeans.predict(data_input)
    data_frame['labels'] = kmeans.labels_
    silhouette_values = silhouette_samples(data_input, y_means)
    print('\nKMeans Number of clusters: ', n_clusters)
    print('\nSilhouette values for KMean: ', np.mean(silhouette_values))
    print('\nLabels of the clusters: ')
    for i in range(n_clusters):
        print('\nNumber in cluster ' + str(i) + ' : ', len(kmeans.labels_[kmeans.labels_ == i]))

    return data_frame, kmeans.labels_


# # implements a decision tree classification. The max depth must be small in order to see the decision making
# # attributes. It is the first type of characterization
def characterize_dt(data_frame, max_depth):
    plot_X = data_frame.drop('labels', axis=1)
    X = data_frame.drop('labels', axis=1).values
    Y = data_frame.labels.values
    # X = preprocessing.scale(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, min_samples_leaf=1)
    clf.fit(X_train, Y_train)
    Y_train_pred_DT = clf.predict(X_train)
    Y_test_pred_DT = clf.predict(X_test)
    cmDT_test = metrics.confusion_matrix(Y_test, Y_test_pred_DT, labels=None)
    cmDT_train = metrics.confusion_matrix(Y_train, Y_train_pred_DT, labels=None)
    print('\nDecision Tree confusion matrix for test:\n', cmDT_test)
    print('\nDecision Tree confusion matrix for test:\n', cmDT_train)
    print('\n Decision Tree metrics: ')
    print(metrics.precision_recall_fscore_support(Y_test, Y_test_pred_DT, average='macro'))
    fig = plt.figure(1, figsize=(15, 15))
    _ = tree.plot_tree(clf,
                       feature_names=plot_X.columns,
                       filled=True)
    fig.savefig("Decision_tree.png")
    plt.show()
    text_representation = tree.export_text(clf, feature_names=list(plot_X.columns))
    print('\n', text_representation)


# # represents every attribute of the of the dataset in a boxplot for every different cluster
# # this was used in order to compare more easily the clusters
def box_plots(data_frame, n_clusters, title, noise='off'):
    if noise == 'off':
        for col in data_frame.columns:
            data = [data_frame[data_frame['labels'] == i][col] for i in range(n_clusters)]
            fig7, ax7 = plt.subplots()
            ax7.set_title(title + ' for column ' + col)
            ax7.boxplot(data, labels=[str(i) for i in range(n_clusters)])
            plt.show()

    if noise == 'on':
        for col in data_frame.columns:
            data = [data_frame[data_frame['labels'] == i][col] for i in range(-1, n_clusters - 1)]
            fig7, ax7 = plt.subplots()
            ax7.set_title(title + ' for column ' + col)
            ax7.boxplot(data, labels=[str(i) for i in range(-1, n_clusters - 1)])
            plt.show()


# # provides boxplot for each cluster for all the columns
def box_plots_whole(data_frame, n_clusters, title, noise='off'):
    if noise == 'off':
        for i in range(n_clusters):
            plt.figure(1, figsize=(20, 10))
            plt.boxplot(data_frame[data_frame['labels'] == i].values, labels=data_frame.columns)
            plt.title(title + ' for cluster ' + str(i))
            plt.xticks(rotation=90)
            plt.show()
    if noise == 'on':
        for i in range(-1, n_clusters - 1):
            plt.figure(1, figsize=(20, 10))
            plt.boxplot(data_frame[data_frame['labels'] == i].values, labels=data_frame.columns)
            plt.title(title + ' for cluster ' + str(i))
            plt.xticks(rotation=90)
            plt.show()


# # uses pca with 2 components to create a plot
def pca_viz(data_input, n_colors, labels, title):
    pca = PCA(n_components=2)
    X = pca.fit_transform(data_input)

    print('explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels,
                    palette=sns.color_palette('muted', n_colors=n_colors))
    plt.title('Pca for ' + title)
    plt.show()


# # implements dbscan clustering and outputs the dataframe with the labels and the array of labels
def dbscan(data_frame, data_input, epsilon, mn_pts):
    db = DBSCAN(eps=epsilon, min_samples=mn_pts).fit(data_input)
    data_frame['labels'] = db.labels_
    n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    print('\nDbscan number of clusters: ', n_clusters_)
    print('\nSilhouette values for Dbscan ', metrics.silhouette_score(data_input, db.labels_))
    print('\nLabels of the clusters: ')
    for i in data_frame.labels.unique():
        print('\nNumber in cluster ' + str(i) + ' : ', len(db.labels_[db.labels_ == i]))

    return data_frame, db.labels_


# # takes as argument the data_frame and computes the distance of each point from the n_neighbors. Provides a plot
# # for the distances sorted. In the max curviture we find the best epsilon to use.
def k_nearest_plot(data_input, n_neighbors):
    neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    neighbors_fit = neighbors.fit(data_input)
    distances, indices = neighbors_fit.kneighbors(data_input)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.plot(distances)
    plt.title('K nearest neighbors distances in sorted order')
    plt.show()


# # runs in a for loop different values for epsilon and min_samples and creates a plot in order to find the parameters
# # with the maximum number of silhouette
def best_dbscan_clusters(data_input):
    silhouette_values = []
    clustersAll = []

    for i in [4, 4.5, 5, 5.5, 6, 6.5]:
        for j in range(90, 110):
            db = DBSCAN(eps=i, min_samples=j).fit(data_input)

            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

            if n_clusters_ > 1 and n_clusters_ < len(data_input):
                silhouette_values.append(metrics.silhouette_score(data_input, labels))
                clustersAll.append(n_clusters_)
                print(i, j, metrics.silhouette_score(data_input, labels), n_clusters_, labels)

    plt.figure(3)
    plt.title('Silhouette score for DBScan:')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Values')
    plt.plot(clustersAll, silhouette_values, '*')
    plt.show()


# # uses the yellow brick library to provide a plot for the cluster numbers with the best silhouette and inertia score
# # k represents the range of clusters to search for the best parameters
def best_hierarchical_clusters(data_input):
    model = AgglomerativeClustering()

    visualizer = KElbowVisualizer(model, k=(2, 10), metric='silhouette', timings=False)
    visualizer.fit(data_input)  # Fit the data to the visualizer
    visualizer.show()

    visualizer = KElbowVisualizer(model, k=(2, 10), metric='distortion', timings=False)
    visualizer.fit(data_input)  # Fit the data to the visualizer
    visualizer.show()


# # implements hierarchical clustering and outputs the dataframe with the labels and the array of labels
def hierarchical(data_frame, data_input, n_clusters):
    Hclustering = AgglomerativeClustering(n_clusters=n_clusters,
                                          affinity='euclidean', linkage='ward', compute_full_tree=True)
    Hclustering.fit(data_input)
    silhouette = silhouette_score(data_input, Hclustering.labels_)
    data_frame['labels'] = Hclustering.labels_
    print('\nHierarchical Number of clusters: ', n_clusters)
    print('\nSilhouette values for Hierarchical: ', silhouette)
    print('\nLabels of the clusters: ')
    for i in range(n_clusters):
        print('\nNumber in cluster ' + str(i) + ' : ', len(Hclustering.labels_[Hclustering.labels_ == i]))

    return data_frame, Hclustering.labels_


# -------------------------------------------------------------------------- #
# Preprocessing
# -------------------------------------------------------------------------- #
df = pd.read_csv('USCensus1990.data.txt', header=0, sep=',')

# # drop caseid as it is a feature with unique ids
df.drop('caseid', axis=1, inplace=True)

# # check nan values
missing_data = pd.DataFrame({'percent_missing': df.isnull().sum() * 100 / len(df)})
with pd.option_context("display.max_rows", None):
    print(missing_data.sort_values('percent_missing', ascending=False))

# sampling random using 1% of the dataset
sample = 0.01
df = df.sample(frac=sample, random_state=1)

# lower data size using a custom function
df = modifier.bits(df)

# # replace bad mapped prices. Same keys had been mapped to the same values
df['iRPOB'] = df['iRPOB'].replace(10, 0).replace(21, 1).replace(22, 2).replace(23, 3).replace(24, 4).replace(31, 5) \
    .replace(32, 6).replace(33, 7).replace(34, 8).replace(35, 9).replace(36, 10).replace(40, 11) \
    .replace(51, 12).replace(52, 13)

df['iRemplpar'] = df['iRemplpar'].replace(111, 11).replace(112, 1).replace(113, 2).replace(114, 3).replace(121,
                                                                                                           4).replace(
    122, 5) \
    .replace(133, 6).replace(134, 7).replace(141, 8).replace(211, 4).replace(212, 5).replace(213, 9) \
    .replace(221, 6).replace(222, 7).replace(223, 10)

# # first we drop the highly correlated
df = correlation(df)

# # second we drop the ones with the lowest variance
df = variance(df)

# # scale the data
X = df.values
X = preprocessing.scale(X)

# -------------------------------------------------------------------------- #
# KMeans process implementation of the above functions
# -------------------------------------------------------------------------- #

# # find best parameters
best_kmeans_clusters(X)

# # run for the best parameters
data_kmeans, labels_kmeans = kmeans(df, X, 3)

# # characterize the clusters
box_plots_whole(data_kmeans, 3, 'KMeans')

characterize_dt(data_kmeans, 2)

box_plots(data_kmeans, 3, 'KMeans')

# # visualize the clusters
pca_viz(X, 3, labels_kmeans, 'KMeans')

# -------------------------------------------------------------------------- #
# Dbscan process implementation of the above functions
# -------------------------------------------------------------------------- #

# # find the best epsilon (approximately)
k_nearest_plot(X, 96)

# # find best parameters, best were found for epsilon 5 and min_samples 96
best_dbscan_clusters(X)

# # run for the best parameters
data_db, labels_dbscan = dbscan(df, X, 5, 96)

# # characterize the clusters
box_plots_whole(data_db, 3, 'DBScan', 'on')

characterize_dt(data_db, 2)

box_plots(data_db, 3, 'DBScan', 'on')

# # visualize the clusters
pca_viz(X, 3, labels_dbscan, 'DBScan')

# -------------------------------------------------------------------------- #
# Hierarchical process implementation of the above functions
# -------------------------------------------------------------------------- #

# # find best parameters
best_hierarchical_clusters(X)

# # run for the best parameters
data_hierarchical, labels_hierarchical = hierarchical(df, X, 4)

# # characterize the clusters
box_plots_whole(data_hierarchical, 4, 'Hierarchical')

characterize_dt(data_hierarchical, 2)

box_plots(data_hierarchical, 4, 'Hierarchical')

# # visualize the clusters
pca_viz(X, 4, labels_hierarchical, 'Hierarchical')
