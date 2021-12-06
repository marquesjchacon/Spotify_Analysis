import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math
import numpy as np
#import cse163_utils  # noqa: F401

# Sklearn imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error as rmse
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

#print(__file__)
# read in main dataframe as df
df = pd.read_csv('./InputData/SpotifyFeatures.csv')
df = df.dropna()
# filter out columns to create new dataframes
popularity = df['popularity']
features = ['acousticness', 'danceability', 'duration_ms', 'energy',
            'instrumentalness', 'liveness', 'loudness', 'speechiness',
            'tempo', 'valence']
gen_feat_list = ['acousticness', 'danceability', 'duration_ms', 'energy',
                 'instrumentalness', 'liveness', 'loudness', 'speechiness',
                 'tempo', 'valence', 'genre']
pop_feat_list = ['popularity', 'acousticness', 'danceability', 'duration_ms',
                 'energy', 'instrumentalness', 'liveness', 'loudness',
                 'speechiness', 'tempo', 'valence']

# filtered dataframes needed
music_features = df.filter(items=features)
genre_features = df.filter(items=gen_feat_list)
popularity_features = df.filter(items=pop_feat_list)

# list of genres (total of 26 genres)
gen = ['Opera', 'A Capella', 'Alternative', 'Blues', 'Dance', 'Pop',
       'Electronic', 'R&B', 'Children’s Music', 'Folk', 'Anime', 'Rap',
       'Classical', 'Reggae', 'Hip-Hop', 'Comedy', 'Country', 'Reggaeton',
       'Ska', 'Indie', 'Rock', 'Soul', 'Soundtrack', 'Jazz', 'World',
       'Movie']


def main():
    # Questions 1 and 2
    plot_genre_distribution(df)
    plot_popularity_corr(popularity_features)
    svr_4_features_pop(df)
    lasso_pop(df)
    plot_speechiness(df)
    genre_classifier_knn(genre_features, gen)

    # Questions 3 and 4
    whole_data = pd.read_csv('./InputData/SpotifyFeatures.csv')
    playlist = pd.read_csv('./InputData/Playlist_CSVs/Iron_Steel.csv', usecols=range(0, 9))
    merged = join_playlist_with_dataset(whole_data, playlist)
    playlist_zscores = set_up_zscores(whole_data, merged)
    scatterplot_zscores(playlist_zscores)
    playlist_zscores = playlist_zscores.drop(['indices'], axis=1)
    avg_zscore_bar_chart(playlist_zscores)
    plot_categorical_data(whole_data, merged)
    dataset_zscores = set_up_zscores(whole_data, whole_data)
    dataset_zscores = filter_acoustic_data(dataset_zscores)
    given_song_zscores = set_up_zscores(whole_data, pd.DataFrame(merged.iloc
                                        [len(merged) // 2]).transpose())
    given_song_zscores = filter_acoustic_data(given_song_zscores)
    song_acoustics = given_song_zscores.iloc[0].drop(['artist_name',
                                                      'track_name'])
    given_acoustics = song_acoustics.to_numpy()
    mae_per_song = {}
    rmse_per_song = {}
    for index, row in dataset_zscores.iterrows():
        acoustic_data = row.drop(['artist_name', 'track_name']).to_numpy()
        mae = mean_absolute_error(acoustic_data, given_acoustics)
        mae_per_song[row['artist_name'] + ' \"' + row['track_name'] +
                     '\"'] = mae
        rmse_song = math.sqrt(mean_squared_error(acoustic_data, given_acoustics))
        rmse_per_song[row['artist_name'] + ' \"' + row['track_name'] +
                      '\"'] = rmse_song
    mae_playlist = pd.Series(mae_per_song,
                             index=mae_per_song.keys()).sort_values()
    rmse_playlist = pd.Series(rmse_per_song,
                              index=rmse_per_song.keys()).sort_values()
    print(mae_playlist.iloc[:51])
    print(rmse_playlist.iloc[:51])
    print('DONE')



def plot_genre_distribution(DF):
    """
    Plots distribution of genres in the spotify dataset
    Assumes the dataframe given has columns genre and track_id
    """
    mask = DF.groupby('genre')['track_id'].count()
    plt.figure(figsize=(16, 7))
    sns.set()
    f = sns.barplot(x=mask.index, y=mask.values)
    f.set_xticklabels(rotation=30, labels=mask.index)
    plt.savefig('./Output/figure_1_count_songs_in_genres')


def plot_popularity_corr(popularity_features):
    """
    Calculates correlation and plots a heat map for
    musical feature's correation with popularity scores
    Assumes the input dataframe music_features has all relevant columns.
    """
    corr = popularity_features.corr()
    plt.figure(figsize=(15, 15))
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Spotify Music Features')
    plt.savefig('./Output/figure_2_populiarty_correlation')


def svr_4_features_pop(DF):
    """
    Trains Support Vector Machine - Regressor
    predicts popularity score of a pop music genre using 4 selected features.
    Takes in a dataframe as input and prints out R2 Score and
    RMSE from the model.
    """
    pop_df = DF[DF['genre'] == 'Pop']
    pop_features = pop_df.filter(items=['acousticness', 'danceability',
                                        'energy', 'loudness'])
    pop_labels = pop_df['popularity']
    pop_labels = np.array(pop_labels).flatten()
    X = pop_features
    target = pop_labels
    X_train, X_test, y_train, y_test = train_test_split(X, target,
                                                        test_size=0.3,
                                                        random_state=0)
    model = svm.SVR(kernel='linear', C=1).fit(X_train, y_train)
    print('Score: ', model.score(X_test, y_test))
    predicted = model.predict(X_test)
    print('RMSE: ', rmse(predicted, y_test))
    print('The popularity score ranges from',
          min(pop_labels), 'to', max(pop_labels))


def lasso_pop(DF):
    """
    Trains Lasso - Regressor
    predicts popularity score of a pop music genre using 4 selected features.
    Takes in a dataframe as input and prints out R2 Score and
    RMSE from the model. The function prints out 7 different alpha values.
    """
    pop_df = DF[DF['genre'] == 'Pop']
    X = pop_df.filter(items=['acousticness', 'danceability',
                             'energy', 'loudness'])
    y = np.array(pop_df['popularity']).flatten()
    # holds out values
    X_, X_hold, y_, y_hold = train_test_split(X, y, test_size=0.2,
                                              random_state=0)
    for a in [.1, .5, .7, .9, .95, .99, 1]:
        X_train, X_test, y_train, y_test = \
            train_test_split(X_, y_, test_size=0.3, random_state=0)
        lasso_model = Lasso(alpha=a, fit_intercept=True,
                            copy_X=True, max_iter=1000,
                            tol=0.0001, warm_start=False,
                            positive=False, random_state=None,
                            selection='cyclic')
        lasso_model.fit(X_train, y_train)
        predicted = lasso_model.predict(X_test)
        print('Results for alpha = ', a)
        print('R2 score: ', lasso_model.score(X_test, y_test))
        print('RMSE: ', rmse(predicted, y_test))


def translate_genre(genre_features):
    """
    Translates genres to number labels
    """
    gen = ['Opera', 'A Capella', 'Alternative', 'Blues', 'Dance', 'Pop',
           'Electronic', 'R&B', 'Children’s Music', 'Folk', 'Anime', 'Rap',
           'Classical', 'Reggae', 'Hip-Hop', 'Comedy', 'Country', 'Reggaeton',
           'Ska', 'Indie', 'Rock', 'Soul', 'Soundtrack', 'Jazz', 'World',
           'Movie']
    trans_dict = {gen[i]: i for i in range(len(gen))}
    genre_features['number_label'] = genre_features['genre'].map(trans_dict)
    return genre_features


def plot_speechiness(df):
    """
    Takes in original dataframe as input and plots
    speechiness feature for different genres.
    Produces figure 3
    """
    plt.figure(figsize=(27, 10))
    sns.boxplot(x="genre", y="speechiness", data=df)
    plt.savefig('./Output/figure_3_speechiness_box_plot')


# Code Credits: sklearn confusion matrix documentation
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=(20, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax



def genre_classifier_knn(genre_features, gen):
    """
    Takes in an input list of genres and musical feature dataframe
    Performs a grid search cross validation using pipeline and GridSearchCV
    Trains a K-Nearest Neighbor classifier and prints out best parameters
    and its scores. This function will put out multiple line of warnings,
    but the warnings will not impact the results.

    Confusion matrix is plotted and printed with the best parameters - figure 4
    """
    nl_genre_features = translate_genre(genre_features)
    X = nl_genre_features.drop(['number_label', 'genre'], axis=1)
    y = nl_genre_features['number_label']
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3, stratify=df.genre)
    scaler = MinMaxScaler()
    knn_classifier = KNeighborsClassifier()
    # Creates pipeline
    pipe = make_pipeline(scaler, knn_classifier)

    # Disclaimer: Running grid search takes a long time, so k is already
    # set to a known best parameter k = 20. To reproduce the grid search
    # uncomment the lower k with 7 values and comment out k and weights.
    # k = np.array([1, 3, 5, 7, 9, 15, 20])
    # weights = ["uniform", "distance"]
    k = [20]
    weights = ["distance"]
    param_grid = {'kneighborsclassifier__n_neighbors': k,
                  'kneighborsclassifier__weights': weights}

    # Performs grid search of pipeline
    grid = GridSearchCV(pipe, param_grid, scoring="neg_mean_absolute_error")

    # Perform grid search and train a model
    grid.fit(X_train, y_train)
    print('Score: ', grid.score(X_test, y_test))
    print('Best Parameters: ', grid.best_params_)
    predicted = grid.predict(X_test)
    preds = predicted.tolist()
    labels = y_test.values.tolist()
    np.set_printoptions(precision=2)
    # Plot normalized confusion matrix
    plt.figure(figsize=(27, 10))
    plot_confusion_matrix(labels, preds, classes=np.array(gen),
                          normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig('./Output/figure_4_knn_confusion_matrix')


def join_playlist_with_dataset(dataset, playlist):
    """
    merges the playlist with the dataset
    """
    merged = playlist.merge(dataset, right_on=['artist_name', 'track_name'],
                            left_on=['Artist Name', 'Track Name'])
    merged = merged.drop(['\ufeffSpotify URI', 'Track Name', 'Artist Name',
                          'Album Name', 'Disc Number', 'Track Number',
                          'Track Duration (ms)', 'Added By', 'Added At'],
                         axis=1)
    merged = merged.drop_duplicates(subset=['artist_name', 'track_name'])
    swap_first_and_middle(merged)
    return merged


def set_up_zscores(dataset, playlist):
    """
     Calculates the zscores by calculating mean of the dataset,
     standard dev. and uses the calculate zscore function to get the zscores,
     and it returns a dataframe with artist, song name, and the zscores of the
     acoustic attributes
    """
    quantitative_data = dataset.drop_duplicates(subset=['artist_name',
                                                        'track_name'])
    quantitative_data = quantitative_data.drop(['genre', 'track_id', 'key',
                                                'mode', 'time_signature'],
                                               axis=1)
    playlist_quantitative_data = playlist.drop(['genre', 'track_id', 'key',
                                                'mode', 'time_signature'],
                                               axis=1)
    playlist_quantitative_data = \
        playlist_quantitative_data.drop_duplicates(subset=['artist_name',
                                                           'track_name'])
    artists_tracks = playlist_quantitative_data.loc[:, :'track_name']
    means = quantitative_data.mean(axis=0, numeric_only=True)
    stds = quantitative_data.std(axis=0, numeric_only=True)
    playlist_quantitative_data = playlist_quantitative_data.loc[:,
                                                                'popularity':]
    zscores = create_playlist_zscores(playlist_quantitative_data, means, stds)
    return pd.concat([artists_tracks, zscores], axis=1)


def plot_categorical_data(dataset, playlist):
    """
    filters the dataframe for categorical data and calls the
    categorical_ratios function for plotting.
    """
    playlist_categorical_data = playlist.drop(['genre', 'track_name',
                                               'track_id', 'popularity',
                                               'acousticness', 'danceability',
                                               'duration_ms', 'energy',
                                               'instrumentalness', 'liveness',
                                               'loudness', 'speechiness',
                                               'tempo', 'valence'], axis=1)
    categorical_data = \
        dataset.drop_duplicates(subset=['artist_name', 'track_name'])
    categorical_data = categorical_data.drop(['genre', 'track_name',
                                              'track_id', 'popularity',
                                              'acousticness', 'danceability',
                                              'duration_ms', 'energy',
                                              'instrumentalness', 'liveness',
                                              'loudness', 'speechiness',
                                              'tempo', 'valence'], axis=1)
    categorical_ratios(playlist_categorical_data, categorical_data)


def swap_first_and_middle(data):
    """
    moves the given song to the middle of the dataset
    so that it looks pretty when plotting
    """
    initial_song, middle_song = data.iloc[0], data.iloc[len(data) // 2]
    data.iloc[0] = middle_song
    data.iloc[len(data) // 2] = initial_song


def calculate_zscores(given_song, sample_means, sample_stds):
    """
    calculates the zscores by iterating over the given playlist
    and use the given mean and standard dev.
    """
    return (given_song - sample_means) / sample_stds


def create_playlist_zscores(playlist, data_means, data_stds):
    return playlist.apply(lambda row:
                          calculate_zscores(row, sample_means=data_means,
                                            sample_stds=data_stds), axis=1)


def scatterplot_zscores(scores):
    """
    creates a scatterplot of zscores for each acoustic attribute,
    put on one file and it is saved to a file
    """
    fig, axes = plt.subplots(4, 3, figsize=(6.4, 6))
    colors = ['red' if i == len(scores.index) // 2 else 'grey' for i in
              range(len(scores.index))]
    i = 0
    scores['indices'] = scores.index
    scores.indices = scores.indices.astype(str)
    sns.set_palette(tuple(colors))
    for ax in axes.reshape(-1):
        if i < 11:
            attribute = scores.columns[i + 2]
            sns.swarmplot(x='indices', y=attribute, ax=ax, data=scores)
            ax.set_xticks([], [])
            ax.set_xlabel('')
            if i % 3 == 0:
                ax.set_ylabel('Z-scores', labelpad=5, fontsize='x-small')
            else:
                ax.set_ylabel('')
            ax.set_title(attribute.upper()[0] + attribute.lower()[1:] +
                         ' Dist.', pad=1, fontsize='small')
            ax.set_ylim((-3.5, 3.5))
            i = i + 1
        else:
            ax.axis('off')
            ax.legend(loc='center', labels=['Given Song', 'Recommended Song'],
                      handles=[matplotlib.patches.Patch(color='red'),
                      matplotlib.patches.Patch(color='grey')])
    plt.suptitle('Distribution of Acoustic Features', fontsize='x-large')
    plt.savefig('./Output/z-score_scatterplot.png')


def avg_zscore_bar_chart(scores):
    """
    creates a bar chart comparing the zscore of the given songs acoustic
    attributes compared to the average of the playlist
    """
    fig, axes = plt.subplots(4, 3, figsize=(6.4, 6))
    colors = ['red' if i == 0 else 'grey' for i in range(2)]
    i = 0
    given_song = pd.DataFrame(scores.iloc[len(scores) // 2]).transpose()
    recommended_songs = scores.drop(scores.index[len(scores) // 2])
    given_song = \
        given_song.loc[:, 'popularity':]. \
        append(pd.DataFrame(recommended_songs.mean(axis=0,
                                                   numeric_only=True)
                            ).transpose()) + 2
    print(given_song)
    for ax in axes.reshape(-1):
        if i < 11:
            attribute = scores.columns[i + 2]
            given_song.plot.bar(ax=ax, y=attribute, use_index=False,
                                legend=False, color=colors)
            ax.set_title(attribute.upper()[0] + attribute.lower()[1:] +
                         ' Comparison.', pad=1, fontsize='small')
            ax.set_xticks([], [])
            ax.set_xlabel('')
            ax.set_ylim((0, 4))
            if i % 3 == 0:
                ax.set_ylabel('Z-scores', labelpad=5, fontsize='x-small')
            i = i + 1
        else:
            ax.axis('off')
            ax.legend(loc='center', labels=['Given Song', 'Playlist Average'],
                      handles=[matplotlib.patches.Patch(color='red'),
                      matplotlib.patches.Patch(color='grey')])
    plt.suptitle('Song Z-Scores (Shifted Up 2) Compared to Playlist Averages')
    plt.savefig('./Output/avg_z-score_bar_chart.png')


def categorical_ratios(playlist_data, dataset):
    """
    plots a bar chart of the categorical data and
    denotes importance by higher bars
    """
    fig, [ax1, ax2] = plt.subplots(2, 1)
    given_song = \
        pd.DataFrame(playlist_data.iloc[len(playlist_data) // 2]).transpose()
    updated_playlist = \
        playlist_data.drop(playlist_data.index[len(playlist_data) // 2])
    attribute_ratios = {}
    for attribute in given_song.columns:
        playlist_ratio = \
            len(updated_playlist[updated_playlist[attribute] ==
                                 given_song.iloc[0][attribute]]
                ) / len(updated_playlist)
        dataset_ratio = \
            len(dataset[dataset[attribute] ==
                        given_song.iloc[0][attribute]]) / len(dataset)
        attribute_ratios[attribute] = (playlist_ratio / dataset_ratio)
    ratios = pd.Series(attribute_ratios)
    ratios.plot.bar(ax=ax1, title='For A Given Song\'s Attributes')
    plt.suptitle('Ratio of Occurence Proportion in Playlist vs. Dataset')
    ratios.plot.bar(ax=ax2, title='Zoomed In to Show Smaller Attributes')
    ax2.set_ylim((0, 2))
    ax1.set_xticklabels(labels=['Artist Name', 'Key', 'Mode',
                                'Time Signature'], rotation='horizontal')
    ax2.set_xticklabels(labels=['Artist Name', 'Key', 'Mode',
                                'Time Signature'], rotation='horizontal')
    pos = ax2.get_position()
    ax2.set_position([pos.x0, pos.y0 - 0.05, pos.width, pos.height])
    plt.savefig('./Output/categorical_data_ratios.png')


def filter_acoustic_data(playlist):
    """
    filters the dataframe just for quantitative acoustic data
    """
    return playlist.drop(['popularity', 'duration_ms'], axis=1)


if __name__ == "__main__":
    main()
