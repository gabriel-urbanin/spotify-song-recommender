import pandas as pd
import pickle
from collections import defaultdict
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

VOLUME__PATH = '/app/model/song_recommendations.pkl'
DATASET_URL = 'https://raw.githubusercontent.com/gabriel-urbanin/song-recommender-config/refs/heads/main/spotify_dataset.csv'

def fetch_and_process_dataset() -> None:
    df_playlists = _fetch_dataset()
    rules = _create_rules_from_dataset(df_playlists)
    recommendations = _create_song_recommendations(rules)

    _export_song_recommendations(recommendations)

def _fetch_dataset() -> None:
    try:
        df = pd.read_csv(DATASET_URL)
        print('Dataset successfully fetched from Github and loaded as Dataframe!')
        return df
    except Exception as e:
        print(f'Could not fetch dataset from GitHub: {e}')
        return


def _create_rules_from_dataset(df_playlists: pd.DataFrame) -> pd.DataFrame:
    playlists = df_playlists.groupby('pid')['track_name'].apply(list).tolist()
    
    te = TransactionEncoder()
    encoded_playlists = te.fit(playlists).transform(playlists)
    df_encoded_playlists = pd.DataFrame(encoded_playlists, columns=te.columns_)
    
    print('Creating association rules from playlist data. Please wait a few minutes...')

    frequent_itemsets = fpgrowth(df_encoded_playlists, min_support=0.03, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7, num_itemsets=len(playlists))
    high_lift_rules = rules[rules['lift'] > 1]

    print(f'Successfully created {len(high_lift_rules)} rules!')

    return high_lift_rules
     
def _create_song_recommendations(rules: pd.DataFrame) -> defaultdict:
    if rules is None:
        return
    
    recommendations = defaultdict(set)
    for antecedents, consequents in zip(rules['antecedents'], rules['consequents']):
        for antecedent in antecedents:
            recommendations[antecedent].update(consequents)    

    recommendations['default_recommendation'].update(_get_songs_with_high_support(rules))

    return recommendations

def _get_songs_with_high_support(rules: pd.DataFrame) -> list:
    rules_sorted_by_support = rules.sort_values(by='support', ascending=False)
    top_5_most_popular_songs = rules_sorted_by_support['antecedents'].head(5).tolist()
    
    default_recommendations = []
    for song in top_5_most_popular_songs:
        default_recommendations.extend(song)
        
    return default_recommendations

def _export_song_recommendations(recommendations: defaultdict) -> None:
    try:
        with open(VOLUME__PATH, 'rb') as file:
            print('A previous set of recommendations was found. Updating it with new recommendations...')
            current_recommendations = pickle.load(file)
    except FileNotFoundError:
        print('No previous set of recommendations was found. Exporting the new recommendations...')
        current_recommendations = defaultdict(set)

    for antecedents, consequents in recommendations.items():
        current_recommendations[antecedents].update(consequents)

    try:
        with open(VOLUME__PATH, 'wb') as file:
            pickle.dump(current_recommendations, file)
        print('Recommendations successfully exported!')
    except Exception as e:
        print(f'Failed to export recommendations: {e}')