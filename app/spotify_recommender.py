import pandas as pd
import pickle
from collections import defaultdict
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

VOLUME__PATH = 'app/model/song_recommendations.pkl'
DATASET_URL = 'https://raw.githubusercontent.com/gabriel-urbanin/song-recommender-config/refs/heads/main/spotify_dataset.csv'

def fetch_and_process_dataset():
    df_playlists = _fetch_dataset()
    rules = _create_rules_from_dataset(df_playlists)
    recommendations = _create_song_recommendations(rules)

    _export_song_recommendations(recommendations)

def _fetch_dataset():
    try:
        df = pd.read_csv(DATASET_URL)
        print('Dataset loaded successfully as Dataframe!')
        return df
    except Exception as e:
        print('Could not fetch dataset from GitHub: {e}')
        return


def _create_rules_from_dataset(df_playlists: pd.DataFrame) -> pd.DataFrame:
    playlists = df_playlists.groupby('pid')['track_name'].apply(list).tolist()
    
    te = TransactionEncoder()
    encoded_playlists = te.fit(playlists).transform(playlists)
    df_encoded_playlists = pd.DataFrame(encoded_playlists, columns=te.columns_)

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
        recommendations[antecedents].update(consequents)

    return recommendations

def _export_song_recommendations(recommendations):
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
        print('Failed to export recommendations: {e}')