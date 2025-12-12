import sqlite3
import pandas as pd
from pathlib import Path

def create_database(db_path, schema_path):
    conn = sqlite3.connect(db_path)
    with open(schema_path, 'r') as f:
        conn.executescript(f.read())
    conn.commit()
    return conn

def populate_artists(conn, df):
    artists = df['artists'].str.split(';').explode().str.strip().unique()
    artists = [(artist,) for artist in artists if pd.notna(artist)]
    conn.executemany("INSERT OR IGNORE INTO artists (artist_name) VALUES (?)", artists)
    conn.commit()

def populate_albums(conn, df):
    artist_map = pd.read_sql("SELECT artist_id, artist_name FROM artists", conn).set_index('artist_name')['artist_id'].to_dict()
    
    albums_data = []
    for _, row in df.iterrows():
        if pd.isna(row['album_name']) or pd.isna(row['artists']):
            continue
        artist_name = row['artists'].split(';')[0].strip()
        if artist_name in artist_map:
            albums_data.append((row['album_name'], artist_map[artist_name]))
    
    conn.executemany("INSERT OR IGNORE INTO albums (album_name, artist_id) VALUES (?, ?)", albums_data)
    conn.commit()

def populate_songs(conn, df):
    artist_map = pd.read_sql("SELECT artist_id, artist_name FROM artists", conn).set_index('artist_name')['artist_id'].to_dict()
    album_map = pd.read_sql("SELECT album_id, album_name, artist_id FROM albums", conn)
    album_map = album_map.set_index(['album_name', 'artist_id'])['album_id'].to_dict()
    
    songs_data = []
    for _, row in df.iterrows():
        try:
            if pd.isna(row['artists']) or pd.isna(row['album_name']):
                continue
            
            artist_name = row['artists'].split(';')[0].strip()
            if artist_name not in artist_map:
                continue
            
            artist_id = artist_map[artist_name]
            album_key = (row['album_name'], artist_id)
            if album_key not in album_map:
                continue
            
            songs_data.append((
                row['track_id'], row['track_name'], artist_id, album_map[album_key], row['track_genre'],
                int(row['popularity']), int(row['duration_ms']), 1 if row['explicit'] else 0,
                float(row['danceability']), float(row['energy']), float(row['loudness']),
                float(row['speechiness']), float(row['acousticness']), float(row['instrumentalness']),
                float(row['liveness']), float(row['valence']), int(row['key']),
                int(row['mode']), float(row['tempo']), int(row['time_signature'])
            ))
        except (KeyError, ValueError):
            continue
    
    conn.executemany("""
        INSERT OR IGNORE INTO song_features (
            track_id, track_name, artist_id, album_id, genre, popularity, duration_ms, explicit,
            danceability, energy, loudness, speechiness, acousticness, instrumentalness,
            liveness, valence, key, mode, tempo, time_signature
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, songs_data)
    conn.commit()

def insert_model_performance(conn):
    models = [
        ('rf_v1.0', 'Random Forest', 0.78, 0.66, 0.63, 0.64, 0.84, 'n_estimators=500, random_state=3'),
        ('xgb_v1.0', 'XGBoost', 0.75, 0.68, 0.57, 0.62, 0.81, 'learning_rate=0.1, max_depth=9, n_estimators=500'),
        ('knn_v1.0', 'K-Nearest Neighbors', 0.72, 0.71, 0.53, 0.60, 0.80, 'n_neighbors=3, weights=distance, p=1'),
        ('logreg_v1.0', 'Logistic Regression', 0.58, 0.64, 0.38, 0.47, 0.64, 'max_iter=1000')
    ]
    conn.executemany("""
        INSERT OR IGNORE INTO model_performance (
            model_version, model_type, accuracy, precision_score, recall_score, 
            f1_score, auc_score, hyperparameters
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, models)
    conn.commit()

def main():
    project_root = Path(__file__).parent
    data_path = project_root / 'spotipy.csv'
    schema_path = project_root / 'spotipy_schema.sql'
    db_path = project_root / 'spotify.db'
    
    df = pd.read_csv(data_path)
    conn = create_database(db_path, schema_path)
    
    populate_artists(conn, df)
    populate_albums(conn, df)
    populate_songs(conn, df)
    insert_model_performance(conn)
    
    conn.close()

if __name__ == "__main__":
    main()