DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS model_performance;
DROP TABLE IF EXISTS song_features;
DROP TABLE IF EXISTS albums;
DROP TABLE IF EXISTS artists;

CREATE TABLE artists (
    artist_id INTEGER PRIMARY KEY AUTOINCREMENT,
    artist_name TEXT NOT NULL UNIQUE
);

CREATE INDEX idx_artist_name ON artists(artist_name);

CREATE TABLE albums (
    album_id INTEGER PRIMARY KEY AUTOINCREMENT,
    album_name TEXT NOT NULL,
    artist_id INTEGER NOT NULL,
    FOREIGN KEY (artist_id) REFERENCES artists(artist_id),
    UNIQUE(album_name, artist_id)
);

CREATE INDEX idx_album_name ON albums(album_name);
CREATE INDEX idx_album_artist ON albums(artist_id);

CREATE TABLE song_features (
    song_id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id TEXT NOT NULL UNIQUE,
    track_name TEXT NOT NULL,
    artist_id INTEGER NOT NULL,
    album_id INTEGER NOT NULL,
    genre TEXT NOT NULL,
    popularity INTEGER NOT NULL CHECK(popularity BETWEEN 0 AND 100),
    duration_ms INTEGER NOT NULL CHECK(duration_ms > 0),
    explicit BOOLEAN NOT NULL DEFAULT 0,
    
    danceability REAL NOT NULL CHECK(danceability BETWEEN 0 AND 1),
    energy REAL NOT NULL CHECK(energy BETWEEN 0 AND 1),
    loudness REAL NOT NULL,
    speechiness REAL NOT NULL CHECK(speechiness BETWEEN 0 AND 1),
    acousticness REAL NOT NULL CHECK(acousticness BETWEEN 0 AND 1),
    instrumentalness REAL NOT NULL CHECK(instrumentalness BETWEEN 0 AND 1),
    liveness REAL NOT NULL CHECK(liveness BETWEEN 0 AND 1),
    valence REAL NOT NULL CHECK(valence BETWEEN 0 AND 1),
    
    key INTEGER NOT NULL CHECK(key BETWEEN 0 AND 11),
    mode INTEGER NOT NULL CHECK(mode IN (0, 1)),
    tempo REAL NOT NULL CHECK(tempo > 0),
    time_signature INTEGER NOT NULL CHECK(time_signature BETWEEN 0 AND 7),
    
    FOREIGN KEY (artist_id) REFERENCES artists(artist_id),
    FOREIGN KEY (album_id) REFERENCES albums(album_id)
);

CREATE INDEX idx_track_id ON song_features(track_id);
CREATE INDEX idx_popularity ON song_features(popularity);
CREATE INDEX idx_genre ON song_features(genre);
CREATE INDEX idx_artist_id ON song_features(artist_id);
CREATE INDEX idx_album_id ON song_features(album_id);
CREATE INDEX idx_genre_popularity ON song_features(genre, popularity);
CREATE INDEX idx_popularity_range ON song_features(popularity DESC);

CREATE TABLE predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    song_id INTEGER NOT NULL,
    model_version TEXT NOT NULL,
    predicted_class TEXT NOT NULL CHECK(predicted_class IN ('popular', 'unpopular')),
    probability REAL NOT NULL CHECK(probability BETWEEN 0 AND 1),
    FOREIGN KEY (song_id) REFERENCES song_features(song_id)
);

CREATE INDEX idx_pred_song ON predictions(song_id);
CREATE INDEX idx_pred_model ON predictions(model_version);

CREATE TABLE model_performance (
    performance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL UNIQUE,
    model_type TEXT NOT NULL,
    accuracy REAL NOT NULL CHECK(accuracy BETWEEN 0 AND 1),
    precision_score REAL NOT NULL CHECK(precision_score BETWEEN 0 AND 1),
    recall_score REAL NOT NULL CHECK(recall_score BETWEEN 0 AND 1),
    f1_score REAL NOT NULL CHECK(f1_score BETWEEN 0 AND 1),
    auc_score REAL NOT NULL CHECK(auc_score BETWEEN 0 AND 1),
    hyperparameters TEXT
);

CREATE INDEX idx_model_version ON model_performance(model_version);
CREATE INDEX idx_model_accuracy ON model_performance(accuracy DESC);

CREATE VIEW popular_songs AS
SELECT 
    sf.track_id,
    sf.track_name,
    a.artist_name,
    al.album_name,
    sf.genre,
    sf.popularity,
    sf.danceability,
    sf.energy,
    sf.valence
FROM song_features sf
JOIN artists a ON sf.artist_id = a.artist_id
JOIN albums al ON sf.album_id = al.album_id
WHERE sf.popularity >= 70
ORDER BY sf.popularity DESC;

CREATE VIEW genre_stats AS
SELECT 
    genre,
    COUNT(*) as song_count,
    ROUND(AVG(popularity), 2) as avg_popularity,
    ROUND(AVG(danceability), 3) as avg_danceability,
    ROUND(AVG(energy), 3) as avg_energy,
    ROUND(AVG(valence), 3) as avg_valence
FROM song_features
GROUP BY genre
ORDER BY avg_popularity DESC;

CREATE VIEW artist_stats AS
SELECT 
    a.artist_id,
    a.artist_name,
    COUNT(DISTINCT sf.song_id) as total_songs,
    COUNT(DISTINCT sf.album_id) as total_albums,
    ROUND(AVG(sf.popularity), 2) as avg_popularity,
    MAX(sf.popularity) as max_popularity,
    COUNT(CASE WHEN sf.popularity >= 70 THEN 1 END) as popular_songs
FROM artists a
LEFT JOIN song_features sf ON a.artist_id = sf.artist_id
GROUP BY a.artist_id, a.artist_name
HAVING total_songs > 0
ORDER BY avg_popularity DESC;

CREATE VIEW model_comparison AS
SELECT 
    model_version,
    model_type,
    accuracy,
    precision_score,
    recall_score,
    f1_score,
    auc_score
FROM model_performance
ORDER BY accuracy DESC, auc_score DESC;