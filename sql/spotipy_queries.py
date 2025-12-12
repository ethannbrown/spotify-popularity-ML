import sqlite3
import pandas as pd

def run_queries(db_path='spotify.db'):
    conn = sqlite3.connect(db_path)
    
    queries = {
        "Database Overview": """
            SELECT 
                (SELECT COUNT(*) FROM song_features) as total_songs,
                (SELECT COUNT(*) FROM artists) as total_artists,
                (SELECT COUNT(*) FROM albums) as total_albums,
                (SELECT COUNT(DISTINCT genre) FROM song_features) as total_genres
        """,
        
        "Top 10 Most Popular Songs": """
            SELECT 
                sf.track_name,
                a.artist_name,
                sf.popularity
            FROM song_features sf
            JOIN artists a ON sf.artist_id = a.artist_id
            ORDER BY sf.popularity DESC
            LIMIT 10
        """,
        
        "Genre Performance": """
            SELECT 
                genre,
                COUNT(*) as song_count,
                ROUND(AVG(popularity), 2) as avg_popularity
            FROM song_features
            GROUP BY genre
            HAVING song_count > 1000
            ORDER BY avg_popularity DESC
            LIMIT 10
        """,
        
        "Popular vs Unpopular": """
            SELECT 
                CASE 
                    WHEN popularity >= 50 THEN 'Popular'
                    ELSE 'Unpopular'
                END as category,
                COUNT(*) as count,
                ROUND(AVG(danceability), 3) as avg_danceability,
                ROUND(AVG(energy), 3) as avg_energy,
                ROUND(AVG(loudness), 3) as avg_loudness,
                ROUND(AVG(valence), 3) as avg_valence
            FROM song_features
            GROUP BY category
        """,
        
        "Top Artists": """
            SELECT 
                a.artist_name,
                COUNT(sf.song_id) as total_songs,
                ROUND(AVG(sf.popularity), 2) as avg_popularity
            FROM artists a
            JOIN song_features sf ON a.artist_id = sf.artist_id
            GROUP BY a.artist_id, a.artist_name
            HAVING total_songs >= 10
            ORDER BY avg_popularity DESC
            LIMIT 10
        """,
        
        "Model Performance": """
            SELECT 
                model_type,
                accuracy,
                precision_score,
                recall_score,
                f1_score,
                auc_score
            FROM model_comparison
        """
    }
    
    for name, query in queries.items():
        try:
            result = pd.read_sql(query, conn)
            print(result.to_string(index=False))
        except Exception as e:
            print(f"Error: {e}")
    
    conn.close()

if __name__ == "__main__":
    run_queries()