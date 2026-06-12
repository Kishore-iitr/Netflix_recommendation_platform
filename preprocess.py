import os
import pandas as pd
import random

def preprocess_data(input_file, output_file, sample_prob=0.1, min_ratings_per_user=10):
    """
    Parses the Netflix combined_data_1.txt and extracts ratings.
    To ensure variety and proper distribution, we randomly sample movies across the entire file
    with probability `sample_prob` (e.g., 0.1 means ~450 movies scattered throughout).
    """
    print(f"Parsing {input_file} with movie sampling probability {sample_prob}...")
    
    data = []
    current_movie = None
    keep_current_movie = False
    
    random.seed(42) # For reproducibility
    
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.endswith(':'):
                current_movie = line[:-1]
                # Randomly decide whether to include this movie to ensure variety
                keep_current_movie = random.random() < sample_prob
            else:
                if keep_current_movie and current_movie is not None:
                    # Line format: CustomerID,Rating,Date
                    parts = line.split(',')
                    if len(parts) == 3:
                        user_id, rating, date = parts
                        data.append([user_id, current_movie, rating, date])
                        
    df = pd.DataFrame(data, columns=['user_id', 'movie_id', 'rating', 'date'])
    df['rating'] = df['rating'].astype(float)
    
    print(f"Extracted {len(df)} ratings from a diverse set of movies.")
    
    # Filter out users with very few ratings within this subset to increase density
    user_counts = df['user_id'].value_counts()
    active_users = user_counts[user_counts >= min_ratings_per_user].index
    df_filtered = df[df['user_id'].isin(active_users)]
    
    print(f"Filtered to {len(df_filtered)} ratings from users with >= {min_ratings_per_user} ratings.")
    
    df_filtered.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    input_file = "data/combined_data_1.txt"
    output_file = "data/sampled_ratings.csv"
    if not os.path.exists("data"):
        os.makedirs("data")
    
    if os.path.exists(input_file):
        preprocess_data(input_file, output_file, sample_prob=0.08, min_ratings_per_user=8)
    else:
        print(f"Error: {input_file} not found. Ensure dataset is placed correctly.")
