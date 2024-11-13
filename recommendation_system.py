import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

"""
This is a contend based recomandation system for movies. i didnt implent a colaborative filtering recomandation system because i want to make it only based of one user data.
I know that this approach is not as good as a combination of both but it is a first step. 

"""


# Load the data
def load_data():
    movies = pd.read_csv('movie_dataset/movies.csv')
    actors = pd.read_csv('movie_dataset/actors.csv')
    genres = pd.read_csv('movie_dataset/genres.csv')
    themes = pd.read_csv('movie_dataset/themes.csv')
    crew = pd.read_csv('movie_dataset/crew.csv')
    return movies, actors, genres, themes, crew

# Data Preprocessing
def merge_datasets(movies, actors, genres, themes, crew):
    genres_grouped = genres.groupby('id')['genre'].apply(lambda x: ' '.join(x)).reset_index()
    movies = movies.merge(genres_grouped, on='id', how='left')

    themes_grouped = themes.groupby('id')['theme'].apply(lambda x: ' '.join(x)).reset_index()
    movies = movies.merge(themes_grouped, on='id', how='left')

    actors_grouped = actors.groupby('id')['name'].apply(lambda x: ' '.join(x.astype(str))).reset_index()
    movies = movies.merge(actors_grouped, on='id', how='left')

    # Filter crew to include only directors and merge
    directors = crew[crew['role'] == 'Director']
    crew_grouped = directors.groupby('id')['name'].apply(lambda x: ' '.join(x.astype(str))).reset_index()
    crew_grouped.rename(columns={'name': 'director'}, inplace=True)  # Ensure 'director' column is named correctly
    movies = movies.merge(crew_grouped, on='id', how='left')

    # Ensure column renaming doesn't remove needed columns
    movies.rename(columns={'name_x': 'name', 'name_y': 'actors'}, inplace=True)
    
    # Drop movies without ratings before handling NA values
    movies = movies.dropna(subset=['rating'])
    
    # Fix the fillna warning by handling numeric and string columns separately
    numeric_columns = movies.select_dtypes(include=['float64', 'int64']).columns
    string_columns = movies.select_dtypes(include=['object']).columns

    # Fill NA values appropriately based on dtype
    movies[numeric_columns] = movies[numeric_columns].fillna(0)
    movies[string_columns] = movies[string_columns].fillna('')

    return movies

# Build TF-IDF matrices for each feature
def build_feature_tfidf_matrix(movies, feature):
    tfidf = TfidfVectorizer(stop_words='english' if feature in ['tagline', 'description'] else None)
    return tfidf.fit_transform(movies[feature].fillna(''))

# Create User Profile
def create_user_profile(user_movies, tfidf_matrices, features):
    weighted_profiles = hstack([tfidf_matrices[feature][user_movies['row_index']] for feature in features]).multiply(
        ((user_movies['rating'] - 2.5) / 2.5).values.reshape(-1, 1))
    return weighted_profiles.sum(axis=0).flatten() / np.linalg.norm(weighted_profiles.sum(axis=0).flatten())

# Calculate cosine similarity
def get_cosine_similarity(user_profile, tfidf_matrix_combined):
    return cosine_similarity(user_profile.reshape(1, -1), tfidf_matrix_combined).flatten()

# Get top recommendations
def get_top_recommendations(similarity_scores, user_movies, movies_df, n=10):
    already_rated_ids = set(user_movies['id'])
    top_indices = [i for i, score in sorted(enumerate(similarity_scores), key=lambda x: x[1], reverse=True)
                   if movies_df.iloc[i]['id'] not in already_rated_ids][:n]
    return movies_df.iloc[top_indices][['id', 'name', 'genre', 'rating']]

# Main script
if __name__ == "__main__":
    movies_df, actors_df, genres_df, themes_df, crew_df = load_data()
    movies_df = merge_datasets(movies_df, actors_df, genres_df, themes_df, crew_df)
    # Reset index to make sure we have consecutive indices
    movies_df = movies_df.reset_index(drop=True)
    
    features = ['genre', 'theme', 'actors', 'description', 'tagline', 'director']
    tfidf_matrices = {feature: build_feature_tfidf_matrix(movies_df, feature) for feature in features}
    
    user_ratings = pd.DataFrame({
        'name': ['Dune: Part Two', 'Oppenheimer', 'Interstellar', "Rio 2", "Noelle"],
        'rating': [5, 5, 5, 2.5, 3]
    })
    
    # Merge and get indices in a more reliable way
    user_movies = user_ratings.merge(movies_df[['id', 'name']], on='name', how='left')
    # Find indices using boolean indexing
    user_movies['row_index'] = user_movies['name'].apply(lambda x: movies_df[movies_df['name'] == x].index[0])

    user_profile = create_user_profile(user_movies, tfidf_matrices, features)
    tfidf_matrix_combined = hstack([tfidf_matrices[feature] for feature in features])
    similarity_scores = get_cosine_similarity(user_profile, tfidf_matrix_combined)

    recommendations = get_top_recommendations(similarity_scores, user_movies, movies_df)
    print("\nTop Recommendations:")
    print(recommendations)
