import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from scipy.optimize import minimize

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
    movies.fillna('', inplace=True)
    return movies


# Build TF-IDF matrices for each feature
def build_feature_tfidf_matrix(movies, feature):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies[feature])
    return tfidf, tfidf_matrix

# Create User Profile
def create_user_profile(user_movies, tfidf_matrix_combined):
    user_profiles = tfidf_matrix_combined[user_movies['row_index']]
    user_ratings_array = user_movies['rating'].values.reshape(-1, 1)
    user_profile = np.sum(user_profiles.multiply(user_ratings_array), axis=0)
    user_profile = np.asarray(user_profile)[0]
    return user_profile / np.linalg.norm(user_profile)

# Calculate cosine similarity
def get_cosine_similarity(user_profile_normalized, tfidf_matrix_combined):
    return cosine_similarity(user_profile_normalized.reshape(1, -1), tfidf_matrix_combined)

# Filter recommendations based on user ratings
def filter_recommendations(similarity_scores, user_movies, movies_df):
    already_rated_ids = set(user_movies['id'].tolist())
    return [score for score in similarity_scores if movies_df.iloc[score[0]]['id'] not in already_rated_ids]

# Get diverse recommendations
def get_diverse_recommendations(similarity_scores, tfidf_matrix_combined, n=10, diversity_threshold=0.2):
    recommendations = []
    for score in similarity_scores:
        if len(recommendations) >= n:
            break
        if all(cosine_similarity(tfidf_matrix_combined[score[0]], tfidf_matrix_combined[rec[0]])[0][0] < 1 - diversity_threshold for rec in recommendations):
            recommendations.append(score)
    return recommendations

# Explain recommendations based on features
def explain_recommendation(movie_index, user_profile, tfidf_matrix_combined, feature_names):
    movie_vector = tfidf_matrix_combined[movie_index].toarray()[0]
    feature_importance = user_profile * movie_vector
    top_features = sorted(zip(feature_importance, feature_names), reverse=True)[:10]
    return [feature[1] for feature in top_features]

def format_explanation(explanation):
    return ', '.join([name.title() for name in explanation])

# Convert weights between array and dictionary
def weights_to_array(weights):
    return np.array([weights[feature] for feature in ['genre', 'theme', 'actors', 'description', 'tagline', 'director']])

def array_to_weights(array):
    return dict(zip(['genre', 'theme', 'actors', 'description', 'tagline', "director"], array))

# Compute loss for optimization
def compute_loss(weights_array, user_movies, tfidf_matrices):
    weights = array_to_weights(weights_array)
    weighted_tfidf_matrices = [weights[feature] * tfidf_matrices[feature][1] for feature in weights]
    tfidf_matrix_combined = hstack(weighted_tfidf_matrices)

    user_profiles = tfidf_matrix_combined[user_movies['row_index']]
    user_ratings_array = user_movies['rating'].values.reshape(-1, 1)
    user_profile = np.sum(user_profiles.multiply(user_ratings_array), axis=0)
    user_profile = np.asarray(user_profile)[0]
    user_profile_normalized = user_profile / np.linalg.norm(user_profile)

    similarities = cosine_similarity(user_profile_normalized.reshape(1, -1), tfidf_matrix_combined)[0]
    rated_movie_indices = user_movies['row_index'].tolist()
    rated_movie_similarities = similarities[rated_movie_indices]
    ratings = user_movies['rating'].values

    high_rated_similarities = rated_movie_similarities[ratings >= 4.0]
    low_rated_similarities = rated_movie_similarities[ratings <= 3.0]

    loss = -np.sum(high_rated_similarities) + np.sum(low_rated_similarities)
    return loss

# Main script
if __name__ == "__main__":
    movies_df, actors_df, genres_df, themes_df, crew_df = load_data()
    movies_df = merge_datasets(movies_df, actors_df, genres_df, themes_df, crew_df)
    movies_df.reset_index(drop=True, inplace=True)
    movies_df['row_index'] = movies_df.index

    features = ['genre', 'theme', 'actors', 'description', 'tagline', 'director']
    tfidf_matrices = {feature: build_feature_tfidf_matrix(movies_df, feature) for feature in features}

    user_ratings = pd.DataFrame({
        'name': ['Dune: Part Two', 'Oppenheimer', 'Interstellar', "Rio 2", "Noelle"],
        'rating': [5, 5, 5, 2.5, 3]
    })
    user_movies = user_ratings.merge(movies_df[['id', 'name', 'row_index']], on='name', how='left')
    user_movies['id'] = user_movies['id'].astype(int)
    user_movies['row_index'] = user_movies['row_index'].astype(int)

    initial_weights = np.ones(len(features))
    bounds = [(0, None)] * len(features)
    result = minimize(compute_loss, initial_weights, args=(user_movies, tfidf_matrices), bounds=bounds)
    optimized_weights = array_to_weights(result.x)

    weighted_tfidf_matrices = [optimized_weights[feature] * tfidf_matrices[feature][1] for feature in features]
    tfidf_matrix_combined = hstack(weighted_tfidf_matrices)

    user_profile_normalized = create_user_profile(user_movies, tfidf_matrix_combined)

    cosine_sim = get_cosine_similarity(user_profile_normalized, tfidf_matrix_combined)
    similarity_scores = sorted(enumerate(cosine_sim[0]), key=lambda x: x[1], reverse=True)

    recommendations = filter_recommendations(similarity_scores, user_movies, movies_df)
    diverse_recommendations = get_diverse_recommendations(recommendations, tfidf_matrix_combined)

    recommended_indices = [i[0] for i in diverse_recommendations]
    recommended_movies = movies_df.iloc[recommended_indices][['id', 'name', 'genre', 'date']]
    recommended_movies.reset_index(drop=True, inplace=True)

    all_feature_names = []
    for feature in features:
        feature_names = tfidf_matrices[feature][0].get_feature_names_out()
        feature_prefix = f"{feature}_"
        feature_names_prefixed = [feature_prefix + name for name in feature_names]
        all_feature_names.extend(feature_names_prefixed)

    recommended_movies['explanation'] = recommended_movies.index.map(
        lambda x: explain_recommendation(x, user_profile_normalized, tfidf_matrix_combined, all_feature_names)
    )
    recommended_movies['Explanation'] = recommended_movies['explanation'].apply(format_explanation)

    print(recommended_movies[['name', 'Explanation']])
