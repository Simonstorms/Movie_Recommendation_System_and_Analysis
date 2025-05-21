import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import cosine
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_data():
    # Load dataset files
    movies = pd.read_csv('movie_dataset/movies.csv')
    actors = pd.read_csv('movie_dataset/actors.csv')
    genres = pd.read_csv('movie_dataset/genres.csv')
    themes = pd.read_csv('movie_dataset/themes.csv')
    crew = pd.read_csv('movie_dataset/crew.csv')
    return movies, actors, genres, themes, crew

def merge_datasets(movies, actors, genres, themes, crew):
    # Merge all datasets
    genres_grouped = genres.groupby('id')['genre'].apply(lambda x: ' '.join(x)).reset_index()
    movies = movies.merge(genres_grouped, on='id', how='left')

    themes_grouped = themes.groupby('id')['theme'].apply(lambda x: ' '.join(x)).reset_index()
    movies = movies.merge(themes_grouped, on='id', how='left')

    actors_grouped = actors.groupby('id')['name'].apply(lambda x: ' '.join(x.astype(str))).reset_index()
    movies = movies.merge(actors_grouped, on='id', how='left')

    directors = crew[crew['role'] == 'Director']
    crew_grouped = directors.groupby('id')['name'].apply(lambda x: ' '.join(x.astype(str))).reset_index()
    crew_grouped.rename(columns={'name': 'director'}, inplace=True)
    movies = movies.merge(crew_grouped, on='id', how='left')

    movies.rename(columns={'name_x': 'name', 'name_y': 'actors'}, inplace=True)

    # Fill missing values
    numeric_columns = movies.select_dtypes(include=['float64', 'int64']).columns
    string_columns = movies.select_dtypes(include=['object']).columns

    movies = movies.dropna(subset=['rating'])
    movies[numeric_columns] = movies[numeric_columns].fillna(0)
    movies[string_columns] = movies[string_columns].fillna('')

    movies['log_minute'] = np.log1p(movies['minute'])
    movies['minute_squared'] = movies['minute'] ** 2

    return movies

def add_genre_similarity_features(movies_df):
    # Use cosine similarity for genres
    genre_vectorizer = TfidfVectorizer()
    genre_vectors = genre_vectorizer.fit_transform(movies_df['genre'].fillna(''))

    # Calculate average genre vector
    avg_genre_vector = np.array(genre_vectors.mean(axis=0)).flatten()

    # Calculate cosine similarity to average genre
    similarities = []
    for i in range(genre_vectors.shape[0]):
        row_vector = genre_vectors[i].toarray().flatten()
        if np.sum(row_vector) == 0 or np.sum(avg_genre_vector) == 0:
            sim = 0.0
        else:
            try:
                sim = 1 - cosine(row_vector, avg_genre_vector)
            except:
                sim = 0.0
        similarities.append(sim)

    movies_df['genre_similarity'] = similarities
    return movies_df

def prepare_features(movies_df):
    # Apply genre similarity feature
    movies_df = add_genre_similarity_features(movies_df)

    # Apply TF-IDF to text features
    tfidf = TfidfVectorizer(
        max_features=200,
        stop_words='english',
        min_df=5,
        max_df=0.8
    )

    genre_features = tfidf.fit_transform(movies_df['genre'].fillna(''))
    theme_features = tfidf.fit_transform(movies_df['theme'].fillna(''))
    director_features = tfidf.fit_transform(movies_df['director'].fillna(''))
    actor_features = tfidf.fit_transform(movies_df['actors'].fillna('')) if 'actors' in movies_df.columns else csr_matrix((len(movies_df), 0))
    description_features = tfidf.fit_transform(movies_df['description'].fillna('')) if 'description' in movies_df.columns else csr_matrix((len(movies_df), 0))

    # SVD dimensionality reduction
    n_components_genre = min(50, genre_features.shape[1])
    n_components_theme = min(50, theme_features.shape[1])

    svd_genre = TruncatedSVD(n_components=n_components_genre) if n_components_genre > 0 else None
    svd_theme = TruncatedSVD(n_components=n_components_theme) if n_components_theme > 0 else None

    if svd_genre and n_components_genre > 0:
        genre_features_reduced = svd_genre.fit_transform(genre_features)
    else:
        genre_features_reduced = genre_features.toarray()

    if svd_theme and n_components_theme > 0:
        theme_features_reduced = svd_theme.fit_transform(theme_features)
    else:
        theme_features_reduced = theme_features.toarray()

    # Prepare numerical features
    numerical_cols = ['date', 'minute', 'log_minute', 'minute_squared', 'genre_similarity']

    # Fill NaN values with column mean or 0
    for col in numerical_cols:
        if movies_df[col].isna().any():
            fill_value = movies_df[col].mean() if movies_df[col].mean() > 0 else 0
            movies_df[col] = movies_df[col].fillna(fill_value)

    numerical_features = np.column_stack([
        movies_df[['date', 'minute']].values,
        movies_df['minute'].apply(lambda x: x / 30).values.reshape(-1, 1),  # Normalized duration
        movies_df[['log_minute', 'minute_squared', 'genre_similarity']].values
    ])

    # Apply polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    numerical_poly = poly.fit_transform(numerical_features)

    # Normalize features
    scaler = StandardScaler()
    numerical_features_scaled = scaler.fit_transform(numerical_poly)

    # Combine all features
    X = hstack([
        csr_matrix(numerical_features_scaled),
        csr_matrix(genre_features_reduced),
        csr_matrix(theme_features_reduced),
        actor_features,
        director_features,
        description_features
    ])
    y = movies_df['rating']

    # Split data with stratification based on rating bins
    rating_bins = pd.cut(y, bins=5, labels=False)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=rating_bins)

def build_model(X_train, y_train):
    gradient_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    gradient_model.fit(X_train, y_train)
    return gradient_model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 1, 5)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Calculate prediction accuracy bands
    within_half_star = np.mean(abs(y_test - y_pred) <= 0.5) * 100
    within_one_star = np.mean(abs(y_test - y_pred) <= 1.0) * 100

    print(f"Model Performance: RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    print(f"Prediction Accuracy: Within 0.5 stars: {within_half_star:.1f}%, Within 1.0 stars: {within_one_star:.1f}%")

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2,
            "within_half_star": within_half_star, "within_one_star": within_one_star}

def compare_with_original(improved_metrics):
    # Define approximate metrics from the original model
    original_metrics = {
        "mse": 0.12,
        "rmse": 0.34,
        "mae": 0.254,
        "r2": 0.34,
        "within_half_star": 85.0,
        "within_one_star": 98.0
    }

    print("\nModel Comparison - Original vs. Improved:")
    for metric in ["rmse", "mae", "r2"]:
        if metric == "r2":
            change = ((improved_metrics[metric] - original_metrics[metric]) / original_metrics[metric]) * 100
        else:
            change = ((original_metrics[metric] - improved_metrics[metric]) / original_metrics[metric]) * 100
        print(f"{metric.upper()}: Original={original_metrics[metric]:.4f}, Improved={improved_metrics[metric]:.4f}, Change={change:+.2f}%")

def main():
    print("Loading and processing data...")
    movies_df, actors_df, genres_df, themes_df, crew_df = load_data()
    movies_df = merge_datasets(movies_df, actors_df, genres_df, themes_df, crew_df)

    print(f"Preparing features for {len(movies_df)} movies...")
    X_train, X_test, y_train, y_test = prepare_features(movies_df)

    print("Building and training gradient boosting model...")
    model = build_model(X_train, y_train)

    print("\nEvaluating model performance:")
    metrics = evaluate_model(model, X_test, y_test)

    compare_with_original(metrics)

if __name__ == "__main__":
    main()
