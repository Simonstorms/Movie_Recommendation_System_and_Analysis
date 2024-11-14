import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from scipy.optimize import minimize

class MovieRecommender:
    """
    A content-based movie recommendation system that generates recommendations
    based on movie features including genres, themes, actors, and more.
    """

    # Constants for ratings
    MIN_RATING = 2
    RATING_SCALE = 2

    def __init__(self):
        self.movies_df = None
        self.tfidf_matrices = {}
        self.features = ['genre', 'theme', 'actors', 'description', 'tagline', 'director']
        self.feature_weights = {feature: 1.0 for feature in self.features}

    def load_data(self):
        """Load all necessary datasets"""
        return (
            pd.read_csv('movie_dataset/movies.csv'),
            pd.read_csv('movie_dataset/actors.csv'),
            pd.read_csv('movie_dataset/genres.csv'),
            pd.read_csv('movie_dataset/themes.csv'),
            pd.read_csv('movie_dataset/crew.csv'),
        )

    def merge_datasets(self, movies, actors, genres, themes, crew):
        """Merge and preprocess all datasets"""
        movies = self._merge_genres(movies, genres)
        movies = self._merge_themes(movies, themes)
        movies = self._merge_actors(movies, actors)
        movies = self._merge_directors(movies, crew)
        movies = self._clean_movies(movies)
        return movies

    def _merge_genres(self, movies, genres):
        """Merge genres into movies DataFrame"""
        genres_grouped = genres.groupby('id')['genre'].apply(' '.join).reset_index()
        return movies.merge(genres_grouped, on='id', how='left')

    def _merge_themes(self, movies, themes):
        """Merge themes into movies DataFrame"""
        themes_grouped = themes.groupby('id')['theme'].apply(' '.join).reset_index()
        return movies.merge(themes_grouped, on='id', how='left')

    def _merge_actors(self, movies, actors):
        """Merge actors into movies DataFrame"""
        actors_grouped = actors.groupby('id')['name'].apply(lambda x: ' '.join(x.astype(str))).reset_index()
        return movies.merge(actors_grouped, on='id', how='left')

    def _merge_directors(self, movies, crew):
        """Merge directors into movies DataFrame"""
        directors = crew[crew['role'] == 'Director']
        crew_grouped = directors.groupby('id')['name'].apply(lambda x: ' '.join(x.astype(str))).reset_index()
        crew_grouped.rename(columns={'name': 'director'}, inplace=True)
        return movies.merge(crew_grouped, on='id', how='left')

    def _clean_movies(self, movies):
        """Clean and preprocess the movies DataFrame"""
        movies.rename(columns={'name_x': 'name', 'name_y': 'actors'}, inplace=True)
        movies = movies.dropna(subset=['rating'])
        numeric_columns = movies.select_dtypes(include=['float64', 'int64']).columns
        string_columns = movies.select_dtypes(include=['object']).columns
        movies.loc[:, numeric_columns] = movies[numeric_columns].fillna(0)
        movies.loc[:, string_columns] = movies[string_columns].fillna('')
        return movies

    def build_feature_matrices(self):
        """Build TF-IDF matrices for all features"""
        for feature in self.features:
            tfidf = TfidfVectorizer(stop_words='english' if feature in ['tagline', 'description'] else None)
            self.tfidf_matrices[feature] = tfidf.fit_transform(self.movies_df[feature].fillna(''))

    def create_user_profile(self, user_movies):
        """Create a user profile based on ratings and movie features"""
        weighted_profiles = self._calculate_weighted_profiles(user_movies)
        return self._normalize_profile(weighted_profiles)

    def _calculate_weighted_profiles(self, user_movies):
        """Calculate weighted profiles for user movies"""
        feature_matrices = [
            self.tfidf_matrices[feature][user_movies['row_index']].multiply(self.feature_weights[feature])
            for feature in self.features
        ]
        weighted_profiles = hstack(feature_matrices)
        adjusted_ratings = ((user_movies['rating'] - self.MIN_RATING) / self.RATING_SCALE).values.reshape(-1, 1)
        weighted_profiles = weighted_profiles.multiply(adjusted_ratings)
        return weighted_profiles

    def _normalize_profile(self, weighted_profiles):
        """Normalize the user profile"""
        profile_sum = weighted_profiles.sum(axis=0)
        norm = np.linalg.norm(profile_sum)
        if norm == 0:
            return np.zeros(profile_sum.shape[1])
        return np.asarray(profile_sum / norm)[0]

    def _error_function(self, weights, user_movies):
        """Compute the error between predicted scores and actual ratings"""
        self.feature_weights = dict(zip(self.features, weights))

        errors = []
        for idx, row in user_movies.iterrows():
            other_movies = user_movies[user_movies.index != idx]
            if other_movies.empty:
                continue

            user_profile = self.create_user_profile(other_movies)

            feature_vectors = [
                self.tfidf_matrices[feature][row['row_index']].multiply(self.feature_weights[feature])
                for feature in self.features
            ]
            movie_vector = hstack(feature_vectors)
            movie_vector = movie_vector.toarray()

            similarity = cosine_similarity(user_profile.reshape(1, -1), movie_vector).flatten()[0]

            predicted_rating = similarity * self.RATING_SCALE + self.MIN_RATING

            actual_rating = row['rating']
            error = (predicted_rating - actual_rating) ** 2
            errors.append(error)

        return sum(errors)

    def optimize_feature_weights(self, user_movies):
        """Optimize feature weights to minimize prediction error"""
        initial_weights = [1.0] * len(self.features)
        bounds = [(0, None)] * len(self.features)

        result = minimize(
            self._error_function,
            initial_weights,
            args=(user_movies,),
            bounds=bounds,
            method='L-BFGS-B'
        )

        self.feature_weights = dict(zip(self.features, result.x))
        print("Optimized feature weights:", self.feature_weights)

    def get_recommendations(self, user_ratings, n=10):
        """Generate movie recommendations based on user ratings"""
        # Prepare user movies data
        user_movies = user_ratings.merge(self.movies_df[['id', 'name']], on='name', how='left')
        user_movies['row_index'] = user_movies['name'].apply(
            lambda x: self.movies_df[self.movies_df['name'] == x].index[0]
        )

        # Optimize feature weights
        self.optimize_feature_weights(user_movies)

        # Generate recommendations
        user_profile = self.create_user_profile(user_movies)
        tfidf_matrix_combined = hstack([
            self.tfidf_matrices[feature].multiply(self.feature_weights[feature])
            for feature in self.features
        ])
        similarity_scores = cosine_similarity(
            user_profile.reshape(1, -1),
            tfidf_matrix_combined
        ).flatten()

        # Get top recommendations
        already_rated_ids = set(user_movies['id'])
        top_indices = [
            i for i, score in sorted(enumerate(similarity_scores), key=lambda x: x[1], reverse=True)
            if self.movies_df.iloc[i]['id'] not in already_rated_ids
        ][:n]

        return self.movies_df.iloc[top_indices][['id', 'name', 'genre', 'rating']]

    def fit(self):
        """Initialize and prepare the recommendation system"""
        # Load and prepare data
        movies_df, actors_df, genres_df, themes_df, crew_df = self.load_data()
        self.movies_df = self.merge_datasets(movies_df, actors_df, genres_df, themes_df, crew_df)
        self.movies_df = self.movies_df.reset_index(drop=True)

        self.build_feature_matrices()
        return self

if __name__ == "__main__":
    recommender = MovieRecommender().fit()

    # Example user ratings
    user_ratings = pd.DataFrame({
        'name': ['Dune: Part Two', 'Oppenheimer', 'Interstellar', "Rio 2", "Noelle"],
        'rating': [5, 5, 5, 2.5, 3]
    })

    recommendations = recommender.get_recommendations(user_ratings)
    print("\nTop Recommendations:")
    print(recommendations)
