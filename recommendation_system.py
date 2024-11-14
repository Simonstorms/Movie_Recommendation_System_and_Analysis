from typing import List, Dict, Set, Tuple, Optional, Union
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

class MovieRecommender:
    """
    A content-based movie recommendation system that generates recommendations
    based on movie features including genres, themes, actors, and more.
    """

    # Constants for ratings
    MIN_RATING: float = 2
    RATING_SCALE: float = 2

    def __init__(self) -> None:
        self.movies_df: Optional[pd.DataFrame] = None
        self.tfidf_matrices: Dict[str, csr_matrix] = {}
        self.features: List[str] = ['genre', 'theme', 'actors', 'description', 'tagline', 'director']

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all necessary datasets"""
        return (
            pd.read_csv('movie_dataset/movies.csv'),
            pd.read_csv('movie_dataset/actors.csv'),
            pd.read_csv('movie_dataset/genres.csv'),
            pd.read_csv('movie_dataset/themes.csv'),
            pd.read_csv('movie_dataset/crew.csv'),
        )

    def merge_datasets(
        self,
        movies: pd.DataFrame,
        actors: pd.DataFrame,
        genres: pd.DataFrame,
        themes: pd.DataFrame,
        crew: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge and preprocess all datasets"""
        movies = self._merge_genres(movies, genres)
        movies = self._merge_themes(movies, themes)
        movies = self._merge_actors(movies, actors)
        movies = self._merge_directors(movies, crew)
        movies = self._clean_movies(movies)
        return movies

    def _merge_genres(self, movies: pd.DataFrame, genres: pd.DataFrame) -> pd.DataFrame:
        """Merge genres into movies DataFrame"""
        genres_grouped = genres.groupby('id')['genre'].apply(' '.join).reset_index()
        return movies.merge(genres_grouped, on='id', how='left')

    def _merge_themes(self, movies: pd.DataFrame, themes: pd.DataFrame) -> pd.DataFrame:
        """Merge themes into movies DataFrame"""
        themes_grouped = themes.groupby('id')['theme'].apply(' '.join).reset_index()
        return movies.merge(themes_grouped, on='id', how='left')

    def _merge_actors(self, movies: pd.DataFrame, actors: pd.DataFrame) -> pd.DataFrame:
        """Merge actors into movies DataFrame"""
        actors_grouped = actors.groupby('id')['name'].apply(lambda x: ' '.join(x.astype(str))).reset_index()
        return movies.merge(actors_grouped, on='id', how='left')

    def _merge_directors(self, movies: pd.DataFrame, crew: pd.DataFrame) -> pd.DataFrame:
        """Merge directors into movies DataFrame"""
        directors = crew[crew['role'] == 'Director']
        crew_grouped = directors.groupby('id')['name'].apply(lambda x: ' '.join(x.astype(str))).reset_index()
        crew_grouped.rename(columns={'name': 'director'}, inplace=True)
        return movies.merge(crew_grouped, on='id', how='left')

    def _clean_movies(self, movies: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the movies DataFrame"""
        movies.rename(columns={'name_x': 'name', 'name_y': 'actors'}, inplace=True)
        movies = movies.dropna(subset=['rating'])
        numeric_columns = movies.select_dtypes(include=['float64', 'int64']).columns
        string_columns = movies.select_dtypes(include=['object']).columns
        movies.loc[:, numeric_columns] = movies[numeric_columns].fillna(0)
        movies.loc[:, string_columns] = movies[string_columns].fillna('')
        return movies

    def build_feature_matrices(self) -> None:
        """Build TF-IDF matrices for all features"""
        for feature in self.features:
            tfidf = TfidfVectorizer(stop_words='english' if feature in ['tagline', 'description'] else None)
            self.tfidf_matrices[feature] = tfidf.fit_transform(self.movies_df[feature].fillna(''))

    def create_user_profile(self, user_movies: pd.DataFrame) -> np.ndarray:
        """Create a user profile based on ratings and movie features"""
        weighted_profiles = self._calculate_weighted_profiles(user_movies)
        return self._normalize_profile(weighted_profiles)

    def _calculate_weighted_profiles(self, user_movies: pd.DataFrame) -> csr_matrix:
        """Calculate weighted profiles for user movies"""
        return hstack([
            self.tfidf_matrices[feature][user_movies['row_index']]
            for feature in self.features
        ]).multiply(((user_movies['rating'] - self.MIN_RATING) / self.RATING_SCALE).values.reshape(-1, 1))

    def _normalize_profile(self, weighted_profiles: csr_matrix) -> np.ndarray:
        """Normalize the user profile"""
        return np.asarray(weighted_profiles.sum(axis=0).flatten() /
                         np.linalg.norm(weighted_profiles.sum(axis=0).flatten()))[0]

    def get_recommendations(self, user_ratings: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Generate movie recommendations based on user ratings"""
        user_movies = user_ratings.merge(self.movies_df[['id', 'name']], on='name', how='left')
        user_movies['row_index'] = user_movies['name'].apply(
            lambda x: self.movies_df[self.movies_df['name'] == x].index[0]
        )

        user_profile = self.create_user_profile(user_movies)
        tfidf_matrix_combined = hstack([self.tfidf_matrices[feature] for feature in self.features])
        similarity_scores = cosine_similarity(
            user_profile.reshape(1, -1),
            tfidf_matrix_combined
        ).flatten()

        already_rated_ids: Set[int] = set(user_movies['id'])
        top_indices = [
            i for i, score in sorted(enumerate(similarity_scores), key=lambda x: x[1], reverse=True)
            if self.movies_df.iloc[i]['id'] not in already_rated_ids
        ][:n]

        return self.movies_df.iloc[top_indices][['id', 'name', 'genre', 'rating']]

    def fit(self) -> 'MovieRecommender':
        """Initialize and prepare the recommendation system"""
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
