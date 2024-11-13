import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


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
    
    # Fix the fillna warning by handling numeric and string columns separately
    numeric_columns = movies.select_dtypes(include=['float64', 'int64']).columns
    string_columns = movies.select_dtypes(include=['object']).columns

    movies = movies.dropna(subset=['rating'])
    # Fill NA values appropriately based on dtype
    movies[numeric_columns] = movies[numeric_columns].fillna(0)
    movies[string_columns] = movies[string_columns].fillna('')

    return movies

def prepare_features(movies_df):
    
    # Print rating statistics
    print("\nRating Statistics:")
    print("-" * 30)
    print(f"Mean Rating:    {movies_df['rating'].mean():.2f}")
    print(f"Median Rating:  {movies_df['rating'].median():.2f}")
    print(f"Std Deviation:  {movies_df['rating'].std():.2f}")
    print(f"Min Rating:     {movies_df['rating'].min():.2f}")
    print(f"Max Rating:     {movies_df['rating'].max():.2f}")
    
    # Enhanced TF-IDF vectorization with more features
    tfidf = TfidfVectorizer(
        max_features=500,  # Increased from 300
        stop_words='english',
        min_df=2,  # Reduced from 3 to capture more rare terms
        max_df=0.9,  # Increased slightly
        ngram_range=(1, 3)  # Added trigrams
    )
    
    # Separate text features for better control
    genre_features = tfidf.fit_transform(movies_df['genre'].astype(str))
    theme_features = tfidf.fit_transform(movies_df['theme'].astype(str))
    actor_features = tfidf.fit_transform(movies_df['actors'].astype(str))
    director_features = tfidf.fit_transform(movies_df['director'].astype(str))
    description_features = tfidf.fit_transform(movies_df['description'].astype(str))
    
    # Add more numerical features
    numerical_features = np.column_stack([
        movies_df[['date', 'minute']].values,
        movies_df['minute'].apply(lambda x: x/30).values.reshape(-1, 1),  # Normalized duration
    ])
    
    scaler = StandardScaler()
    numerical_features_scaled = scaler.fit_transform(numerical_features)
    
    # Combine all features
    X = hstack([
        csr_matrix(numerical_features_scaled),
        genre_features,
        theme_features,
        actor_features,
        director_features,
        description_features
    ])
    y = movies_df['rating']
    
    return train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Add these evaluation metrics
def evaluate_predictions(y_true, y_pred):
    # Calculate metrics
    within_half_star = np.mean(abs(y_true - y_pred) <= 0.5) * 100
    within_one_star = np.mean(abs(y_true - y_pred) <= 1.0) * 100
    
    print("\nPrediction Accuracy:")
    print("-" * 30)
    print(f"✓ Within 0.5 stars: {within_half_star:.1f}%")
    print(f"✓ Within 1.0 stars: {within_one_star:.1f}%")
    
    # Create confusion matrix heatmap
    y_pred_rounded = np.round(y_pred)
    y_true_rounded = np.round(y_true)
    cm = confusion_matrix(y_true_rounded, y_pred_rounded)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=range(1, 6),
                yticklabels=range(1, 6))
    plt.title('Prediction Confusion Matrix')
    plt.xlabel('Predicted Rating')
    plt.ylabel('True Rating')
    plt.show()

# Main script
if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    
    # Load and merge data
    movies_df, actors_df, genres_df, themes_df, crew_df = load_data()
    movies_df = merge_datasets(movies_df, actors_df, genres_df, themes_df, crew_df)
    
    # Prepare features and split data
    X_train, X_test, y_train, y_test = prepare_features(movies_df)
    
    # Enhanced Random Forest with better hyperparameters
    rf_model = RandomForestRegressor(
        n_estimators=500,  
        max_depth=25,      
        min_samples_split=5,  
        min_samples_leaf=2, 
        max_features='sqrt',
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )
    
    # Add cross-validation
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
    print("\nCross-validation scores:", cv_scores)
    print("Average CV score:", cv_scores.mean())
    
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # After making predictions, clip them to valid range
    y_pred = np.clip(y_pred, 1, 5)
    
    # Calculate and print evaluation metrics
    print("\nModel Performance Metrics:")
    print("-" * 30)
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    
    # Format feature importance better
    feature_importance = pd.DataFrame({
        'Feature': range(len(rf_model.feature_importances_)),
        'Importance': rf_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values(
        by='Importance', ascending=False
    ).head(10)
    
    print("\nTop 10 Most Important Features:")
    print("-" * 30)
    for idx, row in feature_importance.iterrows():
        print(f"Feature {int(row['Feature']):3}: {row['Importance']:.4f}")
    
    # Call after predictions
    evaluate_predictions(y_test, y_pred)

