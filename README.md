# Movie Dataset Analysis and Movie Recommendation System
**Creator**: Simon Gneuß
**Enrollment Number**: 23.01.038

![image](https://github.com/user-attachments/assets/13b17921-540a-4f8f-a643-7c4776979b25)

## Overview
This repository showcases an exploration of a movie dataset and a machine learning-based movie recommendation system. The project includes:

1. **Movie Dataset Analysis**: Insights into genres, trends, ratings, and actors.
2. **Movie Recommendation System**: A content-based movie recommendation system that generates recommendations
    based on movie features including genres, themes, actors, and more.
3. **Score Prediction**: A model that predicts movie ratings based on features such as genres, themes, and actors.

---

## Part 1: Movie Dataset Analysis

### Introduction
This analysis explores movie trends, patterns, and relationships, laying the groundwork for a recommendation system. Key areas include **genres**, **actors**, **movie releases**, and **ratings**.

### Key Questions:
1. **What are the most popular movie genres?**
2. **What trends exist in movie releases?**
3. **How do average ratings vary by genre?**
4. **Is there a correlation between movie duration and rating?**
5. **Who are the most prolific actors, and how do their movies perform?**

### Dataset
Sourced from [Kaggle](https://www.kaggle.com/datasets/gsimonx37/letterboxd/data), the dataset includes:
- **Genres**
- **Actors**
- **Movies**
- **Themes**

---

### Some key visual examples

#### 1. Genre Distribution
![output](https://github.com/user-attachments/assets/1370deb5-7d6d-4f30-82df-59750542ae70)



#### 2. Movie Release Trends
<img width="1020" alt="Screenshot 2024-11-10 at 13 58 38" src="https://github.com/user-attachments/assets/25ac6f9b-22d5-476d-9cee-8a419649d6f2">


#### 3. Actor Analysis
<img width="986" alt="Screenshot 2024-11-10 at 13 58 23" src="https://github.com/user-attachments/assets/1d41b4bf-9250-42be-871a-4319e4e2199f">

---

### Conclusion
This analysis highlights:
- **Genre popularity** and trends.
- **Duration and ratings** correlations.
- **Top actors** and their impact.

This serves as a strong base for building a recommendation system.


---

## Part 2: Movie Recommendation System

### Overview
A content-based movie recommendation system that analyzes movie features including genres, themes, actors, and descriptions to provide personalized recommendations.

### Key Features
- Content-based filtering using multiple movie attributes
- TF-IDF vectorization for text processing
- Personalized user profiles based on ratings
- Fast recommendation generation

### Technical Implementation

#### Core Components
1. **Feature Processing**
   - Movies metadata (genres, themes, actors)
   - Text content (descriptions, taglines)
   - User ratings integration

2. **Adding your own Recommendations**
    - Add your own movies + ratings to generate personalized recommendations

```python
# Example usage
user_ratings = pd.DataFrame({
    'name': ['Dune: Part Two', 'Oppenheimer', 'Interstellar'],
    'rating': [5, 5, 5]
})
```

### Performance
- Average response time: 30 seconds
- Recommendations based on content similarity

### Setup and Usage

1. **Installation**
```bash
git clone https://github.com/yourusername/movie_recommendation_system.git
cd movie_recommendation_system
```

2. **Install Dependencies**
- pandas
- numpy
- scikit-learn
- scipy

3. **Run System**
```bash
python recommendation_system.py
```

### Future Improvements
- Use my dowloaded Letterboxd data to recommend movies for my profile
- Include more movie metadata, such as directors, writers, and studios

---

## Part 3: Score Prediction Model

### Overview
A machine learning model that predicts movie ratings based on various features including genres, themes, actors, directors, and descriptions using Random Forest Regression.

### Key Features
- Content-based prediction using multiple movie attributes
- TF-IDF vectorization for text processing
- Integration with movie metadata
- Rating predictions on a 1-5 scale

### Technical Implementation

#### Core Components
1. **Feature Processing**
   - Movies metadata (genres, themes, actors)
   - Text content (descriptions, taglines)
   - Release dates and duration
   - Numerical feature scaling

2. **Model Architecture**
   - Random Forest Regressor with:
     - 500 trees
     - Max depth of 25
     - Minimum samples split of 5
     - Minimum samples leaf of 2

### Performance
- R² Score: 0.34
- 87% of predictions within 0.5 stars
- 98.8% of predictions within 1.0 stars

### Model Insights

**Strengths**: Good at handling both numerical and categorical features

**Limitations**: Struggles with extreme ratings

### Future Improvements

1. **Model Optimization**
   - Experiment with different algorithms (XGBoost, LightGBM)
   - Fine-tune hyperparameters further

2. **Data Enhancement**
   - Add box office and budget information
   - Expand the dataset with more rated movies

3. **Data Quality**
   - Improve missing value handling
   - Balance rating distribution


---

**Note**: Explore the repository for more details, data, and code. Feel free to contribute or raise issues.

---

## License
This project is licensed under the [MIT License](LICENSE).
