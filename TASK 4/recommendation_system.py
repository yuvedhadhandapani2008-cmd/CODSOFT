# -----------------------------------------
# SIMPLE CONTENT-BASED RECOMMENDATION SYSTEM
# -----------------------------------------

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Step 1: Create Movie Dataset
# -------------------------------
data = {
    'title': [
        'The Matrix',
        'Titanic',
        'Avengers Endgame',
        'The Notebook',
        'Inception',
        'Interstellar'
    ],
    'genre': [
        'action science fiction futuristic technology',
        'romance drama emotional love story',
        'action adventure superhero fantasy',
        'romance drama emotional love story',
        'action science fiction mind bending thriller',
        'science fiction space drama emotional'
    ]
}

movies = pd.DataFrame(data)

# Create a lowercase version of titles for matching
movies['title_lower'] = movies['title'].str.lower()

# -------------------------------
# Step 2: Convert Genre to Numbers
# -------------------------------
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies['genre'])

# -------------------------------
# Step 3: Calculate Similarity
# -------------------------------
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# -------------------------------
# Step 4: Recommendation Function
# -------------------------------
def recommend_movie(movie_name):
    movie_name = movie_name.lower()  # convert user input to lowercase

    if movie_name not in movies['title_lower'].values:
        return "Movie not found in database."

    index = movies[movies['title_lower'] == movie_name].index[0]
    similarity_scores = list(enumerate(cosine_sim[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommended_movies = []
    for i in similarity_scores[1:4]:
        recommended_movies.append(movies.iloc[i[0]]['title'])

    return recommended_movies

# -------------------------------
# Step 5: User Input
# -------------------------------
print("Available Movies:")
print(movies['title'].to_string(index=False))

user_movie = input("\nEnter a movie you like: ")

recommendations = recommend_movie(user_movie)

print("\nRecommended Movies:")
print(recommendations)