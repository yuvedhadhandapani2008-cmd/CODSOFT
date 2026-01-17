# Content-Based Recommendation System (Task 4)

## Project Overview
This project implements a **simple content-based recommendation system** using Python.  
The system suggests items to users based on their preferences by analyzing similarities between item features.

For demonstration purposes, a **movie recommendation system** is implemented.  
The same approach can be applied to **books, products, courses, or music** by modifying the dataset.

## Objective
The main objective of this project is to:
- Build a recommendation system
- Suggest items based on user preferences
- Use content-based filtering techniques

## Recommendation Technique Used
### Content-Based Filtering
- Uses item features (movie genres)
- Recommends similar items based on user choice
- Does not require data from other users

## Technologies Used
- Python 3
- Pandas
- Scikit-learn
- Visual Studio Code

## Dataset Description
A small in-memory dataset is used consisting of:
- Movie Title
- Movie Genre

### Example:
- Titanic → Romance, Drama  
- Inception → Action, Sci-Fi  

## Working of the System
1. Movie genres are converted into numerical form using **TF-IDF Vectorizer**
2. **Cosine similarity** is calculated between all movies
3. When the user enters a movie name:
   - The system finds similar movies based on genre
   - Top 3 similar movies are recommended

## How to Run the Program

### Step 1: Install Required Libraries

pip install pandas scikit-learn

### Step 2: Run the Python File

python recommendation_system.py

## Sample Output
Available Movies:
The Matrix
Titanic
Avengers Endgame
The Notebook
Inception
Interstellar

Enter a movie you like: Titanic

Recommended Movies:
['The Notebook', 'Interstellar', 'The Matrix']

# Execution Video
Watch the execution of the tic_tac_toe AI here:
Click here to view execution video : https://drive.google.com/file/d/14dljAqoxVg1xYHp_QFDlnHoc_RYp45AG/view?usp=sharing

## Author
Yuvedha Dhandapani