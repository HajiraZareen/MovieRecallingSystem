from flask import Flask, request, render_template
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the vectorizer and dataset
vectorizer = joblib.load('vectorizer.pkl')

# Load the cleaned dataset
df_cleaned = pd.read_csv('cleaned_movies.csv')

def recommend_movie(plot_description, vectorizer, df):
    # Transform the input plot description
    query_vector = vectorizer.transform([plot_description])
    # Transform all movie overviews
    movie_vectors = vectorizer.transform(df['overview'])
    # Calculate cosine similarity
    similarities = cosine_similarity(query_vector, movie_vectors).flatten()
    # Get indices of top 5 most similar movies
    top_indices = similarities.argsort()[-5:][::-1]
    recommended_movies = df.iloc[top_indices]
    
    # Prepare the movie list with title, homepage, and poster URL
    movies = [
        {
            'title': row['title'],
            'homepage': row['homepage'] if pd.notna(row['homepage']) else '#',  # Default to '#' if no homepage
            'poster': row.get('poster_path', 'https://via.placeholder.com/70x100')  # Replace with actual column
        }
        for _, row in recommended_movies.iterrows()
    ]
    return movies

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get the plot description from the form
    plot_description = request.form['plot']
    # Get recommendations
    recommended_movies = recommend_movie(plot_description, vectorizer, df_cleaned)
    # Render the result page with recommendations
    return render_template('result.html', movies=recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)
