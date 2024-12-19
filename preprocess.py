import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load the Kaggle Movies Dataset
df = pd.read_csv('dataset/movies_metadata.csv', low_memory=False)

# Clean the dataset by removing rows with missing plot descriptions and homepages
df_cleaned = df.dropna(subset=['overview', 'homepage'])
df_cleaned = df_cleaned[['title', 'overview', 'homepage']]

# Initialize TfidfVectorizer to convert plot descriptions into vectors
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
vectorizer.fit(df_cleaned['overview'])

# Save the vectorizer and cleaned dataset
joblib.dump(vectorizer, 'vectorizer.pkl')
df_cleaned.to_csv('cleaned_movies.csv', index=False)
