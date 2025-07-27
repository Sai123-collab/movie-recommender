from flask import Flask, render_template, request
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# -----------------------------
# Load data when needed (lazy)
# -----------------------------
def load_data():
    movies = pd.read_csv("ml-latest-small/movies.csv")
    tags = pd.read_csv("ml-latest-small/tags.csv")

    tags_grouped = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    merged = pd.merge(movies, tags_grouped, on='movieId', how='left')
    merged['tag'] = merged['tag'].fillna('')
    merged['content'] = merged['genres'].str.replace('|', ' ') + ' ' + merged['tag']

    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(merged['content'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return merged, cosine_sim

# -----------------------------
# Get recommendations
# -----------------------------
def get_recommendations(title):
    merged, cosine_sim = load_data()
    if title not in merged['title'].values:
        return []
    idx = merged[merged['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return merged['title'].iloc[movie_indices].tolist()

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    if request.method == "POST":
        movie_name = request.form["movie"]
        recommendations = get_recommendations(movie_name)
    return render_template("index.html", recommendations=recommendations)

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
