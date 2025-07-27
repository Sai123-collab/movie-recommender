from flask import Flask, render_template, request
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

merged = None
cosine_sim = None

def load_data():
    global merged, cosine_sim
    if merged is None:
        movies = pd.read_csv("ml-latest-small/movies.csv")
        tags = pd.read_csv("ml-latest-small/tags.csv")
        tags_grouped = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
        merged_data = pd.merge(movies, tags_grouped, on='movieId', how='left')
        merged_data['tag'] = merged_data['tag'].fillna('')
        merged_data['content'] = merged_data['genres'].str.replace('|', ' ') + ' ' + merged_data['tag']

        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(merged_data['content'])
        cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        merged = merged_data
        cosine_sim = cosine_sim_matrix

def get_recommendations(title):
    load_data()
    if title not in merged['title'].values:
        return []
    idx = merged[merged['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return merged['title'].iloc[movie_indices].tolist()

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    if request.method == "POST":
        movie_name = request.form["movie"]
        recommendations = get_recommendations(movie_name)
    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
