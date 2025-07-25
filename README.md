# 📘 Movie Recommender System – User Manual

This project is a **Content-Based Movie Recommendation System** built using **Python** and **Flask**. It allows users to input a movie title and get recommendations based on tag similarity using machine learning techniques.

---

## 📂 Project Structure

```
movie-recommender/
│
├── app.py                      # Main Flask application
├── templates/
│   └── index.html              # Frontend form for movie input
├── ml-latest-small/
│   ├── movies.csv              # Dataset with movie titles and genres
│   └── tags.csv                # User-generated tags
└── README.md                   # Project documentation (this file)
```

---

## 🔧 Setup Instructions

Follow the steps below to set up and run the project on your local machine.

### ✅ Step 1: Clone the Repository

```bash
git clone https://github.com/Sai123-collab/movie-recommender.git
cd movie-recommender
```

### ✅ Step 2: Install Required Libraries

Make sure Python is installed. Then install required packages:

```bash
pip install flask pandas scikit-learn
```

> You can also use `requirements.txt` if available:
> ```
> pip install -r requirements.txt
> ```

### ✅ Step 3: Run the Flask App

```bash
python app.py
```

After running, open your browser and go to:

```
http://127.0.0.1:5000/
```

---

## 🧠 How It Works

1. Loads movie and tag data from CSV files.
2. Combines movie tags and applies **TF-IDF vectorization**.
3. Calculates **cosine similarity** between all movies.
4. User enters a movie title on the webpage.
5. System returns a list of similar movies based on tag similarity.

---

## 🧪 Example Usage

- Enter: `The Matrix`
- Output: Recommendations like `The Matrix Reloaded`, `Inception`, `Equilibrium`, etc.

---

## 💡 Customization Tips

- To use a different dataset, replace `movies.csv` and `tags.csv` in the `ml-latest-small/` folder.
- You can modify the number of recommendations in the Python logic (`top_n`).

---

## 📸 Screenshot

> 
![Movie Recommender Screenshot](screenshot.png)

---

## 👤 Author

- **Sai Balaji**
- GitHub: [Sai123-collab](https://github.com/Sai123-collab)

---

## 📝 License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute it.

---

## 🌟 Star This Repo

If you find this project helpful, consider giving it a ⭐ on GitHub!

