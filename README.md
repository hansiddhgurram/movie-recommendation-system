# 🎬 Movie Recommendation System

An **end-to-end Movie Recommendation System** built using classical
**Machine Learning techniques** and deployed as an interactive
**Streamlit web application**.

The project demonstrates how recommendation systems evolve from
simple baselines to advanced latent-factor models.

---

## 🚀 Features

This project implements and compares multiple recommendation strategies:

### 1️⃣ Popularity-Based Recommendation
- Recommends movies based on:
  - Average rating
  - Number of ratings
- Serves as a **baseline model**

---

### 2️⃣ Content-Based Filtering
- Uses movie **genre features**
- Represents each movie as a feature vector
- Computes **cosine similarity** between movies
- Recommends movies similar to a selected movie

---

### 3️⃣ Collaborative Filtering (User-Based)
- Builds a **user–item rating matrix**
- Computes similarity between users
- Recommends movies liked by similar users
- Captures collective user behavior

---

### 4️⃣ Matrix Factorization (SVD)
- Applies **Truncated SVD** to the user–item matrix
- Learns **latent user and movie factors**
- Predicts missing ratings
- Produces more personalized recommendations
- Handles sparse data effectively

This is a core industry technique used by platforms like **Netflix and Spotify**.

---

### 5️⃣ Interactive Streamlit Application
- Clean, user-friendly web interface
- Choose between multiple recommendation methods
- Real-time recommendations
- Adjustable SVD latent factors
- End-to-end ML deployment

---

## 📊 Dataset

This project uses the **MovieLens 100K dataset** provided by GroupLens.

- Users: 943
- Movies: 1,682
- Ratings: 100,000

🔗 Dataset link:  
https://grouplens.org/datasets/movielens/100k/

After downloading, place the following files in:
data/raw/
├── u.data
├── u.item
└── u.user

> ⚠️ Dataset files are intentionally excluded from this repository.

---

## 📂 Project Structure

movie-recommendation-system/
├── data/
│ ├── raw/
│ └── processed/
├── notebooks/
│ ├── 01_exploration.ipynb
│ ├── 02_content_based.ipynb
│ ├── 03_collaborative.ipynb
│ └── 04_matrix_factorization.ipynb
├── src/
│ ├── popularity.py
│ ├── content_based.py
│ ├── collaborative.py
│ ├── matrix_factorization.py
│ └── evaluation.py
├── app.py
├── README.md
├── requirements.txt
└── .gitignore

---

## 🛠 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Jupyter Notebooks (via VS Code)

---

## ▶️ How to Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/hansiddhgurram/movie-recommendation-system.git
cd movie-recommendation-system
