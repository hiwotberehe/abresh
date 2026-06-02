
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

warnings.filterwarnings('ignore')

# --- Global Variables and Load Models/Data ---

# Load datasets
try:
    books_df  = pd.read_csv('books_dataset.csv')
    users_df  = pd.read_csv('users_dataset.csv')
    loans_df  = pd.read_csv('loans_dataset.csv')
    trains_df = pd.read_csv('trains_dataset.csv')
except FileNotFoundError:
    print("Error: Ensure all dataset CSVs (books_dataset.csv, users_dataset.csv, loans_dataset.csv, trains_dataset.csv) are in the same directory.")
    exit()

# Load models and scalers
try:
    tfidf             = joblib.load('tfidf_recommender.pkl')
    cosine_sim        = np.load('cosine_sim_matrix.npy')
    overdue_model     = joblib.load('overdue_model.pkl')
    overdue_scaler    = joblib.load('overdue_scaler.pkl')
    train_delay_model = joblib.load('train_delay_model.pkl')
    train_scaler      = joblib.load('train_scaler.pkl')

    # Re-initialize LabelEncoders (assuming original fit on full data)
    le_genre = LabelEncoder()
    le_genre.fit(loans_df['genre'].fillna('Unknown').unique())

    le_weather = LabelEncoder()
    le_weather.fit(trains_df['weather'].unique())

    le_station = LabelEncoder()
    all_stations = list(set(trains_df['origin'].tolist() + trains_df['destination'].tolist()))
    le_station.fit(all_stations)

except FileNotFoundError:
    print("Error: Ensure all model files (.pkl, .npy) are in the same directory.")
    exit()

# Book Recommendation Engine (from STEP 4)
books_df['content'] = (
    books_df['title'] + ' ' +
    books_df['author'] + ' ' +
    books_df['genre'] + ' ' +
    books_df['description']
)

def recommend_books(title, top_n=5):
    matches = books_df[books_df['title'].str.contains(title, case=False, na=False)]
    if matches.empty:
        return f'Book "{title}" not found in catalog.'
    idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:top_n]
    rec_indices = [s[0] for s in sim_scores]
    result = books_df.iloc[rec_indices][['title','author','genre','rating']].copy()
    result['similarity'] = [round(s[1], 3) for s in sim_scores]
    return result

# NLP Search (from STEP 5)
def nlp_search(query, top_n=5):
    query_vec  = tfidf.transform([query])
    scores     = cosine_similarity(query_vec, cosine_sim).flatten()
    top_idx    = scores.argsort()[::-1][:top_n]
    results    = books_df.iloc[top_idx][['title','author','genre','description','rating','available']].copy()
    results['score'] = scores[top_idx].round(3)
    results = results[results['score'] > 0.01]
    return results

# Overdue Return Prediction (from STEP 6)
OVERDUE_FEATURES = ['user_age','due_days','total_borrowed','genre_enc',
                    'genre_match','borrow_month','borrow_wday','rating']

def predict_late_return(user_age, due_days, total_borrowed, genre,
                        genre_match, borrow_month, borrow_wday, rating):
    try:
        genre_enc = le_genre.transform([genre])[0]
    except ValueError:
        genre_enc = 0 # Default if genre not in training data
    inp = np.array([[user_age, due_days, total_borrowed, genre_enc,
                     genre_match, borrow_month, borrow_wday, rating]])
    inp_s = overdue_scaler.transform(inp)
    prob  = overdue_model.predict_proba(inp_s)[0][1]
    label = '🔴 LIKELY LATE' if prob >= 0.5 else '🟢 ON TIME'
    return label, prob

# Train Schedule System (from STEP 8)
STATIONS  = ['Central', 'North Hub', 'East Park', 'West Gate', 'South Bay',
              'Airport', 'University', 'City Hall', 'Harbor', 'Market Square']

def format_time(h, m):
    return f'{h:02d}:{m:02d}'

def search_trains(origin, destination, hour=None):
    mask = (
        (trains_df['origin'].str.lower() == origin.lower()) &
        (trains_df['destination'].str.lower() == destination.lower())
    )
    results = trains_df[mask].copy()
    if hour is not None:
        results = results[results['sched_hour'] >= hour]
    results = results.sort_values('sched_hour').head(5)
    return results

# Train Delay Prediction (from STEP 9)
TRAIN_FEATURES = ['sched_hour','sched_min','weather_enc','peak_hour',
                  'passengers','origin_enc','dest_enc']

def predict_train_delay(sched_hour, sched_min, weather, peak_hour, passengers, origin, destination):
    try:
        weather_enc = le_weather.transform([weather])[0]
    except ValueError:
        weather_enc = 0 # Default
    try:
        origin_enc  = le_station.transform([origin])[0]
        dest_enc    = le_station.transform([destination])[0]
    except ValueError:
        origin_enc = dest_enc = 0 # Default

    inp   = np.array([[sched_hour, sched_min, weather_enc, peak_hour,
                       passengers, origin_enc, dest_enc]])
    inp_s = train_scaler.transform(inp)
    prob  = train_delay_model.predict_proba(inp_s)[0][1]
    label = '🔴 LIKELY DELAYED' if prob >= 0.5 else '🟢 LIKELY ON TIME'
    return label, prob

# --- Main Application Logic (for command line demo) ---
if __name__ == "__main__":
    print("\n🚀 Library AI & Train System App")
    print("-----------------------------------")
    print("Choose a functionality:")
    print("1. Book Recommendations")
    print("2. NLP Book Search")
    print("3. Predict Overdue Return")
    print("4. Search Train Schedules")
    print("5. Predict Train Delay")
    print("6. Exit")

    while True:
        choice = input("Enter your choice (1-6): ")

        if choice == '1':
            book_title = input("Enter a book title to find recommendations for: ")
            recs = recommend_books(book_title)
            if isinstance(recs, str):
                print(recs)
            else:
                print("\n--- Recommendations ---")
                print(recs.to_string(index=False))

        elif choice == '2':
            query = input("Enter your search query (e.g., 'books about space'): ")
            results = nlp_search(query)
            if results.empty:
                print("No results found.")
            else:
                print("\n--- Search Results ---")
                for _, row in results.iterrows():
                    avail_status = '✅' if row['available'] else '❌'
                    print(f"  {avail_status} [{row['score']:.3f}] {row['title']} — {row['genre']} (★{row['rating']})")

        elif choice == '3':
            print("\n--- Overdue Prediction ---")
            user_age     = int(input("User Age: "))
            due_days     = int(input("Due Days (e.g., 7, 14, 21): "))
            total_borrow = int(input("Total Books Borrowed by User: "))
            genre        = input("Book Genre: ")
            genre_match  = int(input("Genre Matches user's favorite (1 for Yes, 0 for No): "))
            borrow_month = int(input("Borrow Month (1-12): "))
            borrow_wday  = int(input("Borrow Weekday (0=Mon, 6=Sun): "))
            rating       = float(input("Book Rating (e.g., 4.5): "))
            label, prob = predict_late_return(user_age, due_days, total_borrow, genre,
                                              genre_match, borrow_month, borrow_wday, rating)
            print(f"Prediction: {label} (probability = {prob:.2%})")

        elif choice == '4':
            print("\n--- Train Schedule Search ---")
            origin      = input("Origin Station: ")
            destination = input("Destination Station: ")
            sch = search_trains(origin, destination)
            if sch.empty:
                print("No trains found for this route.")
            else:
                print("\n--- Available Trains ---")
                for _, row in sch.iterrows():
                    status = '🔴 Often delayed' if row['is_delayed'] else '🟢 Usually on time'
                    print(f"  {row['train_id']} | Departs {format_time(row['sched_hour'], row['sched_min'])} | Weather: {row['weather']} | {status}")

        elif choice == '5':
            print("\n--- Train Delay Prediction ---")
            sched_hour  = int(input("Scheduled Hour (0-23): "))
            sched_min   = int(input("Scheduled Minute (0-59): "))
            weather     = input("Weather (Clear, Rainy, Foggy, Stormy): ")
            peak_hour   = int(input("Is it Peak Hour (1 for Yes, 0 for No): "))
            passengers  = int(input("Number of Passengers: "))
            origin      = input("Origin Station: ")
            destination = input("Destination Station: ")
            label, prob = predict_train_delay(sched_hour, sched_min, weather, peak_hour, passengers, origin, destination)
            print(f"Prediction: {label} (probability = {prob:.2%})")

        elif choice == '6':
            print("Exiting application. Goodbye!")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 6.")
