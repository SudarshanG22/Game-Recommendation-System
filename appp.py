import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load the games list
games = pickle.load(open("game_pickle.pkl", 'rb'))
games_df = pd.DataFrame(games)

# Combine features into one column (you can modify this as per your dataset structure)
games_df['Features'] = games_df['Name'] + " " + games_df['Genre']

# Vectorize the features (simple approach using CountVectorizer or TF-IDF if needed)
vectorizer = CountVectorizer(stop_words='english')
features_matrix = vectorizer.fit_transform(games_df['Features'])

# Function to get top 5 recommendations
def recommend(selected_game):
    # Find the index of the selected game
    game_idx = games_df[games_df['Name'] == selected_game].index[0]
    
    # Compute cosine similarity between the selected game and all other games
    cosine_sim = cosine_similarity(features_matrix[game_idx], features_matrix).flatten()
    
    # Get the indices of the top 6 most similar games (including the selected game itself)
    similar_games_idx = cosine_sim.argsort()[-6:-1]  # Get the top 5 similar games, excluding itself
    
    # Get the names of the top 5 similar games
    recommendations = games_df['Name'].iloc[similar_games_idx].values
    
    # Ensure the selected game is excluded from recommendations
    recommendations = [game for game in recommendations if game != selected_game]
    
    # If fewer than 5 recommendations are returned, fill the rest with other games
    if len(recommendations) < 5:
        # Get the games that are not in the recommendations list
        remaining_games = games_df[~games_df['Name'].isin(recommendations)]['Name'].values
        # Add remaining games to make up 5 recommendations
        recommendations.extend(remaining_games[:5 - len(recommendations)])

    return recommendations

# Adding Background Image

st.markdown(
    """
    <style>
    .stApp {
        background: url("https://img.freepik.com/premium-vector/modern-dark-abstract-background-with-red-light_55870-87.jpg") no-repeat center center fixed;
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create a Streamlit header
st.header("Games Recommendation System")

# Create a selectbox for the user to select a game
selected_game = st.selectbox("Select Game from Dropdown", games_df['Name'])

if st.button("Recommend"):
    # Get the top 5 recommendations
    recommendations = recommend(selected_game)
    
    # Display the recommendations
    st.write("Top 5 Game Recommendations:")
    st.markdown("""
    <style>
        .recommendation-text {
            font-size: 20px;  /* Adjust this value as needed */
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

    # Display the recommendations in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'<p class="recommendation-text">1st: {recommendations[0]}</p>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<p class="recommendation-text">2nd: {recommendations[1]}</p>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<p class="recommendation-text">3rd: {recommendations[2]}</p>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<p class="recommendation-text">4th: {recommendations[3]}</p>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<p class="recommendation-text">5th: {recommendations[4]}</p>', unsafe_allow_html=True)

    


