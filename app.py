import streamlit as st
import pandas as pd
import os
import logging
from gensim.models import KeyedVectors
from music_rec.nlp_processing import parse_user_prompt
from music_rec.recommendation import compute_features, generate_playlist, reset_session_history
from music_rec.data_processing import FEATURE_MAPPINGS

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Suppress matplotlib and PIL debug logs
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

# Cache spaCy model and matcher
@st.cache_resource(show_spinner=True)
def load_nlp_and_matcher_cached():
    from music_rec.data_processing import load_nlp_and_matcher
    return load_nlp_and_matcher()

# Initialize session state
if 'session_history' not in st.session_state:
    st.session_state.session_history = {}
if 'stop_app' not in st.session_state:
    st.session_state.stop_app = False
if 'prompt' not in st.session_state:
    st.session_state.prompt = ""
if 'num_songs' not in st.session_state:
    st.session_state.num_songs = 10
if 'playlist_result' not in st.session_state:
    st.session_state.playlist_result = None
if 'confidence_scores' not in st.session_state:
    st.session_state.confidence_scores = {}
if 'overall_confidence' not in st.session_state:
    st.session_state.overall_confidence = 0.0
if 'genres' not in st.session_state:
    st.session_state.genres = []
if 'excluded_genres' not in st.session_state:
    st.session_state.excluded_genres = []
if 'songs' not in st.session_state:
    st.session_state.songs = []
if 'artists' not in st.session_state:
    st.session_state.artists = []
if 'preferences' not in st.session_state:
    st.session_state.preferences = {}
if 'modified_params' not in st.session_state:
    st.session_state.modified_params = {}

# Check for stop request early
if st.session_state.stop_app:
    logger.info("Stopping Streamlit app due to stop_app flag...")
    st.write("Shutting down the application...")
    os._exit(0)

# Load dataset
@st.cache_data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "data", "spotify_dataset.csv")
    cache_path = os.path.join(script_dir, "data", "spotify_dataset_cleaned.parquet")
    
    # Check if cached Parquet file exists and is valid
    if os.path.exists(cache_path):
        try:
            logger.info(f"Loading cached dataset from {cache_path}...")
            df = pd.read_parquet(cache_path)
            logger.info(f"Loaded cached dataset with {len(df)} rows.")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cached dataset: {e}. Reprocessing dataset.")

    # Load and clean dataset
    try:
        logger.info(f"Loading dataset from {file_path}...")
        df = pd.read_csv(file_path, encoding='latin1')
        initial_rows = len(df)
        logger.info(f"Initial dataset size: {initial_rows} rows.")

        # Function to parse Release Date
        def parse_release_date(date_str):
            if pd.isna(date_str):
                return pd.NA
            date_str = str(date_str).strip()
            try:
                # Try parsing as yyyy
                if date_str.isdigit() and len(date_str) == 4:
                    year = int(date_str)
                    if 1900 <= year <= 2025:
                        return year
                # Try parsing as dd-mm-yy
                parsed_date = pd.to_datetime(date_str, dayfirst=True, errors='coerce')
                if pd.notna(parsed_date):
                    return parsed_date.year
                # Log unparseable date
                logger.debug(f"Unparseable date: {date_str}")
                return pd.NA
            except Exception as e:
                logger.debug(f"Error parsing date '{date_str}': {e}")
                return pd.NA

        # Apply date parsing
        df['Release Year'] = df['Release Date'].apply(parse_release_date)
        unparseable_dates = df[df['Release Year'].isna()]['Release Date'].unique()
        if len(unparseable_dates) > 0:
            logger.warning(f"Unparseable dates found: {list(unparseable_dates)[:10]} (showing up to 10)")
        
        # Drop rows with NaN Release Year
        df = df.dropna(subset=['Release Year'])
        logger.info(f"Rows after dropping NaN Release Year: {len(df)} (dropped {initial_rows - len(df)} rows)")

        # Log years dropped by 1900-2025 filter
        df['Release Year'] = df['Release Year'].astype(int)
        invalid_years = df[~df['Release Year'].between(1900, 2025)]['Release Year'].unique()
        if len(invalid_years) > 0:
            logger.info(f"Dropped years outside 1900-2025: {sorted(invalid_years)}")
        df = df[df['Release Year'].between(1900, 2025)]
        logger.info(f"Rows after filtering years (1900-2025): {len(df)}")

        # Process Genres_list
        df['Genres_list'] = df['Genres'].apply(lambda x: [g.strip().lower() for g in str(x).split(',') if g.strip()] if pd.notna(x) else [])
        logger.info(f"Processed Genres_list for {len(df)} rows.")

        # Handle missing Artist Name(s)
        df['Artist Name(s)'] = df['Artist Name(s)'].fillna('Unknown Artist')
        logger.info("Filled missing Artist Name(s) with 'Unknown Artist'.")

        # Diagnostic: Log genre counts for 2020 songs
        df_2020 = df[df['Release Year'] == 2025]
        if not df_2020.empty:
            genre_counts = pd.Series([g for genres in df_2020['Genres_list'] for g in genres]).value_counts()
            logger.info(f"Genres in 2025 songs (top 10): {genre_counts.head(10).to_dict()}")

        # Save to cache
        try:
            df.to_parquet(cache_path, index=False)
            logger.info(f"Saved cleaned dataset to {cache_path}.")
        except Exception as e:
            logger.warning(f"Failed to save cached dataset: {e}")

        logger.info(f"Final dataset size: {len(df)} rows after cleaning.")
        return df
    except Exception as e:
        logger.error(f"Failed to load or clean dataset: {e}")
        st.error(f"Failed to load or clean dataset: {e}")
        return None

# Load ConceptNet Numberbatch
@st.cache_resource(show_spinner=True)
def load_numberbatch():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    numberbatch_txt_path = os.path.join(base_dir, "data", "numberbatch-en.txt")
    numberbatch_bin_path = os.path.join(base_dir, "data", "numberbatch.bin")
    
    logger.info(f"Checking for cached ConceptNet embeddings at: {numberbatch_bin_path}")
    
    if os.path.exists(numberbatch_bin_path):
        try:
            logger.info("Loading ConceptNet Numberbatch embeddings from binary cache...")
            numberbatch = KeyedVectors.load(numberbatch_bin_path, mmap='r')
            logger.info("ConceptNet Numberbatch embeddings loaded from binary cache.")
            return numberbatch
        except Exception as e:
            logger.warning(f"Failed to load binary cache: {e}. Falling back to text file.")
    
    if not os.path.exists(numberbatch_txt_path):
        logger.error(f"ConceptNet file '{numberbatch_txt_path}' not found.")
        st.error(f"ConceptNet file '{numberbatch_txt_path}' not found. Please ensure 'data/numberbatch-en.txt' exists.")
        return None
    
    try:
        logger.info("Loading ConceptNet Numberbatch embeddings from text file...")
        numberbatch = KeyedVectors.load_word2vec_format(numberbatch_txt_path, binary=False)
        logger.info("Saving ConceptNet Numberbatch embeddings to binary cache...")
        numberbatch.save(numberbatch_bin_path)
        logger.info("ConceptNet Numberbatch embeddings loaded and cached successfully.")
        return numberbatch
    except Exception as e:
        logger.error(f"Failed to load ConceptNet embeddings: {e}")
        st.error(f"Failed to load ConceptNet embeddings: {e}")
        return None

# Main app
st.title("🎵 Song Recommendation System")
st.markdown("Find your perfect playlist by describing your music preferences! Powered by Spotify data.")

# Load data and embeddings
df = load_data()
numberbatch = load_numberbatch()
nlp, matcher = load_nlp_and_matcher_cached()
if df is None or numberbatch is None or nlp is None or matcher is None:
    st.stop()

# Sidebar
with st.sidebar:
    st.header("🎸 Set Your Preferences")
    
    # Feature descriptions
    with st.expander("🎹 Available Features"):
        st.markdown("Use these features in your prompt to customize your playlist:")
        for feature, mappings in FEATURE_MAPPINGS.items():
            st.markdown(f"**{feature}**:")
            st.write(", ".join(f"{k} ({v})" if isinstance(v, (int, float)) else f"{k} {v}" for k, v in mappings.items()))
    
    # Example prompts
    with st.expander("🎧 Example Prompts"):
        st.markdown("Try these to test the system:")
        example_prompts = [
            "upbeat pop songs from 2020",
            "relaxing jazz with no vocals from 2010s",
            "high energy rock from 2000s",
            "sad acoustic songs from 1990s",
            "danceable EDM with strong beat from 2023",
            "chill lo-fi for studying, no rap",
            "happy reggae, short duration",
            "instrumental classical from 1960s",
            "melancholic indie with female vocals",
            "cinematic orchestral music, no pop"
        ]
        for prompt in example_prompts:
            if st.button(prompt, key=f"example_{prompt}"):
                st.session_state.prompt = prompt
                st.session_state.artists = []  # Clear artists when selecting example prompt
    
    # Input form
    with st.form(key="playlist_form"):
        st.session_state.prompt = st.text_area(
            "Describe your music preferences (e.g., 'upbeat pop songs from 2020')",
            value=st.session_state.prompt,
            height=100,
            placeholder="Enter your preferences or click an example above"
        )
        # Artist selection
        unique_artists = sorted(
            df['Artist Name(s)'].dropna().str.split(',').explode().str.strip().dropna().unique()
        )
        st.session_state.artists = st.multiselect(
            "Select Artists (optional)",
            options=unique_artists,
            default=st.session_state.artists,
            help="Choose specific artists to filter your playlist."
        )
        st.session_state.num_songs = st.slider("Number of Songs", 1, 20, st.session_state.num_songs)
        submit_button = st.form_submit_button("🎶 Generate Playlist")
    
    # Reset session history button
    if st.button("🔄 Reset Session History"):
        reset_session_history()
        st.session_state.playlist_result = None  # Clear playlist to prevent stale data
        st.write("Session history cleared.")
    
    # Stop App button
    if st.button("🛑 Stop App"):
        logger.info("Stop App button clicked, setting stop_app flag...")
        st.session_state.stop_app = True
        st.rerun()

# Process form submission
if submit_button:
    st.session_state.playlist_result = None  # Clear old playlist
    with st.spinner("🎵 Generating playlist..."):
        if not st.session_state.prompt and not st.session_state.artists:
            st.error("Please enter a prompt or select at least one artist.")
            st.stop()
        logger.info(f"Processing prompt: {st.session_state.prompt}")
        logger.info(f"Selected artists: {st.session_state.artists}")
        try:
            parsed_data = parse_user_prompt(st.session_state.prompt, df, numberbatch=numberbatch, nlp=nlp, matcher=matcher)
            st.session_state.preferences = parsed_data["preferences"]
            st.session_state.genres = parsed_data["genres"]
            st.session_state.excluded_genres = parsed_data["excluded_genres"]
            st.session_state.songs = parsed_data["songs"]
            st.session_state.modified_params = parsed_data["modified_params"]
            st.session_state.confidence_scores = parsed_data["confidence_scores"]
            st.session_state.overall_confidence = parsed_data["overall_confidence"]

            logger.info(f"Parsed preferences: {st.session_state.preferences}")
            logger.info(f"Modified params: {st.session_state.modified_params}")

            result = compute_features(
                df, st.session_state.preferences, st.session_state.genres,
                st.session_state.excluded_genres, st.session_state.artists,
                st.session_state.songs, st.session_state.modified_params
            )
            if result is None or result[0] is None:
                release_year_info = st.session_state.modified_params.get('Release Year', 'Not specified')
                st.error(f"No songs match the criteria. Check your prompt or dataset (Release Year: {release_year_info}). Try a broader prompt, e.g., 'pop songs from 2010s'.")
                logger.warning(f"No songs matched criteria with Release Year: {release_year_info}")
                st.stop()

            st.session_state.playlist_result = result
        except Exception as e:
            logger.error(f"Error generating playlist: {e}")
            st.error(f"Error generating playlist: {e}")
            st.stop()

# Display playlist and visualizations
if st.session_state.playlist_result and all(k in st.session_state for k in ['genres', 'excluded_genres', 'songs', 'artists', 'preferences', 'modified_params']):
    df_filtered, overall_scores, numerical_sim, categorical_scores, categorical_details, genre_scores, genre_details, scaler, specified_features, score_breakdown = st.session_state.playlist_result
    generate_playlist(
        df_filtered, overall_scores, numerical_sim, categorical_scores, categorical_details,
        st.session_state.genres, st.session_state.excluded_genres, st.session_state.songs, st.session_state.artists,
        genre_scores, genre_details, scaler, specified_features, st.session_state.preferences,
        st.session_state.modified_params, st.session_state.num_songs, score_breakdown
    )
    if scaler and specified_features:
        user_values = scaler.transform(pd.DataFrame([[st.session_state.preferences[f] for f in specified_features]], columns=specified_features))[0]
        top_songs = df_filtered.sort_values('overall_score', ascending=False).head(st.session_state.num_songs)
        from music_rec.visualization import plot_visualizations
        plot_visualizations(top_songs, user_values, scaler, specified_features, numerical_sim)
    st.write(f"**Request Confidence**: {st.session_state.overall_confidence:.2f}")
    st.write("**Detailed Confidence Scores**:")
    for feature, score in st.session_state.confidence_scores.items():
        st.write(f"- {feature}: {score:.2f}")