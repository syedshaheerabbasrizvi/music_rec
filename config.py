import os

# Project root directory (dynamically resolved)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data file paths
DATA_DIR = os.path.join(BASE_DIR, "data")
SPOTIFY_DATASET_PATH = os.path.join(DATA_DIR, "spotify_dataset_cleaned.parquet")
NUMBERBATCH_PATH = os.path.join(DATA_DIR, "numberbatch.bin")
GENRES_PATH = os.path.join(DATA_DIR, "genres.txt")

# Recommendation parameters
NUM_SONGS = 5                    # Default number of songs to recommend
SIMILARITY_THRESHOLD = 0.6       # Numberbatch similarity threshold
FEATURE_WEIGHTS = {
    "numerical": 0.6,
    "categorical": 0.3,
    "genre": 0.1
}  # Weights for scoring in recommendation.py
DEFAULT_VIZ_FEATURES = ["Danceability", "Energy", "Valence"]  # Default visualization features
DEFAULT_FEATURE_VALUE = 0.5       # Fallback value for unspecified features

# Numerical features for scaling (matches recommendation.py)
NUMERICAL_FEATURES = [
    "Danceability", "Energy", "Valence", "Acousticness", "Instrumentalness",
    "Speechiness", "Liveness", "Popularity", "Duration (ms)"
]

# UI and Streamlit settings
STREAMLIT_PORT = 8501           # Default Streamlit port
STREAMLIT_HOST = "localhost"    # Default Streamlit host

# Logging configuration
LOG_LEVEL = "INFO"              # Options: DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Optional: Environment variables for sensitive data (e.g., future API keys)
SPOTIFY_API_KEY = os.getenv("SPOTIFY_API_KEY", None)  # Set via environment if needed
