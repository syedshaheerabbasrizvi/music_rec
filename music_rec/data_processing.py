import os
import logging
import spacy
from spacy.matcher import Matcher

logger = logging.getLogger(__name__)

def load_genres_from_drive(file_path=None):
    """
    Loads a list of genres from a local text file.
    Returns a list of spaCy Matcher patterns.
    """
    if file_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "..", "data", "genres.txt")
    genres_list = []
    genre_patterns = []
    try:
        if not os.path.exists(file_path):
            logger.error(f"Genre file '{file_path}' not found. Using default genre patterns.")
            return STATIC_GENRE_PATTERNS
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                genre = line.strip().lower()
                if genre:
                    genres_list.append(genre)
                    genre_words = genre.split()
                    pattern = [{"LOWER": word} for word in genre_words]
                    genre_patterns.append(pattern)
        logger.debug(f"Successfully loaded {len(genres_list)} genres from '{file_path}'.")
        return genre_patterns + STATIC_GENRE_PATTERNS  # Combine with static patterns
    except Exception as e:
        logger.error(f"Error loading genres: {e}. Using default genre patterns.")
        return STATIC_GENRE_PATTERNS

# Static genre patterns as fallback
STATIC_GENRE_PATTERNS = [
    [{'LOWER': 'pop'}],
    [{'LOWER': 'indie'}, {'LOWER': 'pop'}],
    [{'LOWER': 'rock'}],
    [{'LOWER': 'indie'}, {'LOWER': 'rock'}],
    [{'LOWER': 'post'}, {'LOWER': 'rock'}],
    [{'LOWER': 'jazz'}],
    [{'LOWER': 'classical'}],
    [{'LOWER': 'edm'}],
    [{'LOWER': 'electronic'}],
    [{'LOWER': 'hip'}, {'LOWER': 'hop'}],
    [{'LOWER': 'rap'}],
    [{'LOWER': 'reggae'}],
    [{'LOWER': 'lo'}, {'ORTH': '-'}, {'LOWER': 'fi'}],
    [{'LOWER': 'folk'}],
    [{'LOWER': 'acoustic'}],
    [{'LOWER': 'metal'}],
    [{'LOWER': 'doom'}, {'LOWER': 'metal'}],
    [{'LOWER': 'punk'}],
    [{'LOWER': 'blues'}],
    [{'LOWER': 'r&b'}],
    [{'LOWER': 'soul'}],
    [{'LOWER': 'country'}],
    [{'LOWER': 'alternative'}]
]

GENRE_PATTERNS = load_genres_from_drive()

def load_nlp_and_matcher():
    """
    Initializes and returns spaCy NLP model and Matcher with feature phrase patterns.
    """
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    matcher = Matcher(nlp.vocab)
    # Add genre patterns
    for i, pattern in enumerate(GENRE_PATTERNS):
        matcher.add(f"GENRE_{i}", [pattern])
    # Add feature phrase patterns
    for i, pattern in enumerate(FEATURE_PHRASE_PATTERNS):
        matcher.add(f"FEATURE_PHRASE_{i}", [pattern])
    return nlp, matcher

# Dictionary for abstract concepts
ABSTRACT_MAPPINGS = {
    "gym music": {"Danceability": 1, "Energy": 1, "Valence": 0.7},
    "party music": {"Danceability": 1, "Energy": 1, "Valence": 1},
    "study music": {"Acousticness": 1, "Instrumentalness": 1, "Energy": 0.3},
    "relaxing music": {"Valence": 0.5, "Energy": 0.3, "Acousticness": 1},
    "road trip music": {"Danceability": 0.7, "Energy": 0.7, "Valence": 0.7},
    "workout music": {"Danceability": 1, "Energy": 1, "Valence": 0.7},
    "dinner music": {"Energy": 0.3, "Valence": 0.6, "Speechiness": 0.1, "Liveness": 0.2},
    "background music": {"Energy": 0.3, "Valence": 0.5, "Instrumentalness": 0.8},
    "rainy day music": {"Valence": 0.3, "Energy": 0.4, "Acousticness": 0.8},
    "upbeat music": {"Energy": 0.9, "Valence": 0.8, "Danceability": 0.7},
    "chill music": {"Energy": 0.2, "Valence": 0.6, "Danceability": 0.3},
    "sad music": {"Valence": 0.1, "Energy": 0.3},
    "happy music": {"Valence": 0.9, "Energy": 0.8},
    "no vocals": {"Instrumentalness": 1},
    "female vocals": {"Instrumentalness": 0.1},
    "male vocals": {"Instrumentalness": 0.1},
    "strong beat": {"Danceability": 0.9, "Energy": 0.9},
    "driving rhythm": {"Danceability": 0.9, "Energy": 0.9},
    "chart-topping": {"Popularity": 95},
}

FEATURE_MAPPINGS = {
    'Danceability': {
        'danceable': 1, 'groovy': 0.8, 'bouncy': 0.7, 'steady': 0.5,
        'relaxed': 0.3, 'calm': 0.2, 'static': 0
    },
    'Energy': {
        'fast': 1, 'upbeat': 0.8, 'lively': 0.7, 'moderate': 0.5,
        'mellow': 0.3, 'calm': 0.2, 'slow': 0
    },
    'Valence': {
        'happy': 1, 'cheerful': 0.8, 'mild': 0.6, 'neutral': 0.5,
        'nostalgic': 0.4, 'sad': 0.2, 'gloomy': 0
    },
    'Acousticness': {
        'acoustic': 1, 'unplugged': 0.8, 'natural': 0.6, 'mixed': 0.5,
        'electric': 0.3, 'electronic': 0.2, 'synth': 0
    },
    'Instrumentalness': {
        'instrumental': 1, 'orchestral': 0.8, 'ambient': 0.6, 'mixed': 0.5,
        'vocal': 0.3, 'singing': 0.2, 'lyrical': 0
    },
    'Speechiness': {
        'spoken': 1, 'rap': 0.8, 'talking': 0.7,
        'melodic': 0.3, 'instrumental': 0.1, 'nonvocal': 0.0
    },
    'Popularity': {
        'popular': 100, 'mainstream': 80, 'known': 60, 'average': 50,
        'niche': 30, 'underground': 20, 'obscure': 0
    },
    'Liveness': {
        'live': 1, 'concert': 0.8, 'performance': 0.6, 'mixed': 0.5,
        'studio': 0.2, 'clean': 0.1, 'recorded': 0
    },
    'Duration (ms)': {
        'short': 120000, 'brief': 180000, 'medium': 240000, 'average': 240000,
        'long': 300000, 'extended': 360000, 'epic': 480000
    },
    'Release Year': {
        'recent': (2020, 2025), 'new': (2020, 2025), 'modern': (2000, 2025),
        '2000s': (2000, 2009), '2010s': (2010, 2019), '1990s': (1990, 1999),
        '90s': (1990, 1999), '1980s': (1980, 1989), '80s': (1980, 1989),
        '1970s': (1970, 1979), '70s': (1970, 1979), '1960s': (1960, 1969),
        '60s': (1960, 1969), 'classic': (1920, 1989), 'old': (1920, 1989),
        'retro': (1960, 1999)
    },
    'Key': {
        'c': 0, 'c#': 1, 'db': 1, 'd': 2, 'd#': 3, 'eb': 3, 'e': 4, 'f': 5,
        'f#': 6, 'gb': 6, 'g': 7, 'g#': 8, 'ab': 8, 'a': 9, 'a#': 10, 'bb': 10, 'b': 11,
        'any': -1
    },
    'Mode': {
        'major': 0, 'minor': 1, 'any': -1
    },
    'Time Signature': {
        'common': 4, '4/4': 4, 'waltz': 3, '3/4': 3, 'any': -1
    },
}

FEATURE_PHRASE_PATTERNS = [
    [{"LOWER": "strong"}, {"LOWER": "beat"}],
    [{"LOWER": "high"}, {"LOWER": "energy"}],
    [{"LOWER": "low"}, {"LOWER": "energy"}],
    [{"LOWER": "fast"}, {"LOWER": "tempo"}],
    [{"LOWER": "slow"}, {"LOWER": "tempo"}],
    [{"LOWER": "female"}, {"LOWER": "vocals"}],
    [{"LOWER": "male"}, {"LOWER": "vocals"}],
    [{"LOWER": "no"}, {"LOWER": "vocals"}],
    [{"LOWER": "heavy"}, {"LOWER": "percussion"}],
    [{"LOWER": "no"}, {"LOWER": "percussion"}],
    [{"LOWER": "unconventional"}, {"LOWER": "time"}, {"LOWER": "signatures"}],
    [{"LOWER": "freeform"}],
    [{"LOWER": "improvisational"}],
    [{"LOWER": "cinematic"}],
    [{"LOWER": "orchestral"}],
    [{"LOWER": "uplifting"}],
    [{"LOWER": "sing-along"}],
    [{"LOWER": "trippy"}],
    [{"LOWER": "driving"}, {"LOWER": "rhythm"}],
    [{"LOWER": "wistfulness"}],
    [{"LOWER": "sophisticated"}],
    [{"LOWER": "calming"}],
    [{"LOWER": "distracting"}],
    [{"LOWER": "overplayed"}],
    [{"LOWER": "melancholic"}],
    [{"LOWER": "strong"}, {"LOWER": "emotional"}, {"LOWER": "build-up"}],
    [{"LOWER": "majestic"}],
    [{"LOWER": "expansive"}],
    [{"LOWER": "challenging"}, {"LOWER": "traditional"}, {"LOWER": "song"}, {"LOWER": "structures"}]
]