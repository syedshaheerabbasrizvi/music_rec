import pandas as pd
import numpy as np
import logging
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from .data_processing import FEATURE_MAPPINGS

logger = logging.getLogger(__name__)

def reset_session_history():
    """Reset the session history to an empty dictionary."""
    st.session_state.session_history = {}
    logger.info("Session history reset for new script execution.")

def compute_features(df, preferences, genres, excluded_genres, artists, songs, modified_params):
    """
    Computes similarity scores for songs based on user preferences with strict Release Year filtering,
    score breakdown, and exclusion of previously recommended songs.
    Only processes features explicitly specified in preferences.
    """
    logger.info("\n📊 Computing feature similarities and filtering songs...")
    df_filtered = df.copy()
    initial_count = len(df_filtered)

    # Exclude previously recommended songs
    if 'session_history' in st.session_state and st.session_state.session_history:
        song_ids = set(st.session_state.session_history.keys())
        df_filtered = df_filtered[~df_filtered['Track URI'].isin(song_ids)]
        logger.info(f"Excluded {initial_count - len(df_filtered)} previously recommended songs. Remaining songs: {len(df_filtered)}")
        initial_count = len(df_filtered)
        if df_filtered.empty:
            logger.warning("All songs excluded due to session history. Reset session history or try a broader prompt.")
            st.error("All songs have been previously recommended. Please reset session history or try a broader prompt.")
            return None, None, None, None, None, None, None, None, None, []

    # 1. Strict Release Year Filtering
    if 'Release Year' in modified_params:
        try:
            if isinstance(modified_params['Release Year'], tuple):
                min_year, max_year = modified_params['Release Year']
                df_filtered = df_filtered[(df_filtered['Release Year'] >= min_year) & (df_filtered['Release Year'] <= max_year)]
                logger.info(f"Filtered by Release Year range ({min_year}, {max_year}). Remaining songs: {len(df_filtered)} (from {initial_count})")
            else:
                year = int(modified_params['Release Year'])
                df_filtered = df_filtered[df_filtered['Release Year'] == year]
                logger.info(f"Filtered by specific Release Year ({year}). Remaining songs: {len(df_filtered)} (from {initial_count})")
            initial_count = len(df_filtered)
            if df_filtered.empty:
                logger.warning("No songs match the specified Release Year criteria.")
                st.error(f"No songs match the Release Year criteria ({modified_params['Release Year']}). Try a broader year range.")
                return None, None, None, None, None, None, None, None, None, []
        except Exception as e:
            logger.error(f"Error filtering by Release Year: {e}")
            st.error(f"Error filtering by Release Year: {e}")
            return None, None, None, None, None, None, None, None, None, []

    # 2. Filter by Artists
    if artists:
        artist_query = '|'.join([f"(?i)\\b{art}\\b" for art in artists])
        df_filtered = df_filtered[df_filtered['Artist Name(s)'].str.contains(artist_query, na=False, regex=True)]
        logger.info(f"Filtered by Artists. Remaining songs: {len(df_filtered)} (from {initial_count})")
        initial_count = len(df_filtered)
        if df_filtered.empty:
            logger.info("No songs match the specified artists.")
            st.error("No songs match the specified artists.")
            return None, None, None, None, None, None, None, None, None, []

    # 3. Filter by Specific Songs
    if songs:
        song_query = '|'.join([f"(?i)\\b{s}\\b" for s in songs])
        df_filtered = df_filtered[df_filtered['Track Name'].str.contains(song_query, na=False, regex=True)]
        logger.info(f"Filtered by Songs. Remaining songs: {len(df_filtered)} (from {initial_count})")
        initial_count = len(df_filtered)
        if df_filtered.empty:
            logger.info("No songs match the specified song names.")
            st.error("No songs match the specified song names.")
            return None, None, None, None, None, None, None, None, None, []

    # 4. Filter by Genres (exact match)
    if genres and 'Genres_list' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Genres_list'].apply(
            lambda song_genres: any(g.lower() in [sg.lower() for sg in song_genres] for g in genres)
        )]
        logger.info(f"Filtered by Genres (exact match). Remaining songs: {len(df_filtered)} (from {initial_count})")
        initial_count = len(df_filtered)
        if df_filtered.empty:
            logger.info("No songs match the specified genres.")
            st.error("No songs match the specified genres.")
            return None, None, None, None, None, None, None, None, None, []

    # 5. Exclude by Genres (exact match)
    if excluded_genres and 'Genres_list' in df_filtered.columns:
        df_filtered = df_filtered[~df_filtered['Genres_list'].apply(
            lambda song_genres: any(eg.lower() in [sg.lower() for sg in song_genres] for eg in excluded_genres)
        )]
        logger.info(f"Filtered out Excluded Genres (exact match). Remaining songs: {len(df_filtered)} (from {initial_count})")
        initial_count = len(df_filtered)
        if df_filtered.empty:
            logger.info("All songs filtered out by exclusions. Consider refining your exclusions.")
            st.error("All songs filtered out by exclusions. Consider refining your exclusions.")
            return None, None, None, None, None, None, None, None, None, []

    if df_filtered.empty:
        logger.info("No songs remaining after filtering. Try a broader prompt.")
        st.error("No songs remaining after filtering. Try a broader prompt.")
        return None, None, None, None, None, None, None, None, None, []

    numerical_features = ['Danceability', 'Energy', 'Valence', 'Acousticness', 'Instrumentalness',
                         'Speechiness', 'Popularity', 'Liveness', 'Duration (ms)']
    categorical_features = ['Key', 'Mode', 'Time Signature']

    # Use only specified numerical features
    specified_features = [f for f in preferences if f in numerical_features]
    if not specified_features and not categorical_features and not genres:
        logger.warning("No features or genres specified in preferences.")
        st.error("No features or genres specified. Please include specific music attributes or genres in your prompt.")
        return None, None, None, None, None, None, None, None, None, []

    # Scale numerical features
    feature_ranges = {
        'Energy': (0, 1), 'Valence': (0, 1), 'Danceability': (0, 1), 'Acousticness': (0, 1),
        'Instrumentalness': (0, 1), 'Speechiness': (0, 1), 'Liveness': (0, 1),
        'Popularity': (0, 100), 'Duration (ms)': (60000, 600000)
    }
    numerical_sim = np.ones(len(df_filtered))
    scaler = None
    if specified_features:
        scaler = MinMaxScaler(clip=True)
        scaler.fit(pd.DataFrame({f: feature_ranges[f] for f in specified_features}))
        X_num = scaler.transform(df_filtered[specified_features])
        user_input = pd.DataFrame([[preferences[f] for f in specified_features]], columns=specified_features)
        user_input_scaled = scaler.transform(user_input)
        weights = [1.0 / len(specified_features)] * len(specified_features)
        distances = np.sqrt(np.sum([w * (user_input_scaled[0][i] - X_num[:, i])**2 for i, w in enumerate(weights)], axis=0))
        alpha = 2.0
        numerical_sim = np.exp(-alpha * distances)

    # Compute categorical match score and details
    categorical_scores = np.ones(len(df_filtered))
    categorical_details = [[] for _ in range(len(df_filtered))]
    has_categorical = any(f in preferences and preferences[f] != -1 for f in categorical_features)
    if has_categorical:
        num_categorical = sum(1 for f in categorical_features if f in preferences and preferences[f] != -1)
        weight_per_feature = 1.0 / num_categorical if num_categorical > 0 else 0
        for i, (_, song) in enumerate(df_filtered.iterrows()):
            cat_score = 0
            cat_detail = []
            for feature in categorical_features:
                if feature in preferences and preferences[feature] != -1:
                    if song[feature] == preferences[feature]:
                        cat_score += weight_per_feature
                        cat_detail.append(f"{feature} match ({feature}={song[feature]})")
                    else:
                        cat_detail.append(f"No {feature} match (Song {feature}={song[feature]}, User {feature}={preferences[feature]})")
            categorical_scores[i] = cat_score
            categorical_details[i] = cat_detail
    else:
        categorical_details = ["No categorical features specified"] * len(df_filtered)

    # Compute genre match score and details
    genre_scores = np.ones(len(df_filtered))
    genre_details = ["No genres specified"] * len(df_filtered)
    has_genres = len(genres) > 0
    if has_genres:
        for i, (_, song) in enumerate(df_filtered.iterrows()):
            song_genres = song['Genres_list']
            if any(g.lower() in [sg.lower() for sg in song_genres] for g in genres):
                genre_scores[i] = 1
                genre_details[i] = f"Genre match (Song Genres={song_genres}, User Genres={genres})"
            else:
                genre_details[i] = f"No genre match (Song Genres={song_genres}, User Genres={genres})"

    # Compute overall score
    max_score = 0.0
    overall_scores = np.zeros(len(df_filtered))
    if specified_features:
        overall_scores += 0.6 * numerical_sim
        max_score += 0.6
    if has_categorical:
        overall_scores += 0.3 * categorical_scores
        max_score += 0.3
    if has_genres:
        overall_scores += 0.1 * genre_scores
        max_score += 0.1
    if max_score > 0:
        overall_scores /= max_score
    else:
        overall_scores = np.ones(len(df_filtered))  # Default to 1 if no features specified (e.g., only artists/songs)

    # Store score breakdown
    score_breakdown = {}
    for i, idx in enumerate(df_filtered.index):
        score_breakdown[idx] = {
            'numerical': {f: numerical_sim[i] for f in specified_features},
            'categorical': {f: categorical_scores[i] * (1.0 / num_categorical) if has_categorical and f in preferences and preferences[f] != -1 else 0 for f in categorical_features},
            'genre': genre_scores[i]
        }

    logger.info(f"✅ Feature computation complete. {len(df_filtered)} songs processed.")
    return df_filtered, overall_scores, numerical_sim, categorical_scores, categorical_details, genre_scores, genre_details, scaler, specified_features, score_breakdown

def format_feature(feature_name, value):
    """
    Helper function to format feature values for display.
    """
    if feature_name == 'Duration (ms)':
        minutes = value / 60000
        if minutes < 3:
            return f"Short ({minutes:.2f} min)"
        elif minutes <= 5:
            return f"Medium ({minutes:.2f} min)"
        else:
            return f"Long ({minutes:.2f} min)"
    elif feature_name == 'Release Year':
        if isinstance(value, tuple):
            return f"Recent ({value[0]}-{value[1]})" if value[0] > 2010 else f"Classic ({value[0]}-{value[1]})"
        return f"Recent ({int(value)})" if value > 2010 else f"Modern ({int(value)})" if value >= 1990 else f"Classic ({int(value)})"
    elif feature_name in ['Danceability', 'Energy', 'Valence', 'Acousticness', 'Instrumentalness', 'Speechiness', 'Liveness']:
        if value > 0.7:
            return f"High ({value:.2f})"
        elif value >= 0.3:
            return f"Medium ({value:.2f})"
        else:
            return f"Low ({value:.2f})"
    elif feature_name == 'Popularity':
        if value > 60:
            return f"Mainstream ({value:.0f})"
        elif value >= 30:
            return f"Moderate ({value:.0f})"
        else:
            return f"Niche ({value:.0f})"
    elif feature_name == 'Key':
        key_names = {0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F', 6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B', -1: 'Any'}
        return f"Key: {key_names.get(int(value), 'Unknown')}"
    elif feature_name == 'Mode':
        mode_names = {0: 'Major', 1: 'Minor', -1: 'Any'}
        return f"Mode: {mode_names.get(int(value), 'Unknown')}"
    elif feature_name == 'Time Signature':
        ts_names = {3: '3/4 (Waltz)', 4: '4/4 (Common)', -1: 'Any'}
        return f"Time Signature: {ts_names.get(int(value), 'Unknown')}"
    else:
        return f"{feature_name}: {value}"

def generate_playlist(df_filtered, overall_scores, numerical_sim, categorical_scores, categorical_details, genres, excluded_genres, songs, artists, genre_scores, genre_details, scaler, specified_features, preferences, modified_params, num_songs, score_breakdown):
    """
    Generates and displays the playlist with detailed score breakdown and updates session history.
    """
    logger.info(f"\n🎧 Generating your personalized playlist of {num_songs} songs...")

    # Ensure session_history is initialized
    if 'session_history' not in st.session_state:
        st.session_state.session_history = {}
        logger.info("Initialized session_history in generate_playlist.")

    if df_filtered.empty or overall_scores.size == 0:
        logger.info("No songs available to generate a playlist. Try broadening your preferences.")
        st.error("No songs available to generate a playlist. Try broadening your preferences.")
        return

    # Compute max score for normalization
    has_categorical = any(detail != "No categorical features specified" for detail in categorical_details)
    has_genres = len(genres) > 0
    max_score = 0.0
    if specified_features:
        max_score += 0.6
    if has_categorical:
        max_score += 0.3
    if has_genres:
        max_score += 0.1
    if max_score == 0:
        max_score = 1.0  # Default for artist/song-only prompts

    # Select top songs
    df_filtered['overall_score'] = overall_scores
    top_songs = df_filtered.sort_values(by='overall_score', ascending=False).head(num_songs)

    if top_songs.empty:
        logger.info("Could not find any songs matching the criteria to generate a playlist.")
        st.error("Could not find any songs matching the criteria.")
        return

    # Update session history
    for _, song in top_songs.iterrows():
        song_id = song.get('Track URI', song['Track Name'])
        st.session_state.session_history[song_id] = st.session_state.session_history.get(song_id, 0) + 1
    logger.info(f"Updated session history: {st.session_state.session_history}")

    # Compute user vector for display
    user_input_scaled = None
    if specified_features:
        user_input = pd.DataFrame([[preferences[f] for f in specified_features]], columns=specified_features)
        user_input_scaled = scaler.transform(user_input)[0]

    st.markdown(f"## Recommended Playlist (Top {len(top_songs)} Songs, Normalized by {max_score:.2f})")
    st.write("Data provided by Spotify")
    st.write("**User Preferences:**")
    for param, value in modified_params.items():
        st.write(f"- {format_feature(param, value)}")
    if genres:
        st.write(f"- Genres: {', '.join(genres)}")
    if excluded_genres:
        st.write(f"- Excluded Genres: {', '.join(excluded_genres)}")
    if artists:
        st.write(f"- Artists: {', '.join(artists)}")
    if songs:
        st.write(f"- Songs: {', '.join(songs)}")
    if specified_features and user_input_scaled is not None:
        st.write(f"- Scaled Vector: {dict(zip(specified_features, user_input_scaled))}")
    st.markdown("---")

    for idx, (_, song) in enumerate(top_songs.iterrows()):
        duration_min = int(song['Duration (ms)'] // 60000)
        duration_sec = int((song['Duration (ms)'] % 60000) // 1000)
        score = song['overall_score'] * 100

        song_input_scaled = None
        if specified_features:
            song_input = pd.DataFrame([[song[feature] for feature in specified_features]], columns=specified_features)
            song_input_scaled = scaler.transform(song_input)[0]

        st.markdown(f"**{idx+1}. {song['Track Name']} by {song['Artist Name(s)']} (Score: {score:.1f}%)**")
        st.write(f"🎧 **Album**: {song['Album Name']}")
        st.write(f"📅 **Release Date**: {format_feature('Release Year', song['Release Year'])}")
        st.write(f"⏱ **Duration**: {duration_min}:{duration_sec:02d} ({format_feature('Duration (ms)', song['Duration (ms)'])})")
        st.write(f"📈 **Popularity**: {format_feature('Popularity', song['Popularity'])}")
        st.write(f"💃 **Danceability**: {format_feature('Danceability', song['Danceability'])}, "
                f"🔋 **Energy**: {format_feature('Energy', song['Energy'])}, "
                f"😄 **Valence**: {format_feature('Valence', song['Valence'])}")
        st.write(f"🎤 **Speechiness**: {format_feature('Speechiness', song['Speechiness'])}")
        st.write(f"🧘 **Acousticness**: {format_feature('Acousticness', song['Acousticness'])}, "
                f"🎹 **Instrumentalness**: {format_feature('Instrumentalness', song['Instrumentalness'])}")
        st.write(f"🎙 **Liveness**: {format_feature('Liveness', song['Liveness'])}")
        st.write(f"🎸 **Artist Genres**: {', '.join(song['Genres_list'])}")
        st.write(f"🔗 **Spotify URI**: {song.get('Track URI', 'N/A')}")
        st.markdown("**Score Breakdown:**")
        if specified_features:
            st.write("**Numerical Features (Distance-Based Similarity):**")
            st.write(f"- Song Vector (Raw): {dict(zip(specified_features, song_input.iloc[0]))}")
            st.write(f"- Song Vector (Scaled): {dict(zip(specified_features, song_input_scaled))}")
            weights = [1.0 / len(specified_features)] * len(specified_features)
            distance = np.sqrt(np.sum([w * (user_input_scaled[i] - song_input_scaled[i])**2 for i, w in enumerate(weights)]))
            similarity = np.exp(-2.0 * distance)
            st.write(f"- Weighted Differences: {', '.join([f'{f}: {w:.2f} × ({user_input_scaled[i]:.4f} - {song_input_scaled[i]:.4f})²' for i, (f, w) in enumerate(zip(specified_features, weights))])}")
            st.write(f"- Distance: {distance:.4f}")
            st.write(f"- Similarity: {similarity:.4f} = exp(-2.0 × {distance:.4f})")
            st.write(f"- Weighted: {0.6 * similarity:.4f} = 0.6 × {similarity:.4f}")
        st.write(f"**Categorical Matches (Score: {categorical_scores[idx]:.3f}):**")
        for detail in categorical_details[idx]:
            st.write(f"- {detail}")
        st.write(f"**Genre Match (Score: {genre_scores[idx]:.3f}):** {genre_details[idx]}")
        raw_sum = 0.0
        if specified_features:
            raw_sum += 0.6 * similarity
        if has_categorical:
            raw_sum += 0.3 * categorical_scores[idx]
        if has_genres:
            raw_sum += 0.1 * genre_scores[idx]
        st.write(f"**Raw Weighted Sum**: {raw_sum:.3f}")
        st.write(f"**Final Score**: {song['overall_score']:.3f} (Normalized by {max_score:.2f}, {score:.1f}%)")
        st.markdown("---")

    logger.info("\n--- Summary of Preferences Applied ---")
    if modified_params:
        logger.info("Explicitly Set Preferences:")
        for param, value in modified_params.items():
            logger.info(f"  - {format_feature(param, value)}")
    if genres:
        logger.info("Matched Genres:")
        logger.info(f"  - {', '.join(genres)}")
    if excluded_genres:
        logger.info("Excluded Genres:")
        logger.info(f"  - {', '.join(excluded_genres)}")
    if artists:
        logger.info("Selected Artists:")
        logger.info(f"  - {', '.join(artists)}")
    if songs:
        logger.info("Selected Songs:")
        logger.info(f"  - {', '.join(songs)}")
    logger.info(f"Current Session History: {len(st.session_state.session_history)} unique songs recommended this session.")
    logger.info("\n--- Recommendation Generated Successfully! ---")