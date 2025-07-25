import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def plot_visualizations(top_songs, user_values, scaler, specified_features, numerical_sim):
    """
    Generate visualizations for recommended songs based on user preferences.
    
    Args:
        top_songs (pd.DataFrame): DataFrame of recommended songs with features and scores.
        user_values (list): Numerical values for user preferences for specified_features.
        scaler (MinMaxScaler): Fitted scaler for normalizing feature values.
        specified_features (list): List of feature names to visualize.
        numerical_sim: Unused parameter (kept for compatibility).
    """
    logger.info("Starting visualization generation...")

    # Validate inputs
    if not isinstance(top_songs, pd.DataFrame) or top_songs.empty:
        logger.warning("No songs available for visualization. Skipping all plots.")
        st.warning("No songs available for visualization.")
        return
    if not specified_features:
        logger.warning("No specified features provided. Skipping all plots.")
        st.warning("No specified features provided for visualization.")
        return
    if not user_values or len(user_values) != len(specified_features):
        logger.warning(f"Invalid user_values: {user_values}. Expected length {len(specified_features)}. Skipping all plots.")
        st.warning("Invalid user preferences for visualization.")
        return
    if not all(f in top_songs.columns for f in specified_features):
        logger.warning(f"Some specified features {specified_features} not found in top_songs columns {list(top_songs.columns)}. Skipping all plots.")
        st.warning("Specified features not found in song data.")
        return

    # Bar Plot: Recommendation Scores
    scores = [song['overall_score'] * 100 for _, song in top_songs.iterrows()]
    song_names = [f"{song['Track Name']} by {song['Artist Name(s)']}" for _, song in top_songs.iterrows()]
    if scores and song_names:
        logger.info("Generating bar plot for recommendation scores...")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=scores, y=song_names, hue=song_names, legend=False, palette='viridis', ax=ax)
        ax.set_xlabel('Recommendation Score (%)')
        ax.set_ylabel('Song')
        ax.set_title('Top Recommended Songs by Score')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        logger.info("Bar plot displayed successfully.")
    else:
        logger.info("Skipping Bar Plot: No scores or song names available.")
        st.info("Skipping Bar Plot: No songs available.")

    # Radar Chart: User Preferences vs Top Song (requires at least 3 features)
    if len(specified_features) >= 3 and len(top_songs) > 0:
        logger.info("Generating radar chart...")
        categories = specified_features
        num_vars = len(categories)
        top_song = top_songs.iloc[0]
        try:
            top_song_values = scaler.transform(pd.DataFrame([[top_song[f] for f in specified_features]], columns=specified_features))[0]
            scaled_user_values = scaler.transform(pd.DataFrame([user_values], columns=specified_features))[0]
            scaled_user_values = np.clip(scaled_user_values, 0, 1)
            top_song_values = np.clip(top_song_values, 0, 1)

            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1]
            scaled_user_values = np.append(scaled_user_values, scaled_user_values[0])
            top_song_values = np.append(top_song_values, top_song_values[0])

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            ax.fill(angles, scaled_user_values, color='blue', alpha=0.3, label='User Preferences')
            ax.plot(angles, scaled_user_values, color='blue', linewidth=2)
            ax.fill(angles, top_song_values, color='red', alpha=0.3, label=f"{top_songs.iloc[0]['Track Name']} by {top_songs.iloc[0]['Artist Name(s)']}")
            ax.plot(angles, top_song_values, color='red', linewidth=2)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=10, wrap=True)
            ax.set_rlabel_position(0)
            ax.set_ylim(0, 1.2)
            ax.set_title('User Preferences vs Top Song Features', size=15, pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            logger.info("Radar chart displayed successfully.")
        except Exception as e:
            logger.error(f"Error generating radar chart: {e}")
            st.error("Failed to generate radar chart due to data issues.")
    else:
        logger.info(f"Skipping Radar Chart: Insufficient features ({len(specified_features)} < 3) or no songs.")
        st.info("Skipping Radar Chart: Need at least 3 features and 1 song.")

    # Heatmap: Feature Similarity
    if specified_features and len(top_songs) > 0:
        logger.info("Generating heatmap...")
        try:
            similarity_matrix = np.zeros((len(top_songs), len(specified_features)))
            scaled_user_values = scaler.transform(pd.DataFrame([user_values], columns=specified_features))[0]
            for i, (idx, song) in enumerate(top_songs.iterrows()):
                song_vec = scaler.transform(pd.DataFrame([[song[f] for f in specified_features]], columns=specified_features))[0]
                sim_scores = [1 - np.abs(scaled_user_values[j] - song_vec[j]) for j in range(len(specified_features))]
                similarity_matrix[i] = sim_scores

            fig, ax = plt.subplots(figsize=(max(8, len(specified_features) * 1.5), max(6, len(top_songs) * 0.8)))
            sns.heatmap(similarity_matrix, xticklabels=specified_features, yticklabels=song_names, cmap='YlOrRd', annot=True, fmt='.2f', vmin=0, vmax=1, cbar_kws={'label': 'Similarity'}, ax=ax)
            ax.set_title('Feature Similarity to User Preferences')
            ax.set_xlabel('Features')
            ax.set_ylabel('Songs')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            logger.info("Heatmap displayed successfully.")
        except Exception as e:
            logger.error(f"Error generating heatmap: {e}")
            st.error("Failed to generate heatmap due to data issues.")
    else:
        logger.info("Skipping Heatmap: No features or songs available.")
        st.info("Skipping Heatmap: No features or songs available.")