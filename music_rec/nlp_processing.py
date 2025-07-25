import spacy
import logging
import numpy as np
from scipy.spatial.distance import cosine
from .data_processing import FEATURE_MAPPINGS, ABSTRACT_MAPPINGS, GENRE_PATTERNS, FEATURE_PHRASE_PATTERNS

logger = logging.getLogger(__name__)

def parse_user_prompt(prompt, df, numberbatch=None, nlp=None, matcher=None):
    """
    Parse user prompt into preferences, genres, excluded genres, and songs.
    """
    logger.info(f"\n🔍 Starting Prompt Parsing: '{prompt}'")

    if nlp is None or matcher is None:
        logger.error("nlp and matcher must be provided.")
        raise ValueError("nlp and matcher must be provided.")

    doc = nlp(prompt.lower())
    logger.info(f"📜 Tokens detected: {[token.text for token in doc]}")

    preferences = {}
    modified_params = {}
    genres = []
    excluded_genres = []
    songs = []
    artists = []
    confidence_scores = {}
    overall_confidence = 1.0

    # Apply matcher for multi-word expressions
    logger.info("\n⚡ Applying spaCy Matcher for multi-word expressions...")
    matches = matcher(doc)
    matches = sorted(matches, key=lambda x: doc[x[1]:x[2]].text.count(' '), reverse=True)
    matched_spans = set()
    for match_id, start, end in matches:
        span = doc[start:end]
        if any(i in matched_spans for i in range(start, end)):
            continue
        match_text = span.text
        match_label = nlp.vocab.strings[match_id]
        matched_spans.update(range(start, end))
        logger.debug(f"  ✅ Matcher found: '{match_text}' as '{match_label}'")

        if match_label.startswith('GENRE_'):
            if start > 0 and doc[start - 1].text in ["no", "not", "without"]:
                logger.debug(f"      Negation detected for '{match_text}': Adding to excluded_genres")
                excluded_genres.append(match_text)
                confidence_scores[f"Excluded Genre: {match_text}"] = 0.95
            else:
                logger.debug(f"      ✅ Matched '{match_text}' to genre")
                genres.append(match_text)
                confidence_scores[f"Genre: {match_text}"] = 0.95
        elif match_label.startswith('FEATURE_PHRASE_'):
            if match_text in ABSTRACT_MAPPINGS:
                logger.debug(f"      ✅ Matched abstract phrase '{match_text}': Applying mappings {ABSTRACT_MAPPINGS[match_text]}")
                for feature, value in ABSTRACT_MAPPINGS[match_text].items():
                    preferences[feature] = value
                    modified_params[feature] = value
                    confidence_scores[f"Feature: {feature}"] = 0.9
            elif match_text in ["strong beat", "driving rhythm", "fast tempo"]:
                preferences['Danceability'] = 0.9
                preferences['Energy'] = 0.9
                modified_params['Danceability'] = 0.9
                modified_params['Energy'] = 0.9
                confidence_scores['Feature: Danceability'] = 0.8
                confidence_scores['Feature: Energy'] = 0.8
            elif match_text == "high energy":
                preferences['Energy'] = 1.0
                modified_params['Energy'] = 1.0
                confidence_scores['Feature: Energy'] = 0.9
            elif match_text == "low energy":
                preferences['Energy'] = 0.0
                modified_params['Energy'] = 0.0
                confidence_scores['Feature: Energy'] = 0.9
            elif match_text in ["female vocals", "male vocals", "sing-along"]:
                preferences['Instrumentalness'] = 0.1
                modified_params['Instrumentalness'] = 0.1
                confidence_scores['Feature: Instrumentalness'] = 0.7
            elif match_text == "no vocals":
                preferences['Instrumentalness'] = 1.0
                modified_params['Instrumentalness'] = 1.0
                confidence_scores['Feature: Instrumentalness'] = 1.0
            elif match_text == "heavy percussion":
                preferences['Energy'] = 0.8
                preferences['Danceability'] = 0.7
                modified_params['Energy'] = 0.8
                modified_params['Danceability'] = 0.7
                confidence_scores['Feature: Energy'] = 0.7
                confidence_scores['Feature: Danceability'] = 0.7
            elif match_text == "no percussion":
                preferences['Energy'] = 0.2
                preferences['Acousticness'] = 0.7
                modified_params['Energy'] = 0.2
                modified_params['Acousticness'] = 0.7
                confidence_scores['Feature: Energy'] = 0.7
                confidence_scores['Feature: Acousticness'] = 0.7
            elif match_text in ["unconventional time signatures", "freeform", "improvisational"]:
                preferences['Time Signature'] = -1
                preferences['Danceability'] = 0.3
                modified_params['Time Signature'] = -1
                modified_params['Danceability'] = 0.3
                confidence_scores['Feature: Time Signature'] = 0.8
                confidence_scores['Feature: Danceability'] = 0.6
            elif match_text in ["cinematic", "orchestral", "majestic", "expansive"]:
                preferences['Instrumentalness'] = 0.9
                preferences['Energy'] = 0.7
                preferences['Valence'] = 0.6
                modified_params['Instrumentalness'] = 0.9
                modified_params['Energy'] = 0.7
                modified_params['Valence'] = 0.6
                confidence_scores['Feature: Instrumentalness'] = 0.8
                confidence_scores['Feature: Energy'] = 0.7
                confidence_scores['Feature: Valence'] = 0.6
            elif match_text == "uplifting":
                preferences['Energy'] = 0.8
                preferences['Valence'] = 0.9
                modified_params['Energy'] = 0.8
                modified_params['Valence'] = 0.9
                confidence_scores['Feature: Energy'] = 0.9
                confidence_scores['Feature: Valence'] = 0.9
            elif match_text in ["trippy"]:
                preferences['Valence'] = 0.7
                preferences['Energy'] = 0.6
                modified_params['Valence'] = 0.7
                modified_params['Energy'] = 0.6
                confidence_scores['Feature: Valence'] = 0.7
                confidence_scores['Feature: Energy'] = 0.6
            elif match_text in ["wistfulness", "melancholic"]:
                preferences['Valence'] = 0.2
                preferences['Energy'] = 0.3
                modified_params['Valence'] = 0.2
                modified_params['Energy'] = 0.3
                confidence_scores['Feature: Valence'] = 0.8
                confidence_scores['Feature: Energy'] = 0.7
            elif match_text in ["sophisticated", "calming"]:
                preferences['Energy'] = 0.3
                preferences['Valence'] = 0.6
                preferences['Instrumentalness'] = 0.7
                modified_params['Energy'] = 0.3
                modified_params['Valence'] = 0.6
                modified_params['Instrumentalness'] = 0.7
                confidence_scores['Feature: Energy'] = 0.8
                confidence_scores['Feature: Valence'] = 0.7
                confidence_scores['Feature: Instrumentalness'] = 0.7
            elif match_text == "strong emotional build-up":
                preferences['Energy'] = 0.7
                preferences['Valence'] = 0.7
                modified_params['Energy'] = 0.7
                modified_params['Valence'] = 0.7
                confidence_scores['Feature: Energy'] = 0.8
                confidence_scores['Feature: Valence'] = 0.8
            elif match_text == "challenging traditional song structures":
                preferences['Time Signature'] = -1
                preferences['Danceability'] = 0.2
                modified_params['Time Signature'] = -1
                modified_params['Danceability'] = 0.2
                confidence_scores['Feature: Time Signature'] = 0.9
                confidence_scores['Feature: Danceability'] = 0.7

    # Analyze individual tokens
    logger.debug("\n🔬 Analyzing individual tokens for features and years...")
    negation_words = {"not", "no", "never", "non", "non-", "without"}
    for token in doc:
        if token.i in matched_spans:
            logger.debug(f"    ℹ️ Skipping token '{token.text}' at index {token.i}: Already processed by Matcher.")
            continue

        negation = any(t.text in negation_words for t in doc[max(0, token.i - 2):token.i])
        logger.debug(f"  Processing token: '{token.text}' (POS: {token.pos_})")
        logger.debug(f"      Negation detected for '{token.text}': {negation}")

        # Check for year
        if token.pos_ == 'NUM' and token.text.isdigit():
            try:
                year = int(token.text)
                if 1900 <= year <= 2025:
                    logger.debug(f"    ✅ Identified specific year '{year}': Setting Release Year.")
                    preferences['Release Year'] = (year, year)
                    modified_params['Release Year'] = (year, year)
                    confidence_scores[f"Feature: Release Year"] = 1.0
                elif len(token.text) == 4:
                    decade_start = year - (year % 10)
                    decade_end = decade_start + 9
                    if 1900 <= decade_start <= 2025:
                        logger.debug(f"    ✅ Identified decade '{token.text}': Setting Release Year range ({decade_start}, {decade_end}).")
                        preferences['Release Year'] = (decade_start, decade_end)
                        modified_params['Release Year'] = (decade_start, decade_end)
                        confidence_scores[f"Feature: Release Year"] = 0.95
                else:
                    logger.debug(f"    ℹ️ Skipping token '{token.text}': Invalid year.")
            except ValueError:
                logger.debug(f"    ℹ️ Skipping token '{token.text}': Not a valid year.")
            continue

        # Check for decade ranges (e.g., "2010s")
        if token.text.endswith('s') and token.text[:-1].isdigit():
            try:
                decade = int(token.text[:-1])
                if 1900 <= decade <= 2025:
                    logger.debug(f"    ✅ Identified decade '{token.text}': Setting Release Year range ({decade}, {decade + 9}).")
                    preferences['Release Year'] = (decade, decade + 9)
                    modified_params['Release Year'] = (decade, decade + 9)
                    confidence_scores[f"Feature: Release Year"] = 0.95
            except ValueError:
                logger.debug(f"    ℹ️ Skipping token '{token.text}': Invalid decade.")
            continue

        # Check for feature adjectives, verbs, and adverbs
        if token.pos_ in ['ADJ', 'NOUN', 'VERB', 'ADV'] and token.text not in negation_words and token.text not in ["music", "song", "songs", "playlist", "track", "tunes"]:
            token_text = token.text[4:] if token.text.startswith("non-") else token.text
            for feature, mappings in FEATURE_MAPPINGS.items():
                if feature in ['Key', 'Mode', 'Time Signature']:
                    continue
                for adj, value in mappings.items():
                    if adj == token.text:
                        preferences[feature] = value
                        modified_params[feature] = value
                        confidence_scores[f"Feature: {feature}"] = 1.0
                        logger.debug(f"      ✅ Matched '{token.text}' to '{feature}' with value {value}")
                        break
                    elif numberbatch and token_text in numberbatch and adj in numberbatch:
                        try:
                            similarity = 1 - cosine(numberbatch[token_text], numberbatch[adj])
                            logger.debug(f"      🔍 Numberbatch similarity: '{token_text}' vs '{adj}' = {similarity:.3f}")
                            if similarity >= 0.6:
                                preferences[feature] = value
                                modified_params[feature] = value
                                confidence_scores[f"Feature: {feature}"] = similarity
                                logger.debug(f"      ✅ Matched '{token.text}' to '{adj}' for feature '{feature}' (Similarity: {similarity:.3f})")
                                logger.debug(f"        Set {feature} = {value}")
                                break
                        except Exception as e:
                            logger.debug(f"      ⚠️ Numberbatch error for '{token_text}' vs '{adj}': {e}")
                            continue

        # Handle exclusions (e.g., "except pop")
        if token.text in ["except", "but", "rather"]:
            for next_token in doc[token.i + 1:]:
                if next_token.i in matched_spans:
                    continue
                for g_pattern in GENRE_PATTERNS:
                    if len(g_pattern) == 1 and next_token.text == g_pattern[0]["LOWER"]:
                        excluded_genres.append(next_token.text)
                        logger.debug(f"    🚫 Identified exclusion: 'except {next_token.text}'")
                        matched_spans.add(next_token.i)
                        confidence_scores[f"Excluded Genre: {next_token.text}"] = 1.0
                        break
                break

    # Compute overall confidence
    if confidence_scores:
        overall_confidence = np.mean(list(confidence_scores.values()))
    else:
        overall_confidence = 0.0

    # Deduplicate lists
    genres = list(set(genres))
    excluded_genres = list(set(excluded_genres))
    artists = list(set(artists))
    songs = list(set(songs))

    logger.info("\n🎯 Final Parsed Output:")
    logger.info(f"  Preferences: {preferences}")
    logger.info(f"  Modified Parameters: {modified_params}")
    logger.info(f"  Genres: {genres}")
    logger.info(f"  Excluded Genres: {excluded_genres}")
    logger.info(f"  Artists: {artists}")
    logger.info(f"  Songs: {songs}")
    logger.info(f"  Overall Confidence Score: {overall_confidence:.2f}")
    logger.info(f"  Individual Confidence Scores: {confidence_scores}")
    logger.info("✅ Prompt parsing complete.")

    return {
        "preferences": preferences,
        "genres": genres,
        "excluded_genres": excluded_genres,
        "artists": artists,
        "songs": songs,
        "modified_params": modified_params,
        "confidence_scores": confidence_scores,
        "overall_confidence": overall_confidence
    }