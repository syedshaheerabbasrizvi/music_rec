# Music Recommendation System

A Streamlit-based application that generates personalized music playlists from a Spotify dataset, using natural language processing (NLP) to parse user prompts and recommend songs based on genres, artists, and audio features like danceability and energy.

## Features
- **Prompt Parsing**: Enter natural language prompts (e.g., "upbeat indie pop from 2020") to specify genres, artists, songs, or features like "tranquil" or "effervescent."
- **NLP with spaCy**: Matches multi-word phrases and synonyms using ConceptNet Numberbatch embeddings.
- **Song Filtering**: Filters songs by release year, genres, artists, and excludes previously recommended tracks via session history.
- **Visualizations**: Displays bar plots, radar charts, and heatmaps comparing song features to user preferences.
- **Spotify Dataset**: Uses a cleaned Spotify dataset with audio features (Danceability, Energy, Valence, etc.).
- **Version Control**: Managed with Git for easy rollbacks and experimentation.

## Project Structure
```
music_rec/
├── app.py                  # Main Streamlit app
├── music_rec/
│   ├── data_processing.py  # Loads dataset and NLP patterns
│   ├── nlp_processing.py   # Parses user prompts with spaCy and Numberbatch
│   ├── recommendation.py   # Computes song similarities and generates playlists
│   ├── visualization.py    # Creates visual comparisons
├── data/
│   ├── spotify_dataset_cleaned.parquet  # Spotify song data
│   ├── genres.txt                      # Genre list
│   ├── numberbatch.bin                 # ConceptNet embeddings (~1.5 GB)
├── .gitignore              # Excludes large files and temp folders
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

## Prerequisites
- **Python**: 3.8+
- **Git**: For version control
- **Dependencies**: Listed in `requirements.txt`
- **Data Files**:
  - `spotify_dataset_cleaned.parquet`: Spotify song data
  - `numberbatch.bin`: Download from [ConceptNet Numberbatch 19.08](https://github.com/commonsense/conceptnet-numberbatch)
  - `genres.txt`: List of valid genres

## Setup
1. **Clone Repository**:
   ```bash
   git clone https://github.com/your-username/music-rec.git
   cd music-rec
   ```

2. **Create Virtual Environment**:
   ```bash
   python -m venv .venv
   .\venv\Scripts\activate  # Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Example `requirements.txt`:
   ```
   streamlit==1.38.0
   pandas==2.2.2
   numpy==1.26.4
   spacy==3.7.5
   gensim==4.3.3
   scikit-learn==1.5.1
   plotly==5.22.0
   ```

4. **Download Data**:
   - Place `spotify_dataset_cleaned.parquet` and `numberbatch.bin` in `data/`.
   - Create `genres.txt` in `data/` with genres (e.g., `indie pop, jazz, synthpop`).

5. **Run the App**:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Open the app in your browser (default: `http://localhost:8501`).
2. Enter a prompt (e.g., "tranquil jazz from 2010s, except pop").
3. Click **Generate Playlist** to see recommendations, visualizations, and confidence scores.
4. Use **Reset History** to clear session history for new recommendations.
5. Click **Stop App** to halt the app.

## Example Prompts
- "Upbeat indie pop from 2020"
- "Serene synthpop, no vocals"
- "Rhythmic jazz by Miles Davis"
- "Effervescent rock from 1980s, except heavy persecution"

## Development
- **Version Control**: Use Git to track changes:
  ```bash
  git add .
  git commit -m "Add new feature"
  git push origin main
  ```
  Revert to a previous version:
  ```bash
  git log --oneline
  git checkout <commit-hash>
  ```
- **Branches**: Create branches for experiments:
  ```bash
  git checkout -b feature/new-nlp
  ```

## Troubleshooting
- **Numberbatch Error**: Ensure `data/numberbatch.bin` exists and is not corrupted. Test:
  ```python
  from gensim.models import KeyedVectors
  numberbatch = KeyedVectors.load_word2vec_format("data/numberbatch.bin", binary=True)
  ```
- **No Songs Found**: Broaden prompt (e.g., wider year range) or reset history.
- **Logs**: Check logs in console for NLP parsing or data loading issues.

## Future Improvements
- Add more complex synonym mappings.
- Support real-time Spotify API integration.
- Enhance visualizations with interactive filters.

## License
MIT License