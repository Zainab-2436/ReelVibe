
import os
import re
import html
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
POSTER_DIR = BASE_DIR / "posters"

load_dotenv(BASE_DIR / ".env", override=True)
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()

PAGES = ["Home", "Movies", "My List", "Search"]

st.set_page_config(
    page_title="ReelVibe | Movie Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "saved_movie_ids" not in st.session_state:
    st.session_state.saved_movie_ids = []

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Home"

if "selected_user" not in st.session_state:
    st.session_state.selected_user = 2


# =========================================================
# 2. PROFESSIONAL UI
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --red: #e50914;
        --bg: #08090d;
        --panel: #15161d;
        --muted: #a5a8b3;
        --border: #292a32;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, rgba(229,9,20,.10), transparent 32%),
            #08090d;
        color: #f5f5f7;
    }

    [data-testid="stSidebar"] {
        display: none;
    }

    [data-testid="stHeader"] {
        background: rgba(8,9,13,.95);
    }

    .block-container {
        max-width: 1500px;
        padding: 1.5rem 2rem 3rem;
    }

    /* BRAND */

    .top-brand {
        font-size: clamp(25px, 2.5vw, 34px);
        font-weight: 800;
        letter-spacing: -1px;
        line-height: 1.4;
        white-space: nowrap;
    }

    .top-brand span {
        color: #e50914;
    }

    .top-tagline {
        color: #a5a8b3;
        font-size: 9px;
        letter-spacing: 1.5px;
        margin-top: 0;
    }

    /* NAVIGATION */

    div[data-testid="stSegmentedControl"] {
        background: #15161d;
        border: 1px solid #292a32;
        border-radius: 11px;
        padding: 5px;
    }

    div[data-testid="stSegmentedControl"] button {
        color: #dedee5;
        font-size: 13px;
        font-weight: 700;
        border-radius: 8px;
    }

    /* HERO */

    .hero {
        border: 1px solid #38232a;
        border-radius: 18px;
        padding: 45px 38px;
        margin: 18px 0 24px;
        background:
            radial-gradient(ellipse at 85% 35%, #59131f 0%, transparent 45%),
            linear-gradient(105deg, #151016, #250d15 65%, #100b10);
    }

    .hero-label {
        color: #ff7b84;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 3px;
        text-transform: uppercase;
    }

    .hero h1 {
        font-size: clamp(34px, 4vw, 56px);
        color: white;
        line-height: 1.12;
        letter-spacing: -2px;
        margin: 14px 0;
        font-weight: 800;
    }

    .hero p {
        max-width: 600px;
        color: #c3c1c8;
        line-height: 1.8;
        font-size: 14px;
    }

    .hero-tag {
        display: inline-block;
        margin-top: 12px;
        padding: 8px 12px;
        border: 1px solid #65303b;
        border-radius: 20px;
        color: #ffb8bf;
        font-size: 11px;
    }

    /* SECTION HEADINGS */

    .section-heading {
        font-size: 22px;
        font-weight: 800;
        color: #ffffff;
        margin-top: 30px;
        margin-bottom: 5px;
        letter-spacing: -.5px;
    }

    .section-description {
        color: #a5a8b3;
        font-size: 12px;
        margin-bottom: 17px;
    }

    /* MOVIE CARDS */

    .movie-title {
        color: #f8f8fa;
        font-size: 12px;
        font-weight: 700;
        line-height: 1.5;
        min-height: 36px;
        margin-top: 8px;
        overflow-wrap: anywhere;
    }

    .movie-rating {
        color: #f7c75b;
        font-size: 11px;
        font-weight: 700;
        margin-top: 4px;
    }

    .movie-genres {
        color: #a5a8b3;
        font-size: 10px;
        min-height: 30px;
        margin-top: 4px;
        line-height: 1.6;
    }

    [data-testid="stImage"] img {
        border-radius: 10px;
        border: 1px solid #292a32;
    }

    /* BUTTONS */

    div[data-testid="stButton"] > button,
    div[data-testid="stLinkButton"] > a {
        border-radius: 8px;
        font-weight: 700;
        min-height: 36px;
        font-size: 12px;
    }

    div[data-testid="stButton"] > button[kind="primary"] {
        background: #e50914;
        color: white;
        border: 1px solid #e50914;
    }

    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background: #b80710;
        border-color: #b80710;
    }

    /* INPUTS */

    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        border-radius: 8px;
        background: #15161d;
    }

    /* DETAILS */

    div[data-testid="stExpander"] {
        background: #111218;
        border: 1px solid #292a32;
        border-radius: 9px;
    }

    /* STATISTICS */

    .stat-box {
        background: linear-gradient(145deg, #191a22, #101116);
        border: 1px solid #2b2c35;
        border-radius: 12px;
        padding: 17px;
    }

    .stat-label {
        color: #a5a8b3;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1px;
    }

    .stat-number {
        color: white;
        font-size: 24px;
        font-weight: 800;
        margin-top: 6px;
    }

    .empty-box {
        background: #14151c;
        border: 1px solid #292a32;
        border-radius: 12px;
        padding: 25px;
        color: #b7b9c3;
        text-align: center;
    }

    .footer {
        border-top: 1px solid #292a32;
        padding-top: 20px;
        margin-top: 50px;
        text-align: center;
        color: #898c97;
        font-size: 11px;
        line-height: 1.8;
    }

    @media (max-width: 800px) {
        .block-container {
            padding: 1rem 1rem 2rem;
        }

        .top-brand {
            font-size: 25px;
        }

        .hero {
            padding: 28px 20px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. LOAD DATASET
# =========================================================

@st.cache_data(show_spinner=False)
def load_data():
    movies_path = DATA_DIR / "movie.csv"
    ratings_path = DATA_DIR / "rating.csv"
    links_path = DATA_DIR / "link.csv"

    if not movies_path.exists():
        movies_path = BASE_DIR / "movie.csv"

    if not ratings_path.exists():
        ratings_path = BASE_DIR / "rating.csv"

    if not links_path.exists():
        links_path = BASE_DIR / "link.csv"

    if not movies_path.exists():
        raise FileNotFoundError(
            f"Missing movie dataset: {movies_path}"
        )

    if not ratings_path.exists():
        raise FileNotFoundError(
            f"Missing ratings dataset: {ratings_path}"
        )

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path, nrows=300_000)

    for column in ["movieId"]:
        movies[column] = pd.to_numeric(
            movies[column], errors="coerce"
        )

    for column in ["movieId", "userId", "rating"]:
        ratings[column] = pd.to_numeric(
            ratings[column], errors="coerce"
        )

    movies = movies.dropna(
        subset=["movieId", "title"]
    ).copy()

    ratings = ratings.dropna(
        subset=["movieId", "userId", "rating"]
    ).copy()

    movies["movieId"] = movies["movieId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["userId"] = ratings["userId"].astype(int)

    movies["title"] = movies["title"].fillna("Untitled movie")
    movies["genres"] = movies["genres"].fillna("Unknown")

    if links_path.exists():
        links = pd.read_csv(links_path)

        if "movieId" in links.columns:
            links["movieId"] = pd.to_numeric(
                links["movieId"], errors="coerce"
            )

            links = links.dropna(subset=["movieId"])
            links["movieId"] = links["movieId"].astype(int)

            link_columns = [
                column for column in
                ["movieId", "imdbId", "tmdbId"]
                if column in links.columns
            ]

            movies = movies.merge(
                links[link_columns].drop_duplicates("movieId"),
                on="movieId",
                how="left",
            )

    rating_stats = ratings.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count"),
    ).reset_index()

    movies = movies.merge(
        rating_stats,
        on="movieId",
        how="left",
    )

    movies["avg_rating"] = movies["avg_rating"].fillna(0)
    movies["rating_count"] = movies["rating_count"].fillna(0)

    ratings = ratings[
        ratings["movieId"].isin(movies["movieId"])
    ].copy()

    return movies, ratings


# =========================================================
# 4. COLLABORATIVE FILTERING
# =========================================================

@st.cache_data(show_spinner="Preparing recommendations...")
def build_model(ratings):
    matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        fill_value=0,
    )

    similarity_values = cosine_similarity(matrix)

    similarity_df = pd.DataFrame(
        similarity_values,
        index=matrix.index,
        columns=matrix.index,
    )

    return matrix, similarity_df


def get_recommendations(
    user_id,
    matrix,
    similarity_df,
    ratings,
    movies,
    count=50,
):
    if user_id not in matrix.index:
        return pd.DataFrame()

    user_similarities = similarity_df.loc[user_id].drop(
        index=user_id,
        errors="ignore",
    )

    similar_users = user_similarities[
        user_similarities > 0
    ].nlargest(15)

    watched_movies = set(
        ratings.loc[
            ratings["userId"] == user_id,
            "movieId",
        ].tolist()
    )

    if similar_users.empty:
        return movies[
            ~movies["movieId"].isin(watched_movies)
        ].head(count).copy()

    neighbours = ratings[
        ratings["userId"].isin(similar_users.index)
    ].copy()

    neighbours = neighbours[
        ~neighbours["movieId"].isin(watched_movies)
    ]

    neighbours["similarity"] = neighbours["userId"].map(
        similar_users
    )

    neighbours["weighted_rating"] = (
        neighbours["rating"] * neighbours["similarity"]
    )

    scores = neighbours.groupby("movieId").agg(
        weighted_sum=("weighted_rating", "sum"),
        total_weight=("similarity", "sum"),
        neighbour_count=("rating", "count"),
    ).reset_index()

    scores["recommendation_score"] = (
        scores["weighted_sum"]
        / scores["total_weight"].replace(0, np.nan)
    )

    scores = scores.sort_values(
        ["recommendation_score", "neighbour_count"],
        ascending=False,
    )

    results = scores.merge(
        movies,
        on="movieId",
        how="inner",
    )

    return results.head(count)


# =========================================================
# 5. TMDB POSTERS
# =========================================================

@st.cache_data(ttl=86400, show_spinner=False)
def get_movie_poster(title):
    if not TMDB_API_KEY or not title:
        return None

    clean_title = re.sub(
        r"\s*\(\d{4}\)\s*$",
        "",
        str(title),
    ).strip()

    year_match = re.search(
        r"\((\d{4})\)\s*$",
        str(title),
    )

    params = {
        "api_key": TMDB_API_KEY,
        "query": clean_title,
    }

    if year_match:
        params["year"] = year_match.group(1)

    try:
        response = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params=params,
            timeout=5,
        )

        if response.status_code != 200:
            return None

        results = response.json().get("results", [])

        for result in results[:5]:
            poster_path = result.get("poster_path")

            if poster_path:
                return (
                    "https://image.tmdb.org/t/p/w500"
                    + poster_path
                )

    except requests.RequestException:
        return None

    return None


def poster_for(title):
    poster_url = get_movie_poster(str(title))

    if poster_url:
        return poster_url

    fallback = POSTER_DIR / "default.jpg"

    if fallback.exists():
        return str(fallback)

    return (
        "https://placehold.co/350x500/171923/"
        "FFFFFF?text=ReelVibe"
    )


# =========================================================
# 6. REUSABLE UI HELPERS
# =========================================================

def section_heading(title, description=""):
    safe_title = html.escape(str(title))
    safe_description = html.escape(str(description))

    st.markdown(
        f'<div class="section-heading">{safe_title}</div>'
        f'<div class="section-description">{safe_description}</div>',
        unsafe_allow_html=True,
    )


def get_year(title):
    match = re.search(r"\((\d{4})\)\s*$", str(title))
    return match.group(1) if match else "Not available"


def get_imdb_url(movie):
    imdb_id = movie.get("imdbId")

    try:
        if pd.notna(imdb_id):
            imdb_number = str(int(float(imdb_id))).zfill(7)
            return f"https://www.imdb.com/title/tt{imdb_number}/"
    except (ValueError, TypeError, OverflowError):
        pass

    return None


def render_movie_cards(frame, key_prefix, limit=20):
    if frame is None or frame.empty:
        st.markdown(
            '<div class="empty-box">'
            'No movies found. Try another search or genre.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    frame = frame.drop_duplicates("movieId").head(limit)

    # Five movie cards per row
    for start in range(0, len(frame), 5):
        movie_group = frame.iloc[start:start + 5]
        columns = st.columns(5, gap="medium")

        for column, (_, movie) in zip(columns, movie_group.iterrows()):
            movie_id = int(movie["movieId"])
            title = str(movie["title"])
            genres = str(movie.get("genres", "Unknown"))
            rating = float(movie.get("avg_rating", 0) or 0)
            rating_count = int(movie.get("rating_count", 0) or 0)

            with column:
                st.image(
                    poster_for(title),
                    use_container_width=True,
                )

                st.markdown(
                    f'<div class="movie-title">'
                    f'{html.escape(title)}</div>'
                    f'<div class="movie-rating">'
                    f'{"★ " + format(rating, ".1f") + "/5" if rating > 0 else "☆ Not rated"}'
                    f'</div>'
                    f'<div class="movie-genres">'
                    f'{html.escape(genres.replace("|", " · "))}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Details and My List buttons in the same row
                details_col, list_col = st.columns(
                    [1, 1],
                    gap="small",
                )

                with details_col:
                    with st.popover(
                        "Details",
                        use_container_width=True,
                    ):
                        st.write(f"**Movie:** {title}")
                        st.write(f"**Year:** {get_year(title)}")
                        st.write(
                            f"**Genres:** {genres.replace('|', ', ')}"
                        )

                        if rating > 0:
                            st.write(
                                f"**Average Rating:** {rating:.2f}/5"
                            )

                        st.write(
                            f"**Ratings in loaded sample:** {rating_count:,}"
                        )

                        imdb_url = get_imdb_url(movie)

                        if imdb_url:
                            st.link_button(
                                "View on IMDb ↗",
                                imdb_url,
                                use_container_width=True,
                            )

                with list_col:
                    saved = (
                        movie_id
                        in st.session_state.saved_movie_ids
                    )

                    button_label = (
                        "✓ Saved" if saved else "＋ My List"
                    )

                    if st.button(
                        button_label,
                        key=f"{key_prefix}_save_{movie_id}",
                        use_container_width=True,
                    ):
                        if saved:
                            st.session_state.saved_movie_ids.remove(
                                movie_id
                            )
                        else:
                            st.session_state.saved_movie_ids.append(
                                movie_id
                            )

                        st.rerun()


# =========================================================
# 7. LOAD DATA AND MODEL
# =========================================================

try:
    with st.spinner("Loading your movie universe..."):
        movies, ratings = load_data()
        matrix, similarity_df = build_model(ratings)

except Exception as error:
    st.error(
        "ReelVibe could not load the dataset or recommendation model."
    )
    st.exception(error)
    st.stop()


# =========================================================
# 8. TOP BRAND AND NAVIGATION
# =========================================================

brand_col, nav_col = st.columns(
    [1.2, 3.8],
    vertical_alignment="center",
    gap="medium",
)

with brand_col:
    st.markdown(
        """
        <div class="top-brand">
            Reel<span>Vibe</span> 🎬
        </div>
        <div class="top-tagline">
            YOUR PERSONAL MOVIE UNIVERSE
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_col:
    selected_page = st.segmented_control(
        "Main navigation",
        options=PAGES,
        default=(
            st.session_state.nav_page
            if st.session_state.nav_page in PAGES
            else "Home"
        ),
        key="main_navigation",
        label_visibility="collapsed",
        selection_mode="single",
        width="stretch",
    )

    if selected_page:
        st.session_state.nav_page = selected_page

st.markdown(
    """
    <hr style="
        border: 0;
        border-top: 1px solid #292a32;
        margin: 18px 0 26px 0;
    ">
    """,
    unsafe_allow_html=True,
)

current_page = st.session_state.nav_page

user_ids = sorted(int(value) for value in matrix.index)

if not user_ids:
    st.error("No users were found in the loaded ratings dataset.")
    st.stop()

if st.session_state.selected_user not in user_ids:
    st.session_state.selected_user = (
        2 if 2 in user_ids else user_ids[0]
    )


# =========================================================
# 9. PREPARE MOVIE LISTS
# =========================================================

popular_movies = movies[
    movies["rating_count"] > 0
].sort_values(
    ["rating_count", "avg_rating"],
    ascending=False,
)

top_rated_movies = movies[
    movies["rating_count"] >= 10
].sort_values(
    ["avg_rating", "rating_count"],
    ascending=False,
)


# =========================================================
# 10. HOME PAGE
# =========================================================

if current_page == "Home":

    # Hero banner
    st.markdown(
        """
        <div class="hero">
            <div class="hero-label">Your personal cinema</div>
            <h1>Find your next<br>favourite movie.</h1>
            <p>
                Discover personalised recommendations, explore popular titles,
                and find your next favourite movie with ReelVibe.
            </p>
            <div class="hero-tag">
                ✦ PERSONALIZED WITH COLLABORATIVE FILTERING
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


    # Statistics
    stat1, stat2, stat3 = st.columns(3)

    statistics = [
        (stat1, "MOVIES IN CATALOG", f"{len(movies):,}"),
        (stat2, "USERS IN MODEL", f"{ratings['userId'].nunique():,}"),
        (stat3, "RATINGS PROCESSED", f"{len(ratings):,}"),
    ]

    for column, label, value in statistics:
        with column:
            st.markdown(
                f'<div class="stat-box">'
                f'<div class="stat-label">{label}</div>'
                f'<div class="stat-number">{value}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.write("")

    # User profile for recommendations
    with st.expander("⚙️ Personalize Recommendations"):
        st.selectbox(
            "Select a MovieLens user",
            options=user_ids,
            key="selected_user",
            help="Changing the user changes the recommendation profile.",
        )

        st.caption(
            "Recommendations use rating patterns from users "
            "with similar preferences."
        )

    # Home page genre filter and search
    section_heading(
        "Explore Movies by Genre",
        "Choose a genre or search by title to discover movies.",
    )

    home_genres = sorted({
        genre
        for value in movies["genres"].dropna()
        for genre in str(value).split("|")
        if genre and genre != "(no genres listed)"
    })

    filter_col, search_col = st.columns([1, 2])

    with filter_col:
        selected_home_genre = st.selectbox(
            "Choose Genre",
            options=["All Genres"] + home_genres,
            key="home_genre_filter",
        )

    with search_col:
        home_movie_search = st.text_input(
            "Search Movies",
            placeholder="Search movie titles...",
            key="home_movie_search",
        )

    home_filtered_movies = movies.copy()

    if selected_home_genre != "All Genres":
        home_filtered_movies = home_filtered_movies[
            home_filtered_movies["genres"].str.contains(
                re.escape(selected_home_genre),
                case=False,
                na=False,
                regex=True,
            )
        ]

    if home_movie_search.strip():
        home_filtered_movies = home_filtered_movies[
            home_filtered_movies["title"].str.contains(
                home_movie_search.strip(),
                case=False,
                na=False,
                regex=False,
            )
        ]

    home_filtered_movies = home_filtered_movies.sort_values(
        ["rating_count", "avg_rating"],
        ascending=False,
    )

    st.caption(f"{len(home_filtered_movies):,} movies found")

    render_movie_cards(
        home_filtered_movies,
        "home_genre_results",
        limit=20,
    )

    # Recommendations
    recommendations = get_recommendations(
        st.session_state.selected_user,
        matrix,
        similarity_df,
        ratings,
        movies,
        count=100,
    )

    section_heading(
        "Recommended for You",
        "Movies selected using rating patterns from similar users.",
    )

    render_movie_cards(
        recommendations,
        "home_recommended",
        limit=15,
    )

    # Popular movies
    section_heading(
        "Popular Picks",
        "Movies with the most ratings in the loaded dataset.",
    )

    render_movie_cards(
        popular_movies,
        "home_popular",
        limit=10,
    )

    # Top rated movies
    section_heading(
        "Top Rated Discoveries",
        "Explore highly rated movies from the catalogue.",
    )

    render_movie_cards(
        top_rated_movies,
        "home_top",
        limit=10,
    )


# =========================================================
# 11. MOVIES PAGE
# =========================================================

elif current_page == "Movies":

    section_heading(
        "Explore Movies",
        "Browse the catalogue and find something to watch.",
    )

    genre_values = sorted({
        genre
        for value in movies["genres"].dropna()
        for genre in str(value).split("|")
        if genre and genre != "(no genres listed)"
    })

    filter1, filter2 = st.columns([2, 1])

    with filter1:
        movie_search = st.text_input(
            "Search titles",
            placeholder="Search by movie name...",
            key="movies_search",
        )

    with filter2:
        selected_genre = st.selectbox(
            "Filter by genre",
            ["All Genres"] + genre_values,
            key="movies_genre",
        )

    catalogue = movies.copy()

    if movie_search.strip():
        catalogue = catalogue[
            catalogue["title"].str.contains(
                movie_search.strip(),
                case=False,
                na=False,
                regex=False,
            )
        ]

    if selected_genre != "All Genres":
        catalogue = catalogue[
            catalogue["genres"].str.contains(
                re.escape(selected_genre),
                case=False,
                na=False,
                regex=True,
            )
        ]

    catalogue = catalogue.sort_values(
        ["rating_count", "avg_rating"],
        ascending=False,
    )

    st.caption(
        f"{len(catalogue):,} matching movies · showing up to 50"
    )

    render_movie_cards(
        catalogue,
        "catalogue",
        limit=50,
    )


# =========================================================
# 12. MY LIST PAGE
# =========================================================

elif current_page == "My List":

    section_heading(
        "My List ♥",
        "Your saved movies, ready for your next movie night.",
    )

    saved_ids = st.session_state.saved_movie_ids

    if not saved_ids:
        st.markdown(
            """
            <div class="empty-box">
                <h3>Your list is waiting for its first movie 🎬</h3>
                <p>Save movies using the <b>＋ My List</b> button.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Explore Movies", type="primary"):
            st.session_state.nav_page = "Movies"
            st.rerun()

    else:
        saved_movies = movies[
            movies["movieId"].isin(saved_ids)
        ].copy()

        saved_movies = saved_movies.sort_values("title")

        st.caption(f"{len(saved_movies)} saved movies")

        render_movie_cards(
            saved_movies,
            "saved",
            limit=100,
        )

        st.write("")

        if st.button("Clear My List"):
            st.session_state.saved_movie_ids = []
            st.rerun()


# =========================================================
# 13. SEARCH PAGE
# =========================================================

elif current_page == "Search":

    section_heading(
        "Search ReelVibe",
        "Find a movie by its title.",
    )

    query = st.text_input(
        "Movie title",
        placeholder="Try Toy Story, Batman, or Titanic...",
        key="global_movie_search",
    )

    if query.strip():
        search_results = movies[
            movies["title"].str.contains(
                query.strip(),
                case=False,
                na=False,
                regex=False,
            )
        ].copy()

        search_results = search_results.sort_values(
            ["rating_count", "avg_rating"],
            ascending=False,
        )

        st.caption(f"{len(search_results):,} results")

        render_movie_cards(
            search_results,
            "search",
            limit=50,
        )

    else:
        st.markdown(
            '<div class="empty-box">'
            'Enter a movie title above to start searching.'
            '</div>',
            unsafe_allow_html=True,
        )


# =========================================================
# 14. FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        <b>ReelVibe</b> · Discover your next favourite movie
        <br>
        Built with Python, Streamlit, Pandas and collaborative filtering.
        <br>
        Movie metadata and artwork provided by TMDB.
        This product uses the TMDB API but is not endorsed or certified by TMDB.
    </div>
    """,
    unsafe_allow_html=True,
)
