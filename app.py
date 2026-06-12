import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from recommender import train_and_evaluate, get_top_n
from llm_explainer import explain_recommendations

st.set_page_config(page_title="Netflix Recommender", layout="wide")

@st.cache_data
def load_data():
    """Load the Netflix sampled dataset and return ratings and movies DataFrames."""
    ratings_file = "data/sampled_ratings.csv"
    movies_file = "data/movie_titles.csv"
    
    if not os.path.exists(ratings_file):
        st.error(f"File not found: {ratings_file}. Please run preprocess.py first.")
        return None, None
    
    df_ratings = pd.read_csv(ratings_file)
    
    try:
        df_movies = pd.read_csv(movies_file, encoding='ISO-8859-1', header=None, names=['movie_id', 'year', 'title'], on_bad_lines='skip')
    except Exception as e:
        st.warning(f"Could not load movie titles. Error: {e}")
        df_movies = pd.DataFrame(columns=['movie_id', 'year', 'title'])
    
    df_movies['movie_id'] = df_movies['movie_id'].astype(str)
    df_ratings['movie_id'] = df_ratings['movie_id'].astype(str)
    
    df = pd.merge(df_ratings, df_movies, on='movie_id', how='left')
    df['title'] = df['title'].fillna("Unknown Movie (ID: " + df['movie_id'] + ")")
    return df, df_movies

st.title("🎬 Netflix Prize Recommendation System")

# Sidebar settings for experiment controls
st.sidebar.header("Experimentation Settings")
# Fraction of the whole dataset to use for training/testing (0.1 = 10% of data)
sample_fraction = st.sidebar.slider("Data Fraction (portion of whole dataset)", min_value=0.1, max_value=1.0, value=1.0, step=0.05)
# Test set proportion for train_test_split
test_size = st.sidebar.slider("Test Set Size (fraction)", min_value=0.1, max_value=0.9, value=0.2, step=0.05)

# Load full dataset and then sample according to user selection
df, df_movies = load_data()
if df is not None:
    # Apply sampling if fraction is less than 1.0
    if sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)

    tab1, tab2, tab3 = st.tabs(["📊 Data Exploration", "⚙️ Model Evaluation", "🍿 Personalized Recommendations"])
    
    with tab1:
        st.header("Exploratory Data Analysis")
        st.markdown(f"**Total Ratings in Sample:** {len(df):,}")
        st.markdown(f"**Unique Users:** {df['user_id'].nunique():,}")
        st.markdown(f"**Unique Movies:** {df['movie_id'].nunique():,}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Rating Distribution")
            fig, ax = plt.subplots()
            sns.countplot(data=df, x='rating', palette='viridis', ax=ax)
            ax.set_title("Distribution of Ratings")
            st.pyplot(fig)
        
        with col2:
            st.subheader("Top Rated Movies (by count)")
            top_movies = df['title'].value_counts().head(10)
            fig2, ax2 = plt.subplots()
            sns.barplot(x=top_movies.values, y=top_movies.index, palette='magma', ax=ax2)
            ax2.set_title("Most Frequently Rated Movies")
            st.pyplot(fig2)
    
    with tab2:
        st.header("Model Evaluation")
        st.write("Train and compare advanced recommendation models on the selected data fraction.")
        
        available_models = [
            'SVD (Optimized)', 
            'KNN Baseline (Item-Based)', 
            'ALS Baseline', 
            'XGBoost Regressor', 
            'Random Forest Regressor'
        ]
        selected_models = st.multiselect("Select Models to Train:", available_models, default=['SVD (Optimized)', 'KNN Baseline (Item-Based)'])
        
        run_optuna = st.checkbox("Use Optuna for Hyperparameter Tuning (SVD only)", value=False)
        optuna_trials = st.slider("Number of Optuna Trials", min_value=5, max_value=20, value=10, disabled=not run_optuna)
        
        if st.button("Run Model Training & Evaluation"):
            if len(selected_models) == 0:
                st.error("Please select at least one model to train.")
            else:
                custom_svd_params = None
                if run_optuna and 'SVD (Optimized)' in selected_models:
                    optuna_pb = st.progress(0, text=f"Starting Optuna optimization with {optuna_trials} trials...")
                    from recommender import optimize_svd
                    best_params, best_rmse = optimize_svd(df, n_trials=optuna_trials, progress_bar=optuna_pb)
                    st.success(f"Optuna found best parameters! (CV RMSE: {best_rmse:.4f})")
                    st.json(best_params)
                    custom_svd_params = best_params
                
                train_pb = st.progress(0, text=f"Training on selected data (test size={test_size})...")
                trained_models, results, trainset, testset = train_and_evaluate(
                    df,
                    selected_models=selected_models,
                    custom_svd_params=custom_svd_params,
                    progress_bar=train_pb,
                    test_size=test_size
                )
                st.session_state['trained_models'] = trained_models
                st.session_state['trainset'] = trainset
                st.session_state['testset'] = testset
                
                st.success("Models trained successfully!")
                res_df = pd.DataFrame(results).T
                st.table(res_df.style.format("{:.4f}"))
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                colors = ['skyblue', 'salmon', 'lightgreen', 'orange', 'purple'][:len(selected_models)]
                res_df['RMSE'].plot(kind='bar', ax=axes[0], color=colors)
                axes[0].set_title('RMSE (Lower is Better)')
                res_df['MAP@10'].plot(kind='bar', ax=axes[1], color=colors)
                axes[1].set_title('MAP@10 (Higher is Better)')
                st.pyplot(fig)
    
    with tab3:
        st.header("Personalized Recommendations")
        reco_mode = st.radio("Who are we recommending for?", ["Existing User from Dataset", "New User (Cold Start)"], horizontal=True)
        
        if reco_mode == "Existing User from Dataset":
            if 'trained_models' not in st.session_state:
                st.warning("Please train the models in the 'Model Evaluation' tab first.")
            else:
                col_users, col_models = st.columns(2)
                with col_users:
                    top_users = df['user_id'].value_counts().head(50).index.tolist()
                    selected_user = st.selectbox("Select a User ID:", top_users)
                with col_models:
                    selected_model_name = st.selectbox("Select Model:", list(st.session_state['trained_models'].keys()))
                if st.button("Generate Recommendations"):
                    model = st.session_state['trained_models'][selected_model_name]
                    trainset = st.session_state['trainset']
                    user_inner_id = trainset.to_inner_uid(selected_user)
                    user_rated = set([j for (j, _) in trainset.ur[user_inner_id]])
                    all_movies = set(trainset.all_items())
                    unrated_movies = all_movies - user_rated
                    predictions = []
                    for iid in unrated_movies:
                        raw_iid = trainset.to_raw_iid(iid)
                        est = model.predict(selected_user, raw_iid).est
                        predictions.append((raw_iid, est))
                    predictions.sort(key=lambda x: x[1], reverse=True)
                    top_10 = predictions[:10]
                    st.subheader(f"Top 10 Recommendations for User {selected_user}")
                    id_to_title = dict(zip(df_movies['movie_id'], df_movies['title']))
                    rec_titles = []
                    for idx, (iid, est) in enumerate(top_10):
                        title = id_to_title.get(str(iid), f"Movie ID {iid}")
                        rec_titles.append(title)
                        st.write(f"**{idx+1}.** {title} (Estimated Rating: {est:.2f})")
                    st.session_state['last_recs'] = rec_titles
                    user_history_raw = df[(df['user_id'] == selected_user) & (df['rating'] >= 4.0)]
                    history_list = []
                    for _, row in user_history_raw.head(10).iterrows():
                        history_list.append((row['title'], row['rating']))
                    st.session_state['last_history'] = history_list
        else:
            st.write("Tell us what movies you like, and we'll instantly find similar ones using Item‑Item similarity.")
            popular_movies = df['title'].value_counts().head(2000).index.tolist()
            fav_movies = st.multiselect("Select your favorite movies:", popular_movies)
            if st.button("Discover Movies"):
                if 'KNN Baseline (Item-Based)' not in st.session_state.get('trained_models', {}):
                    st.error("Please train the 'KNN Baseline (Item-Based)' model first in the Model Evaluation tab.")
                elif len(fav_movies) == 0:
                    st.warning("Select at least one movie.")
                else:
                    knn_model = st.session_state['trained_models']['KNN Baseline (Item-Based)']
                    trainset = st.session_state['trainset']
                    title_to_id = dict(zip(df_movies['title'], df_movies['movie_id']))
                    id_to_title = dict(zip(df_movies['movie_id'], df_movies['title']))
                    similar_items = {}
                    for title in fav_movies:
                        raw_id = title_to_id.get(title)
                        if raw_id:
                            try:
                                inner_id = trainset.to_inner_iid(raw_id)
                                neighbors = knn_model.get_neighbors(inner_id, k=15)
                                for n in neighbors:
                                    similar_items[n] = similar_items.get(n, 0) + 1
                            except ValueError:
                                pass
                    sorted_neighbors = sorted(similar_items.items(), key=lambda x: x[1], reverse=True)
                    st.subheader("Your Instant Recommendations:")
                    rec_titles = []
                    for inner_id, count in sorted_neighbors:
                        raw_id = trainset.to_raw_iid(inner_id)
                        title = id_to_title.get(raw_id, f"Movie ID {raw_id}")
                        if title not in fav_movies and title not in rec_titles:
                            rec_titles.append(title)
                            st.write(f"- {title} (matches {count} of your favorites)")
                        if len(rec_titles) == 10:
                            break
                    st.session_state['last_recs'] = rec_titles
                    st.session_state['last_history'] = [(t, 5.0) for t in fav_movies]
        
        st.divider()
        st.subheader("🤖 Explainable Recommendations (via LLM)")
        openrouter_models = [
            "openai/gpt-4o-mini",
            "openai/gpt-3.5-turbo",
            "anthropic/claude-3-haiku",
            "google/gemini-flash-1.5",
            "meta-llama/llama-3-8b-instruct"
        ]
        api_key = st.text_input("OpenRouter API Key:", type="password")
        llm_model = st.selectbox("Select LLM Model:", openrouter_models)
        if st.button("Generate Explanation"):
            if 'last_recs' not in st.session_state:
                st.error("Generate recommendations first.")
            elif not api_key:
                st.error("Provide an API Key.")
            else:
                with st.spinner("Asking LLM..."):
                    history = st.session_state.get('last_history', [])
                    recs = st.session_state.get('last_recs', [])
                    explanation = explain_recommendations(api_key, llm_model, history, recs)
                    st.info(explanation)
