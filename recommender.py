import pandas as pd
from collections import defaultdict
from surprise import Dataset, Reader, SVD, KNNBaseline
from surprise.model_selection import train_test_split, cross_validate
from surprise import accuracy
import optuna

def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions."""
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def compute_map_at_k(predictions, k=10, threshold=3.5):
    """Compute MAP@K."""
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        
        if n_rel == 0:
            precisions[uid] = 0
            continue
            
        n_rel_and_rec_k = 0
        sum_precisions = 0
        
        for i in range(min(k, len(user_ratings))):
            est, true_r = user_ratings[i]
            if true_r >= threshold:
                n_rel_and_rec_k += 1
                sum_precisions += n_rel_and_rec_k / (i + 1)
                
        precisions[uid] = sum_precisions / min(k, n_rel)

    return sum(precisions.values()) / len(precisions)

def optimize_svd(df, n_trials=10, progress_bar=None):
    """Uses Optuna to find the best hyperparameters for SVD."""
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)
    
    def objective(trial):
        n_factors = trial.suggest_int('n_factors', 20, 150)
        n_epochs = trial.suggest_int('n_epochs', 10, 40)
        lr_all = trial.suggest_float('lr_all', 0.001, 0.01, log=True)
        reg_all = trial.suggest_float('reg_all', 0.01, 0.1, log=True)
        
        model = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all, random_state=42)
        # Fast cross-validation
        score = cross_validate(model, data, measures=['RMSE'], cv=3, verbose=False)
        return score['test_rmse'].mean()

    def optuna_callback(study, trial):
        if progress_bar:
            progress_bar.progress((trial.number + 1) / n_trials, text=f"Optuna Trial {trial.number + 1}/{n_trials} (Best RMSE: {study.best_value:.4f})")

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, callbacks=[optuna_callback] if progress_bar else None)
    return study.best_params, study.best_value

import numpy as np
from surprise import Prediction

class AdvancedMLWrapper:
    def __init__(self, model):
        self.model = model
        self.user_means = {}
        self.item_means = {}
        self.user_counts = {}
        self.item_counts = {}
        self.global_mean = 3.0
        self.trainset = None
        
    def fit(self, trainset):
        self.trainset = trainset
        self.global_mean = trainset.global_mean
        
        for u in trainset.all_users():
            ratings = [r for (_, r) in trainset.ur[u]]
            self.user_means[u] = np.mean(ratings) if ratings else self.global_mean
            self.user_counts[u] = len(ratings)
            
        for i in trainset.all_items():
            ratings = [r for (_, r) in trainset.ir[i]]
            self.item_means[i] = np.mean(ratings) if ratings else self.global_mean
            self.item_counts[i] = len(ratings)
            
        X_train = []
        y_train = []
        for u, i, r in trainset.all_ratings():
            X_train.append([self.user_means[u], self.user_counts[u], self.item_means[i], self.item_counts[i]])
            y_train.append(r)
            
        self.model.fit(np.array(X_train), np.array(y_train))
        return self
        
    def test(self, testset):
        X_test = []
        for uid, iid, r in testset:
            try:
                u = self.trainset.to_inner_uid(uid)
                u_mean, u_count = self.user_means[u], self.user_counts[u]
            except ValueError:
                u_mean, u_count = self.global_mean, 0
                
            try:
                i = self.trainset.to_inner_iid(iid)
                i_mean, i_count = self.item_means[i], self.item_counts[i]
            except ValueError:
                i_mean, i_count = self.global_mean, 0
                
            X_test.append([u_mean, u_count, i_mean, i_count])
            
        preds = self.model.predict(np.array(X_test))
        
        predictions = []
        for (uid, iid, r), est in zip(testset, preds):
            predictions.append(Prediction(uid, iid, r, est, {}))
            
        return predictions

    def predict(self, uid, iid):
        try:
            u = self.trainset.to_inner_uid(uid)
            u_mean, u_count = self.user_means[u], self.user_counts[u]
        except ValueError:
            u_mean, u_count = self.global_mean, 0
            
        try:
            i = self.trainset.to_inner_iid(iid)
            i_mean, i_count = self.item_means[i], self.item_counts[i]
        except ValueError:
            i_mean, i_count = self.global_mean, 0
            
        est = self.model.predict(np.array([[u_mean, u_count, i_mean, i_count]]))[0]
        return Prediction(uid, iid, None, est, {})

def train_and_evaluate(df, selected_models=None, custom_svd_params=None, progress_bar=None, test_size=0.2):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)
    
    trainset, testset = train_test_split(data, test_size=test_size, random_state=42)
    
    svd_model = SVD(**custom_svd_params, random_state=42) if custom_svd_params else SVD(n_factors=150, n_epochs=30, lr_all=0.005, reg_all=0.04, random_state=42)
    
    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestRegressor
    from surprise import BaselineOnly
    
    all_models = {
        'SVD (Optimized)': svd_model,
        'KNN Baseline (Item-Based)': KNNBaseline(sim_options={'name': 'pearson_baseline', 'user_based': False}),
        'ALS Baseline': BaselineOnly(bsl_options={'method': 'als', 'n_epochs': 20}),
        'XGBoost Regressor': AdvancedMLWrapper(XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)),
        'Random Forest Regressor': AdvancedMLWrapper(RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1))
    }
    
    if selected_models:
        models = {name: all_models[name] for name in selected_models if name in all_models}
    else:
        models = all_models
    
    results = {}
    trained_models = {}
    
    for i, (name, model) in enumerate(models.items()):
        print(f"Training {name}...")
        if progress_bar:
            progress_bar.progress(i / len(models), text=f"Training {name}...")
            
        model.fit(trainset)
        
        if progress_bar:
            progress_bar.progress((i + 0.5) / len(models), text=f"Evaluating {name}...")
            
        predictions = model.test(testset)
        
        rmse = accuracy.rmse(predictions, verbose=False)
        map10 = compute_map_at_k(predictions, k=10, threshold=3.5)
        
        results[name] = {'RMSE': rmse, 'MAP@10': map10}
        trained_models[name] = model
        
    if progress_bar:
        progress_bar.progress(1.0, text="Training and Evaluation Complete!")
        
    return trained_models, results, trainset, testset
