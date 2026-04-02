import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from src.ml.btc_predictor import     BTCPredictor

def optimize_ridge_alpha(X, y, alphas=None, n_splits=3,scoring='r2', n_lags=7,windows=None):
    if alphas is None:
        alphas = [0.1, 1, 10, 50, 100, 200]
    if windows is None:
        windows = [7, 14]
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_alpha = None

    if scoring == 'r2':
        best_score = -np.inf
    else:
        best_score = np.inf

    scores_per_alpha = []

    for alpha in alphas:
        fold_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = Ridge(alpha=alpha)
            scaler =StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            model.fit(X_train_scaled, y_train)

            if scoring == 'r2':
                score = model.score(X_val_scaled, y_val)
            else:
                y_pred = model.predict(X_val_scaled)
                score = -np.mean(np.abs(y_val - y_pred))

            fold_scores.append(score)

        mean_score = np.mean(fold_scores)
        scores_per_alpha.append((alpha, mean_score))

        if (scoring == 'r2' and mean_score > best_score) or \
            (scoring == 'r2' and mean_score <  best_score):
            best_score = mean_score
            best_alpha = alpha

        return best_alpha, best_score, scores_per_alpha
