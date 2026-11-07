"""Standalone regression and survival workflows for the local app and its archives."""

import json
from pathlib import Path


def run(directory):
    import joblib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    config = json.loads((directory / 'config.json').read_text())
    data = pd.read_csv(directory / 'input.csv')
    results = directory / 'results'
    results.mkdir(exist_ok=True)
    survival = config['analysis_type'] == 'survival'
    target = config['outcome_var']
    duration = config.get('duration_column') if survival else None
    predictors = data.drop(columns=[target, *config.get('columns_to_drop', []), *([duration] if survival else [])])
    missingness = predictors.isna().sum()
    categorical = config.get('cat_features', [])
    numeric = [name for name in predictors if name not in categorical]
    for name in categorical:
        predictors[name] = predictors[name].fillna('(missing)').astype(str)
    preprocess = ColumnTransformer([
        ('numeric', make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler()), numeric),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
    ])
    folds = int(config.get('cv_folds', 3))
    seed = 42
    events = None
    if survival:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.nonparametric import kaplan_meier_estimator
        from sksurv.util import Surv

        events = data[target].astype(bool).to_numpy()
        outcome = Surv.from_arrays(events, data[duration].to_numpy())
        models = {'Cox proportional hazards': CoxPHSurvivalAnalysis(alpha=1.0),
                  'Random survival forest': RandomSurvivalForest(n_estimators=100, n_jobs=2, random_state=seed)}
        scoring = None
        metric = 'Concordance index'
    else:
        outcome = data[target].to_numpy()
        models = {'Linear regression': LinearRegression(),
                  'Random forest regression': RandomForestRegressor(n_estimators=100, n_jobs=2, random_state=seed)}
        scoring = 'neg_mean_absolute_error'
        metric = 'MAE'
    train, test = train_test_split(np.arange(len(data)), train_size=float(config.get('train_size_perc', .8)),
                                   random_state=seed, stratify=events)
    if survival and np.bincount(events[train].astype(int), minlength=2).min() < folds:
        raise ValueError('Development data need at least one event and one censored observation per fold.')
    splitter = (StratifiedKFold(folds, shuffle=True, random_state=seed) if survival
                else KFold(folds, shuffle=True, random_state=seed))
    splits = list(splitter.split(train, events[train] if survival else None))
    x_train, x_test = predictors.iloc[train], predictors.iloc[test]
    y_train, y_test = outcome[train], outcome[test]
    comparison = []
    fitted = {}
    for name, model in models.items():
        print(f'Cross-validating {name}', flush=True)
        pipeline = Pipeline([('preprocessing', preprocess), ('model', model)])
        scores = cross_val_score(pipeline, x_train, y_train, cv=splits, scoring=scoring, error_score='raise')
        if not np.isfinite(scores).all():
            raise ValueError(f'Cross-validation produced undefined scores for {name}.')
        fitted[name] = (float(scores.mean()), pipeline)
        comparison.append({'Model': name, f'Mean CV {metric}': scores.mean() if survival else -scores.mean(),
                           'Fold SD': scores.std()})
    selected = max(fitted, key=lambda name: fitted[name][0])
    model = fitted[selected][1].fit(x_train, y_train)
    predicted = model.predict(x_test)
    fig, ax = plt.subplots(figsize=(7, 4))
    if survival:
        times, observed = kaplan_meier_estimator(events[test], data[duration].to_numpy()[test])
        estimator = model.named_steps['model']
        curves = estimator.predict_survival_function(model.named_steps['preprocessing'].transform(x_test))
        end = min(times[-1], curves[0].x[-1])
        grid = np.unique(np.r_[0, curves[0].x[curves[0].x <= end], times[times <= end], end])
        mean_survival = np.mean([
            np.r_[1, curve(curve.x)][np.searchsorted(curve.x, grid, side='right')] for curve in curves
        ], axis=0)
        observed_at_grid = np.r_[1, observed][np.searchsorted(times, grid, side='right')]
        ax.step(grid, observed_at_grid, where='post', label='Observed Kaplan–Meier')
        ax.step(grid, mean_survival, where='post', label='Mean predicted survival')
        ax.set(xlabel=duration, ylabel='Survival probability', ylim=(0, 1.05))
        ax.legend()
        metrics = {metric: model.score(x_test, y_test), 'Observed events': int(events[test].sum()),
                   'Censored observations': int((~events[test]).sum())}
        predictions = {'Follow-up': data[duration].to_numpy()[test], 'Event': events[test], 'Risk score': predicted}
        figure = 'Survival curves'
    else:
        low, high = min(y_test.min(), predicted.min()), max(y_test.max(), predicted.max())
        ax.scatter(y_test, predicted, alpha=.7)
        ax.plot([low, high], [low, high], '--', color='gray')
        ax.set(xlabel='Observed outcome', ylabel='Predicted outcome')
        metrics = {'MAE': mean_absolute_error(y_test, predicted),
                   'RMSE': np.sqrt(mean_squared_error(y_test, predicted)),
                   'R²': r2_score(y_test, predicted) if np.ptp(y_test) else np.nan}
        predictions = {'Observed': y_test, 'Predicted': predicted}
        figure = 'Observed vs predicted'
    fig.savefig(results / f'performance_{figure}.png', bbox_inches='tight', dpi=160)
    plt.close(fig)

    print('Measuring held-out permutation importance', flush=True)
    importance = permutation_importance(model, x_test, y_test, scoring=scoring, n_repeats=5, random_state=seed)
    ranks = pd.Series(importance.importances_mean, index=predictors.columns).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ranks.tail(15).plot.barh(ax=ax)
    ax.set_xlabel('Decrease in concordance index' if survival else 'Increase in MAE')
    fig.savefig(results / 'FI_perm_Held-out importance.png', bbox_inches='tight', dpi=160)
    plt.close(fig)

    pd.DataFrame({'Metric': list(metrics), 'Value': list(metrics.values())}).to_csv(
        results / f'test_performance_{selected}.csv', index=False)
    pd.DataFrame(comparison).to_excel(results / 'aggregated_results_all.xlsx', index=False)
    pd.DataFrame({'CSV record': test + 1, **predictions}).to_csv(results / 'held_out_predictions.csv', index=False)
    missingness.rename('Missing values').to_csv(results / 'missingness.csv', index_label='Predictor')
    joblib.dump(model, results / 'selected_model.pkl')
    (results / 'analysis_summary.json').write_text(json.dumps({
        'analysis_type': config['analysis_type'], 'selected_model': selected,
        'development_rows': len(train), 'test_rows': len(test),
        'selection_rule': f'Development cross-validation ranked models by {metric}.',
    }, indent=2))


if __name__ == '__main__':
    run(Path(__file__).resolve().parent)
