#!/usr/bin/env python3
import pandas as pd, numpy as np, json
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

DATA_1 = Path('data/mess_waste_GIM_500.csv')
DATA_2 = Path('data/mess_waste_GIM_daily_exams.csv')
DATA = DATA_1 if DATA_1.exists() else DATA_2

df = pd.read_csv(DATA, parse_dates=['date']).sort_values('date').reset_index(drop=True)
df2 = df.copy()
df2['is_exam']    = (df2['event_type']=='Exam').astype(int)
df2['is_holiday'] = (df2['event_type']=='Holiday').astype(int)
df2['is_fest']    = (df2['event_type']=='Fest').astype(int)
df2['sweet_milk']  = (df2['sweet_type']=='Milk-based').astype(int)
df2['sweet_fried'] = (df2['sweet_type']=='Fried').astype(int)
df2['sweet_halwa'] = (df2['sweet_type']=='Halwa-type').astype(int)
df2['yesterday_waste'] = df2['food_waste_kg'].shift(1)
df2 = df2.dropna(subset=['yesterday_waste']).reset_index(drop=True)

FEATURES = ['cooked_kg','temp_c','rain_mm','humidity','is_weekend','is_exam','is_holiday','is_fest','sweet_milk','sweet_fried','sweet_halwa','yesterday_waste']
X = df2[FEATURES]; y = df2['food_waste_kg']

n = len(df2); i_tr = int(0.70*n); i_va = int(0.85*n)
X_train, y_train = X.iloc[:i_tr], y.iloc[:i_tr]
X_val,   y_val   = X.iloc[i_tr:i_va], y.iloc[i_tr:i_va]
X_test,  y_test  = X.iloc[i_va:], y.iloc[i_va:]

best = (1e9, None, None)
for md in [3,4,5,6,7]:
    for msl in [1,2,3,5]:
        m = DecisionTreeRegressor(max_depth=md, min_samples_leaf=msl, random_state=42)
        m.fit(X_train, y_train)
        mae = mean_absolute_error(y_val, m.predict(X_val))
        if mae < best[0]: best = (mae, md, msl)
mae, md, msl = best

model = DecisionTreeRegressor(max_depth=md, min_samples_leaf=msl, random_state=42)
model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
pred = model.predict(X_test)
metrics = {
    'MAE': float(mean_absolute_error(y_test, pred)),
    'RMSE': float(mean_squared_error(y_test, pred, squared=False)),
    'R2': float(r2_score(y_test, pred)),
    'best_depth': md, 'best_min_samples_leaf': msl
}

art = Path('artifacts'); art.mkdir(exist_ok=True, parents=True)
joblib.dump({'model': model, 'columns': list(X.columns)}, art/'gim_tree_model.joblib')
json.dump(metrics, open(art/'metrics.json','w'), indent=2)
json.dump(list(X.columns), open(art/'feature_columns.json','w'), indent=2)

print('Saved artifacts to', art)