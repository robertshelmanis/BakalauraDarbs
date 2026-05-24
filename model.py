import warnings
warnings.filterwarnings("ignore")

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Datu ielāde no EXCEL faila
excel_file = "/content/drive/MyDrive/alkoholiskie_dzerieni.xlsx"

df = pd.read_excel(excel_file)
df["week_start"] = pd.to_datetime(df["week_start"])
df = df.sort_values("week_start").reset_index(drop=True)

# Jauno atribūtu izveide, sezonalitātes un citu vēsturisko rādītāju izveide
def create_features(df):
    out = df.copy()
    d = out["week_start"]

    out["week_of_year"] = d.dt.isocalendar().week.astype(int)
    out["month"]        = d.dt.month
    out["quarter"]      = d.dt.quarter
    out["year"]         = d.dt.year
    out["day_of_year"]  = d.dt.dayofyear

    out["week_sin"] = np.sin(2 * np.pi * out["week_of_year"] / 52)
    out["week_cos"] = np.cos(2 * np.pi * out["week_of_year"] / 52)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    # Iepriekšējo nedēļu iekodējumi
    for lag in [1, 2, 3, 4, 8, 12, 26, 52]:
        out[f"lag_{lag}"] = out["demand"].shift(lag)

    # Slīdošā loga vertības (4 - mēnesis, 8 - divi mēneši, 13 - ceturksnis,
    # 26 - pusgads, 52 - gads)
    shifted = out["demand"].shift(1)
    for w in [4, 8, 13, 26, 52]:
        out[f"rmean_{w}"]  = shifted.rolling(w).mean()
        out[f"rstd_{w}"]   = shifted.rolling(w).std()
        out[f"rmin_{w}"]   = shifted.rolling(w).min()
        out[f"rmax_{w}"]   = shifted.rolling(w).max()

    # Pret vidējo
    out["ratio_to_4w_avg"]  = out["demand"] / out["rmean_4"].replace(0, np.nan)
    out["ratio_to_52w_avg"] = out["demand"] / out["rmean_52"].replace(0, np.nan)

    # vai nedēļa ietilpst vasaras periodā (šajā periodā iekļauts arī maijs, ņemot vērā
    # konsultāciju ar noliktavas darbinieku)
    out["is_summer"] = out["month"].isin([5, 6, 7, 8]).astype(int)

    return out

# tiek izveidoti atribūti katrai nedēļai 
df_feat = create_features(df)

# Atbrīvojas no neeksistējošām vērtībām, lai nebūtu kļūdas programmas koda izpildē
df_model = df_feat.dropna().reset_index(drop=True)

FEATURE_COLS = [c for c in df_model.columns if c not in [
    "week_start", "week_number", "year", "demand"
]]

X = df_model[FEATURE_COLS] # Pazīmes 
y = df_model["demand"] # Izejas vērtība

# Modeļa trenēšana ar 5 dažādiem dalījumiem
split_skaits = 5
tscv = TimeSeriesSplit(n_splits=split_skaits)

fold_results = []
oof_preds = np.full(len(X), np.nan)

print(f"{'Fold':>4}  {'Train':>7}  {'Val':>5}  {'MAE':>7}  {'RMSE':>7}  {'R²':>7}  {'MAPE':>7}")
print("-" * 55)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=3.0,
        min_child_weight=3,
        gamma=0.1,
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    preds = model.predict(X_val)
    oof_preds[val_idx] = preds

    mae  = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2   = r2_score(y_val, preds)
    mape = np.mean(np.abs((y_val - preds) / y_val)) * 100

    fold_results.append({"mae": mae, "rmse": rmse, "r2": r2, "mape": mape})
    print(f"  {fold:>2}   {len(train_idx):>6}  {len(val_idx):>5}  {mae:>6.1f}  {rmse:>6.1f}  {r2:>6.3f}  {mape:>5.1f}%")

avg = pd.DataFrame(fold_results).mean()
print("-" * 55)
print(f" Avg                     {avg['mae']:>6.1f}  {avg['rmse']:>6.1f}  {avg['r2']:>6.3f}  {avg['mape']:>5.1f}%")
print()


# Uztrenējam pēdējo gala modeli, kas tiks izmantots prognozēšanai
final_model = XGBRegressor(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=1.0,
    reg_lambda=3.0,
    min_child_weight=3,
    gamma=0.1,
    random_state=42,
    verbosity=0,
)
final_model.fit(X, y, verbose=False)

# Tiek saglabāts modelis pkl formātā, lai to varētu izmantot prognozēšanai ārpus šīs vides
import joblib

joblib.dump({
    "model": final_model,
    "feature_cols": FEATURE_COLS
}, "demand_model.pkl")

importances = pd.Series(final_model.feature_importances_, index=FEATURE_COLS)
importances = importances.sort_values(ascending=False)

# Prognoze priekš nākošā gada
FORECAST_WEEKS = 4 * 12

# Keep a running copy of demand history for lag computation
history = df[["week_start", "demand"]].copy()

forecasts = []
for step in range(FORECAST_WEEKS):
    next_date = history["week_start"].iloc[-1] + pd.DateOffset(weeks=1)
    recent = history["demand"].values

    row = {}
    row["week_of_year"] = next_date.isocalendar()[1]
    row["month"]        = next_date.month
    row["quarter"]      = (next_date.month - 1) // 3 + 1
    row["day_of_year"]  = next_date.timetuple().tm_yday

    row["week_sin"]  = np.sin(2 * np.pi * row["week_of_year"] / 52)
    row["week_cos"]  = np.cos(2 * np.pi * row["week_of_year"] / 52)
    row["month_sin"] = np.sin(2 * np.pi * row["month"] / 12)
    row["month_cos"] = np.cos(2 * np.pi * row["month"] / 12)

    for lag in [1, 2, 3, 4, 8, 12, 26, 52]:
        row[f"lag_{lag}"] = recent[-lag] if lag <= len(recent) else np.nan

    for w in [4, 8, 13, 26, 52]:
        window = recent[-w:] if w <= len(recent) else recent
        row[f"rmean_{w}"]  = np.mean(window)
        row[f"rstd_{w}"]   = np.std(window, ddof=1) if len(window) > 1 else 0
        row[f"rmin_{w}"]   = np.min(window)
        row[f"rmax_{w}"]   = np.max(window)

    row["ratio_to_4w_avg"]  = recent[-1] / np.mean(recent[-4:])
    row["ratio_to_52w_avg"] = recent[-1] / np.mean(recent[-52:]) if len(recent) >= 52 else 1.0

    row["is_summer"]      = 1 if row["month"] in [5, 6, 7, 8] else 0

    # Prognoze
    X_pred = pd.DataFrame([row])[FEATURE_COLS]
    pred = float(final_model.predict(X_pred)[0])
    pred = max(pred, 0)
    pred_rounded = int(round(pred / 9) * 9) 

    forecasts.append({
        "week_start": next_date,
        "demand_raw": round(pred),
        "demand_crate": pred_rounded,
    })

    # Ievadam jaunos iegūtos datus, lai tie var tikt izmantoti kā vēsturiskie dati,
    # lai veiktu nākamās prognozes
    history = pd.concat([
        history,
        pd.DataFrame([{"week_start": next_date, "demand": pred}])
    ], ignore_index=True)

forecast_df = pd.DataFrame(forecasts)

pred_8_week_sum = 0
for i, r in forecast_df.iterrows():
  if (i < 8):
    pred_8_week_sum += r['demand_raw']
  
print("=" * 50)
print(" FIRST 8 WEEK PREDICTION FOR VALIDATION STEP")
print("=" * 50)

print(f"Demand prediction : {pred_8_week_sum}")

# Grafiku izveide, lai varētu uzskatāmi attēlot iegūtos rezultātus
fig, axes = plt.subplots(3, 1, figsize=(15, 13), gridspec_kw={"height_ratios": [3, 1.5, 1.2]})

ax = axes[0]
# Vēsturiskie dati
ax.plot(df["week_start"], df["demand"], color="#1D9E75", linewidth=1.2, alpha=0.8, label="Ieejas dati")
ax.plot(df["week_start"], df["demand"], "o", color="#1D9E75", markersize=2.5, alpha=0.5)

# Prognoze
ax.plot(forecast_df["week_start"], forecast_df["demand_crate"],
        "s-", color="#D85A30", linewidth=2.5, markersize=4, label="Prognoze", zorder=5)

ax.axvline(df["week_start"].iloc[-1], color="gray", linestyle=":", alpha=0.5)

ax.set_title("Prece 1 - Nedēļas pieprasījums un nākamā gada prognoze", fontsize=15, fontweight="bold")
ax.set_ylabel("Pieprasījums (vienības)")
ax.legend(loc="upper left", fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax.grid(axis="y", alpha=0.25)

# Salīdzinājums starp prognozi un reālo pieprasījumu
ax2 = axes[1]
valid_dates = df_model["week_start"][valid_mask]
ax2.plot(valid_dates, y[valid_mask], color="#1D9E75", linewidth=1, alpha=0.7, label="Ieejas dati")
ax2.plot(valid_dates, oof_preds[valid_mask], color="#D85A30", linewidth=1, alpha=0.8, label="Modeļa prognoze")
ax2.set_title("Modeļa veiktspējas novērtējums apmācības datos", fontsize=12, fontweight="bold")
ax2.set_ylabel("Pieprasījums")
ax2.legend(fontsize=9)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax2.grid(axis="y", alpha=0.25)

# Pazīmju nozīmīgums
ax3 = axes[2]
top_n = 5
top = importances.head(top_n)
bars = ax3.barh(top.index[::-1], top.values[::-1], color="#378ADD", height=0.55)
ax3.set_title(f"Top {top_n} pazīmju nozīmīgums", fontsize=12, fontweight="bold")
ax3.set_xlabel("Nozīmīgums")
ax3.grid(axis="x", alpha=0.25)

plt.tight_layout()
plt.savefig("weekly_forecast_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved: weekly_forecast_results.png")
