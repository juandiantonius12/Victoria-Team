"""
Feature engineering for one-hour-ahead PM2.5 forecasting.

Input columns available per row: station, observation_timestamp,
PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, wd (wind direction,
16-point compass), WSPM (wind speed). The target, PM2_5_next_hour, is
the PM2.5 reading one hour after the row's timestamp for the same
station.

PM10 is given a rich lag/rolling treatment as the primary pollutant
signal: PM10 and PM2.5 are both particulate matter, produced by
largely the same sources (traffic, coal heating) and transported by
the same weather, so PM10's own recent history is strongly informative
about how PM2.5 is likely to move.

Beyond that base feature set, three additional feature families are
built, each targeting information the base set does not capture:
  - cross-station "donor" features for every pollutant (not just PM10),
    since different pollutants trace somewhat different emission
    sources and transport patterns;
  - a wind-direction-conditioned version of the PM10 donor feature,
    since which neighbouring station is actually relevant depends on
    which way the wind is blowing right now, not just on average;
  - atmospheric-stability proxies (diurnal temperature range, pressure
    trend, calm-wind hours) capturing the stagnant-air conditions that
    let pollution build up.

All features use only information available at the row's own
timestamp (current and past readings), so they are valid at prediction
time for every row, train or test.
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

OTHER_POLLUTANTS = ["SO2", "NO2", "CO", "O3"]
RICH_POLLUTANT = "PM10"
WEATHER = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]

WD_ORDER = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
WD_ANGLE = {d: i * 22.5 for i, d in enumerate(WD_ORDER)}
WD_BUCKET8 = {d: i // 2 for i, d in enumerate(WD_ORDER)}  # coarsen to 8 compass buckets

RICH_LAGS = [1, 2, 3, 4, 5, 6, 12, 24]
OTHER_LAGS = [1, 2, 3]
ROLL_WINDOWS = [3, 6, 12, 24]
TOP_K_DONOR_STATIONS = 3
CALM_WIND_THRESHOLD = 1.5  # m/s


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build lag, rolling-window, wind, calendar and city-wide
    features for a combined train+test frame.

    A complete hourly timestamp grid is built per station first, so
    that "1 hour ago" always means exactly 1 hour ago even when the
    raw data has gaps.
    """
    orig_index = df.index
    df = df.sort_values(["station", "observation_timestamp"]).copy()

    grid = (df.groupby("station").observation_timestamp.agg(["min", "max"])
              .apply(lambda r: pd.date_range(r["min"], r["max"], freq="h"), axis=1)
              .explode().rename("observation_timestamp").reset_index())
    g = grid.merge(df, on=["station", "observation_timestamp"], how="left")
    g = g.sort_values(["station", "observation_timestamp"]).reset_index(drop=True)
    gb = g.groupby("station", sort=False)

    feats = pd.DataFrame(index=g.index)
    feats["station"] = g.station
    feats["observation_timestamp"] = g.observation_timestamp

    all_pollutants = [RICH_POLLUTANT] + OTHER_POLLUTANTS
    for c in all_pollutants + WEATHER:
        feats[c] = g[c]
    feats["pm10_missing"] = g[RICH_POLLUTANT].isna().astype(int)

    ff = {c: gb[c].ffill(limit=24) for c in all_pollutants + WEATHER}

    # --- rich treatment for PM10, the primary pollutant signal ---
    pm10 = ff[RICH_POLLUTANT]
    pm10_gb = pm10.groupby(g.station, sort=False)
    for k in RICH_LAGS:
        feats[f"pm10_lag{k}"] = pm10_gb.shift(k)
    for k in [1, 2, 3, 6]:
        feats[f"pm10_diff{k}"] = pm10 - pm10_gb.shift(k)
    feats["pm10_accel"] = feats["pm10_diff1"] - (pm10_gb.shift(1) - pm10_gb.shift(2))
    for w in ROLL_WINDOWS:
        r = pm10_gb.rolling(w, min_periods=1)
        feats[f"pm10_rmean{w}"] = r.mean().reset_index(level=0, drop=True)
        feats[f"pm10_rstd{w}"] = r.std().reset_index(level=0, drop=True)
        feats[f"pm10_rmax{w}"] = r.max().reset_index(level=0, drop=True)
        feats[f"pm10_minus_rmean{w}"] = pm10 - feats[f"pm10_rmean{w}"]
    feats["pm10_same_hour_yday"] = pm10_gb.shift(24)
    feats["pm10_rmean24_diff"] = feats["pm10_rmean24"] - feats["pm10_rmean24"].groupby(g.station, sort=False).shift(24)

    # --- standard (lighter) treatment for the remaining pollutants and weather ---
    for c in OTHER_POLLUTANTS + WEATHER:
        s = ff[c]
        s_gb = s.groupby(g.station, sort=False)
        for k in OTHER_LAGS:
            feats[f"{c}_lag{k}"] = s_gb.shift(k)
        feats[f"{c}_diff1"] = s - s_gb.shift(1)
        feats[f"{c}_rmean6"] = s_gb.rolling(6, min_periods=1).mean().reset_index(level=0, drop=True)

    feats["rain_last6"] = ff["RAIN"].groupby(g.station, sort=False).rolling(6, min_periods=1).sum().reset_index(level=0, drop=True)
    feats["temp_dew_spread"] = g.TEMP - g.DEWP

    ang = np.deg2rad(g.wd.map(WD_ANGLE))
    feats["wd_sin"] = np.sin(ang)
    feats["wd_cos"] = np.cos(ang)
    feats["wind_u"] = -g.WSPM * np.sin(ang)
    feats["wind_v"] = -g.WSPM * np.cos(ang)
    feats["wind_u_lag1"] = feats.wind_u.groupby(g.station, sort=False).shift(1)
    feats["wind_v_lag1"] = feats.wind_v.groupby(g.station, sort=False).shift(1)
    feats["wd_code"] = g.wd.map({d: i for i, d in enumerate(WD_ORDER)})

    ts = g.observation_timestamp
    feats["hour"] = ts.dt.hour
    feats["dow"] = ts.dt.dayofweek
    feats["month"] = ts.dt.month
    feats["doy"] = ts.dt.dayofyear
    feats["hour_sin"] = np.sin(2 * np.pi * feats.hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * feats.hour / 24)
    feats["doy_sin"] = np.sin(2 * np.pi * feats.doy / 365.25)
    feats["doy_cos"] = np.cos(2 * np.pi * feats.doy / 365.25)
    feats["is_weekend"] = (feats.dow >= 5).astype(int)

    city10 = pm10.groupby(g.observation_timestamp)
    feats["city_mean_pm10"] = city10.transform("mean")
    feats["city_std_pm10"] = city10.transform("std")
    feats["city_max_pm10"] = city10.transform("max")
    feats["pm10_minus_city"] = pm10 - feats.city_mean_pm10
    feats["city_mean_pm10_lag1"] = feats.city_mean_pm10.groupby(g.station, sort=False).shift(1)
    feats["city_pm10_diff1"] = feats.city_mean_pm10 - feats.city_mean_pm10_lag1
    feats["city_mean_wspm"] = ff["WSPM"].groupby(g.observation_timestamp).transform("mean")
    feats["city_mean_no2"] = ff["NO2"].groupby(g.observation_timestamp).transform("mean")
    feats["city_mean_co"] = ff["CO"].groupby(g.observation_timestamp).transform("mean")

    # --- atmospheric-stability proxies ---
    temp_gb = ff["TEMP"].groupby(g.station, sort=False)
    pres_gb = ff["PRES"].groupby(g.station, sort=False)
    wspm_gb = ff["WSPM"].groupby(g.station, sort=False)
    roll_max = temp_gb.rolling(24, min_periods=3).max().reset_index(level=0, drop=True)
    roll_min = temp_gb.rolling(24, min_periods=3).min().reset_index(level=0, drop=True)
    feats["temp_range24"] = roll_max - roll_min
    feats["pres_trend24"] = ff["PRES"] - pres_gb.shift(24)
    is_calm = (ff["WSPM"] < CALM_WIND_THRESHOLD).groupby(g.station, sort=False)
    feats["calm_hours24"] = is_calm.rolling(24, min_periods=3).sum().reset_index(level=0, drop=True)

    feats["station"] = feats.station.astype("category")

    key = df[["station", "observation_timestamp"]].reset_index()
    out = key.merge(feats, on=["station", "observation_timestamp"], how="left").set_index("index")
    out = out.reindex(orig_index)
    out["station"] = pd.Categorical(out.station, categories=sorted(df.station.unique()))
    return out


def _learn_donor_weights(train: pd.DataFrame, stations, value_col: str, top_k=TOP_K_DONOR_STATIONS):
    """Learn, from training data only, which other stations' `value_col`
    reading is most predictive of each station's next-hour PM2.5.
    PM2_5_next_hour is used only for this one-time, training-only
    correlation computation, never as a per-row input feature."""
    pivot_val = train.pivot_table(index="observation_timestamp", columns="station", values=value_col)
    pivot_next = train.pivot_table(index="observation_timestamp", columns="station", values="PM2_5_next_hour")

    donor_weights = {}
    for s in stations:
        corrs = []
        for other in stations:
            if other == s:
                continue
            pair = pd.concat([pivot_val[other], pivot_next[s]], axis=1).dropna()
            if len(pair) < 100:
                continue
            c = pair.iloc[:, 0].corr(pair.iloc[:, 1])
            corrs.append((other, c))
        corrs.sort(key=lambda x: -x[1])
        donor_weights[s] = corrs[:top_k]
    return donor_weights


def _donor_feature(train, test, stations, value_col, feature_name, donor_weights=None):
    if donor_weights is None:
        donor_weights = _learn_donor_weights(train, stations, value_col)

    grid_val = (pd.concat([train, test])
                  .pivot_table(index="observation_timestamp", columns="station", values=value_col)
                  .sort_index().ffill(limit=24))

    rows = []
    for s in stations:
        donors = donor_weights[s]
        weights = np.array([max(c, 0) for _, c in donors])
        if weights.sum() == 0:
            weights = np.ones(len(donors))
        weights = weights / weights.sum()
        vals = np.zeros(len(grid_val))
        for (donor, _), w in zip(donors, weights):
            vals += grid_val[donor].fillna(grid_val[donor].median()).values * w
        rows.append(pd.DataFrame({"station": s, "observation_timestamp": grid_val.index, feature_name: vals}))
    long_df = pd.concat(rows, ignore_index=True)
    long_df["station"] = long_df["station"].astype(str)
    return long_df


def _merge_on_station_time(F, long_df):
    F = F.copy()
    F["_station_str"] = F["station"].astype(str)
    F = F.merge(long_df, left_on=["_station_str", "observation_timestamp"],
                right_on=["station", "observation_timestamp"], how="left", suffixes=("", "_d"))
    F = F.drop(columns=["_station_str", "station_d"])
    return F


def add_spatial_features(F: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, stations) -> pd.DataFrame:
    """Add cross-station "donor" features: for each pollutant, a
    correlation-weighted average of the stations whose readings are
    most predictive of this station's next-hour PM2.5, at the same
    hour. Uses only same-timestamp readings from other stations, which
    are already known at prediction time (the same principle as the
    city-wide aggregate above)."""
    for value_col, feat_name in [
        ("PM10", "spatial_donor_pm10"),
        ("SO2", "spatial_donor_so2"),
        ("NO2", "spatial_donor_no2"),
        ("CO", "spatial_donor_co"),
        ("O3", "spatial_donor_o3"),
    ]:
        long_df = _donor_feature(train, test, stations, value_col, feat_name)
        F = _merge_on_station_time(F, long_df)
        F[f"{feat_name}_minus_own"] = F[feat_name] - F[value_col]

    return F


def add_wind_dynamic_donor(F: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, stations) -> pd.DataFrame:
    """A wind-direction-conditioned version of the PM10 donor feature:
    for each station and each of 8 wind-direction buckets, the donor
    stations are learned separately (restricted to training rows in
    that bucket), so the feature reflects which neighbour is relevant
    for the wind direction blowing right now, not just on average."""
    train_wb = train.copy()
    train_wb["wbucket"] = train_wb.wd.map(WD_BUCKET8)
    pivot_pm10 = train.pivot_table(index="observation_timestamp", columns="station", values="PM10")
    pivot_next = train.pivot_table(index="observation_timestamp", columns="station", values="PM2_5_next_hour")
    wbucket_by_ts_station = train_wb.set_index(["station", "observation_timestamp"]).wbucket

    donor_weights = {}
    for s in stations:
        s_wb = wbucket_by_ts_station.loc[s]
        for b in range(8):
            ts_in_bucket = s_wb[s_wb == b].index
            if len(ts_in_bucket) < 200:
                donor_weights[(s, b)] = None
                continue
            corrs = []
            for other in stations:
                if other == s:
                    continue
                pair = pd.concat([pivot_pm10[other].reindex(ts_in_bucket), pivot_next[s].reindex(ts_in_bucket)], axis=1).dropna()
                if len(pair) < 50:
                    continue
                c = pair.iloc[:, 0].corr(pair.iloc[:, 1])
                corrs.append((other, c))
            corrs.sort(key=lambda x: -x[1])
            donor_weights[(s, b)] = corrs[:TOP_K_DONOR_STATIONS]

    grid_pm10 = (pd.concat([train, test]).pivot_table(index="observation_timestamp", columns="station", values="PM10")
                   .sort_index().ffill(limit=24))
    all_rows_wd = pd.concat([train[["station", "observation_timestamp", "wd"]], test[["station", "observation_timestamp", "wd"]]])
    all_rows_wd["wbucket"] = all_rows_wd.wd.map(WD_BUCKET8)
    wb_lookup = all_rows_wd.set_index(["station", "observation_timestamp"]).wbucket

    rows = []
    for s in stations:
        vals = np.full(len(grid_pm10), np.nan)
        for b in range(8):
            donors = donor_weights.get((s, b))
            if not donors:
                continue
            weights = np.array([max(c, 0) for _, c in donors])
            if weights.sum() == 0:
                weights = np.ones(len(donors))
            weights = weights / weights.sum()
            contrib = np.zeros(len(grid_pm10))
            for (donor, _), w in zip(donors, weights):
                contrib += grid_pm10[donor].fillna(grid_pm10[donor].median()).values * w
            try:
                ts_this_bucket = wb_lookup.loc[s]
                mask_ts = ts_this_bucket[ts_this_bucket == b].index
                idx_pos = grid_pm10.index.get_indexer(mask_ts)
                idx_pos = idx_pos[idx_pos >= 0]
                vals[idx_pos] = contrib[idx_pos]
            except KeyError:
                pass
        fallback = grid_pm10.mean(axis=1).values
        vals = np.where(np.isnan(vals), fallback, vals)
        rows.append(pd.DataFrame({"station": s, "observation_timestamp": grid_pm10.index, "spatial_donor_pm10_dynamic": vals}))

    long_spatial = pd.concat(rows, ignore_index=True)
    long_spatial["station"] = long_spatial["station"].astype(str)
    F = _merge_on_station_time(F, long_spatial)
    F["spatial_donor_pm10_dynamic_minus_own"] = F["spatial_donor_pm10_dynamic"] - F["PM10"]
    return F


def build_full_feature_set(train: pd.DataFrame, test: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Build the complete feature table for train+test combined."""
    stations = sorted(train.station.unique())
    all_rows = pd.concat([train.assign(_split="train"), test.assign(_split="test")], ignore_index=True)

    F = build_features(all_rows)
    F["_split"] = all_rows["_split"]
    F["id"] = all_rows["id"]
    F[target_col] = all_rows[target_col]
    F["station"] = F["station"].astype("category")

    F = add_spatial_features(F, train, test, stations)
    F = add_wind_dynamic_donor(F, train, test, stations)
    F["station"] = F["station"].astype("category")
    return F
