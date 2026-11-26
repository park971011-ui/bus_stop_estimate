"""
정류장 500m 격자 지역 피처 실험 (dayun/grid_feature_experiment)
=================================================================
가설:
  가까운 정류장들은 지역적인 운행 특성을 공유하므로,
  지역 칸 정보(from_grid_id, to_grid_id)를 추가하면 통행시간 예측이 개선될 수 있다.

비교 기준 (명시):
  - 비교 기준 모델은 이전 실험의 '평일 전용 LightGBM (wd_lgb)' 자체이다.
  - 전체 혼합 모델(50:50 또는 LGB+CB)이나 ExtraTrees가 기준이 아니다.
  - 기존 dayun/weekday_blend_experiment/checkpoints/wd_lgb_*.npz와
    코드·설정·검증 행(row_id)이 일치하면 재사용하고, 일치하지 않거나 없으면 기준도 재학습한다.

좌표 변환 및 500m 격자 정의 (외부 데이터 미사용):
  - 기준점(고정 상수): LAT0 = 36.35 deg, LON0 = 127.35 deg (지구 평균 반지름 R = 6,371,000.0 m)
  - 미터 좌표 변환 (국소 등거리 원통/접평면 근사):
      y_m = (lat - LAT0) * (pi / 180) * R
      x_m = (lon - LON0) * (pi / 180) * R * cos(LAT0 * pi / 180)
  - 500m 고정 격자 인덱스:
      gx = floor(x_m / 500.0),  gy = floor(y_m / 500.0)
      grid_id = f"{gx}_{gy}"
  - 출발 정류장(from_lat, from_lon) -> from_grid_id (범주형)
  - 도착 정류장(to_lat, to_lon)     -> to_grid_id   (범주형)

전처리 및 누출 방지 원칙:
  - 검증 분할: 대회 train(154,327행) 중 평일 월(9/23)·화(9/24)·수(9/25) 3-fold 날짜별 제외 검증.
    무작위 분할 미사용.
  - 새 피처(from_grid_id, to_grid_id)의 범주형 인코딩(pd.CategoricalDtype)은
    각 학습 폴드(f_wd) 내부에서만 적합하고, 검증 폴드(v_wd)에 동일 매핑을 적용한다.
  - 기존 기준 모델의 피처(36개)·LightGBM 하이퍼파라미터·시드(random_state=0)는 동일 유지.
  - 이번 실험에서는 격자 크기(500m)나 다른 설정을 추가로 변경하지 않는다.
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb


# ─── 좌표 변환 및 500m 고정 격자 상수 ─────────────────────────────────────
LAT0 = 36.35
LON0 = 127.35
EARTH_RADIUS_M = 6371000.0
GRID_SIZE_M = 500.0
M_PER_DEG_LAT = np.deg2rad(1.0) * EARTH_RADIUS_M
M_PER_DEG_LON = np.deg2rad(1.0) * EARTH_RADIUS_M * np.cos(np.deg2rad(LAT0))


def coords_to_grid_id(lat: np.ndarray, lon: np.ndarray) -> pd.Series:
    y_m = (lat - LAT0) * M_PER_DEG_LAT
    x_m = (lon - LON0) * M_PER_DEG_LON
    gx = np.floor(x_m / GRID_SIZE_M).astype(np.int64)
    gy = np.floor(y_m / GRID_SIZE_M).astype(np.int64)
    return pd.Series(gx).astype(str) + "_" + pd.Series(gy).astype(str)


def mae(y, p) -> float:
    return float(np.mean(np.abs(y - p)))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def main():
    overall_start = time.time()
    print("=" * 72, flush=True)
    print(">>> [정류장 500m 격자 지역 피처 실험 (기준: 평일 전용 wd_lgb)]", flush=True)
    print("=" * 72, flush=True)

    BASE_DIR  = Path(__file__).resolve().parents[2]
    FINAL_DIR = BASE_DIR / "data" / "final"
    REF_CKPT  = BASE_DIR / "dayun" / "weekday_blend_experiment" / "checkpoints"
    EXP_DIR   = Path(__file__).resolve().parent
    CKPT_DIR  = EXP_DIR / "checkpoints"
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ─── 1. 데이터 로드 ──────────────────────────────────────────────────
    print("\n1. 데이터 로드 (data/final)", flush=True)
    train = pd.read_parquet(FINAL_DIR / "train.parquet")
    test  = pd.read_parquet(FINAL_DIR / "test.parquet")
    print(f"   train: {train.shape[0]:,}행 x {train.shape[1]}컬럼", flush=True)
    print(f"   test:  {test.shape[0]:,}행 x {test.shape[1]}컬럼", flush=True)
    assert train.shape[0] == 154327, f"train 행 수 불일치: {train.shape[0]}"

    # ─── 2. 기준(E56 wd_lgb) 동일 전처리 + 500m 격자 ID 생성 ─────────────
    PEAK = {7, 8, 9, 17, 18, 19}
    SEGK = ['route_id', 'direction', 'from_stop', 'to_stop']
    TRIP = ['date', 'vehicle_id', 'route_id', 'direction', 'trip_no']
    VD   = ['date', 'vehicle_id']
    H    = ['t_prev_gap', 't_blk_n', 't_blk_dist', 't_blk_before', 't_blk_after',
            'v_blk_n', 'v_prev_n', 'v_next_n', 'v_next_trip_gap', 'v_next_trip_n0']
    C    = 3574.0

    def hhmm_to_min(x):
        return (x // 100) * 60 + x % 100

    last_seq = pd.concat([train, test]).groupby(
        ['route_id', 'direction']).stop_seq.max().rename('last_seq')

    def add_static(df):
        df = df.copy()
        dow = pd.to_datetime(df.date).dt.day_of_week
        df['is_weekend']      = (dow >= 5).astype(int)
        df['is_peak']         = df.arrive_hour.isin(PEAK).astype(int)
        df['interval']        = np.select([dow == 5, dow == 6],
                                          [df.interval_sat, df.interval_sun],
                                          default=df.interval_wd)
        df['dlat']            = df.to_lat - df.from_lat
        df['dlon']            = df.to_lon - df.from_lon
        df['seg_id']          = df.from_stop.astype(str) + '_' + df.to_stop.astype(str)
        df['first_min']       = hhmm_to_min(df.first_time)
        df['last_min']        = hhmm_to_min(df.last_time)
        df['hour_from_first'] = df.arrive_hour * 60 - df.first_min
        df = df.join(last_seq, on=['route_id', 'direction'])
        df['is_last_seg']     = (df.stop_seq == df.last_seq).astype(int)
        df['seq_from_end']    = df.last_seq - df.stop_seq
        # 새 격자 피처 2개 (고정 500m 격자, 외부 데이터 미사용)
        df['from_grid_id']    = coords_to_grid_id(df['from_lat'].values, df['from_lon'].values).values
        df['to_grid_id']      = coords_to_grid_id(df['to_lat'].values, df['to_lon'].values).values
        return df

    def add_trip(df):
        df  = df.copy()
        key = ['date', 'vehicle_id', 'route_id', 'direction']
        df  = df.sort_values(key + ['arrive_hour', 'stop_seq'])
        new_trip = (df.groupby(key).stop_seq.diff().fillna(-1) <= 0)
        df['trip_no']          = new_trip.groupby([df[k] for k in key]).cumsum()
        trip = key + ['trip_no']
        df['trip_start_seq']   = df.groupby(trip).stop_seq.transform('min')
        df['seq_in_trip']      = df.stop_seq - df.trip_start_seq
        df['trip_start_hour']  = df.groupby(trip).arrive_hour.transform('min')
        df['hours_in_service'] = df.arrive_hour - df.groupby(
            ['date', 'vehicle_id']).arrive_hour.transform('min')
        return df.sort_index()

    def add_busy(df):
        df = df.copy()
        df['n_bus_hour'] = df.groupby(
            ['date', 'route_id', 'direction', 'arrive_hour']).vehicle_id.transform('nunique')
        return df

    print("\n2. 피처 추출 중...", flush=True)
    t0 = time.time()
    TRAIN = add_busy(add_trip(add_static(train)))
    TEST  = add_busy(add_trip(add_static(test)))
    print(f"   완료 ({time.time() - t0:.1f}초)", flush=True)
    print(f"   좌표 변환 상수: LAT0={LAT0}, LON0={LON0}, R={EARTH_RADIUS_M:.0f}m, "
          f"M_PER_DEG_LAT={M_PER_DEG_LAT:.2f}, M_PER_DEG_LON={M_PER_DEG_LON:.2f}, GRID={GRID_SIZE_M:.0f}m", flush=True)

    # 기준 모델 피처 (E56 wd_lgb 동일 36개)
    FEATS = {
        'base': ['route_id', 'route_type', 'direction', 'interval',
                 'from_lat', 'from_lon', 'to_lat', 'to_lon',
                 'dlat', 'dlon', 'stop_seq', 'dist_m',
                 'arrive_hour', 'is_weekend', 'is_peak'],
        'seg':  ['seg_id'],
        'time': ['first_min', 'last_min', 'hour_from_first', 'is_last_seg', 'seq_from_end'],
        'trip': ['trip_no', 'seq_in_trip', 'trip_start_hour', 'hours_in_service'],
    }
    F_BASE = sum((FEATS[g] for g in ['base', 'seg', 'time', 'trip']), []) + H
    FE_BASE = F_BASE + ['turn_est']                  # 기준 36개
    FE_GRID = FE_BASE + ['from_grid_id', 'to_grid_id']  # 격자 추가 38개
    print(f"   기준 wd_lgb 피처: {len(FE_BASE)}개 | 격자 추가 wd_lgb 피처: {len(FE_GRID)}개", flush=True)

    # 기존 범주형 (기준 wd_lgb와 동일하게 유지)
    BASE_CATS = ['route_id', 'seg_id']
    BASE_CAT_DTYPE = {c: pd.CategoricalDtype(
        sorted(pd.concat([TRAIN, TEST])[c].unique())) for c in BASE_CATS}
    GRID_CATS = ['from_grid_id', 'to_grid_id']

    def to_X_base(df):
        X = df[FE_BASE].copy()
        for c in BASE_CATS:
            X[c] = X[c].astype(BASE_CAT_DTYPE[c])
        return X

    def to_X_grid(df, grid_dtypes):
        X = df[FE_GRID].copy()
        for c in BASE_CATS:
            X[c] = X[c].astype(BASE_CAT_DTYPE[c])
        for c in GRID_CATS:
            X[c] = X[c].astype(grid_dtypes[c])
        return X

    # 회차/구간 중앙값 (E56 동일)
    def turn_of(fit):
        med = fit[fit.direction == 0].groupby(SEGK).tt_sec.median().reset_index()
        return med.loc[med.groupby('route_id').tt_sec.idxmax(), SEGK].assign(is_turn2=1)

    def add_turn_hint(df, TURN):
        df = df.merge(TURN, on=SEGK, how='left').set_index(df.index)
        df['is_turn2'] = df.is_turn2.fillna(0).astype(int)
        s = df.sort_values(TRIP + ['stop_seq']).copy()
        s['t_prev_gap'] = s.arrive_hour - s.groupby(TRIP).arrive_hour.shift(1)
        b = s.groupby(TRIP + ['arrive_hour'])
        s['t_blk_n']      = b.stop_seq.transform('size')
        s['t_blk_dist']   = b.dist_m.transform('sum')
        s['t_blk_before'] = s.stop_seq - b.stop_seq.transform('min')
        s['t_blk_after']  = b.stop_seq.transform('max') - s.stop_seq
        vc = s.groupby(VD + ['arrive_hour']).size()
        def vcount(shift):
            k = pd.MultiIndex.from_arrays([s.date, s.vehicle_id, s.arrive_hour + shift])
            return vc.reindex(k).fillna(0).values
        s['v_blk_n'], s['v_prev_n'], s['v_next_n'] = vcount(0), vcount(-1), vcount(1)
        tr = s.groupby(TRIP).agg(
            start=('arrive_hour', 'min'), end=('arrive_hour', 'max'),
            first_seq=('stop_seq', 'min')).reset_index()
        first_n = (s.merge(tr[TRIP + ['start']], on=TRIP)
                     .query('arrive_hour == start')
                     .groupby(TRIP).size().rename('n0').reset_index())
        tr = tr.merge(first_n, on=TRIP, how='left')
        nxt = []
        for _, g in tr.groupby(VD):
            g = g.sort_values(['start', 'end'])
            for r in g.itertuples():
                c = g[(g.start >= r.end) &
                      ~((g.direction == r.direction) &
                        (g.trip_no == r.trip_no) &
                        (g.route_id == r.route_id))]
                c = c[c.direction != r.direction] if (c.direction != r.direction).any() else c
                nxt.append((r.date, r.vehicle_id, r.route_id, r.direction, r.trip_no,
                            c.start.min() if len(c) else np.nan,
                            c.sort_values('start').n0.iloc[0] if len(c) else np.nan))
        nxt = pd.DataFrame(nxt, columns=TRIP + ['next_start', 'v_next_trip_n0'])
        s = s.merge(nxt, on=TRIP, how='left').set_index(s.index)
        s['v_next_trip_gap'] = s.next_start - s.arrive_hour
        s.loc[s.is_turn2 == 0, H] = np.nan
        return s.loc[df.index]

    def seg_med(src, apply):
        md = src.groupby(SEGK).tt_sec.median()
        return apply[SEGK].join(md.rename('x'), on=SEGK).x.values

    def add_est(df, med):
        df = df.copy()
        df['_m'] = med
        hs = df.groupby(VD + ['arrive_hour'])._m.sum()
        k = lambda s: pd.MultiIndex.from_arrays(
            [df.date, df.vehicle_id, df.arrive_hour + s])
        inner  = k(-1).isin(hs.index) & k(1).isin(hs.index)
        others = hs.reindex(k(0)).values - np.nan_to_num(df._m.values)
        df['turn_est'] = np.where((df.is_turn2 == 1) & inner, C - others, np.nan)
        return df.drop(columns='_m')

    def prep(fit_raw, apply_raw):
        turn = turn_of(fit_raw)
        f, a = add_turn_hint(fit_raw, turn), add_turn_hint(apply_raw, turn)
        mf = np.full(len(f), np.nan)
        for dd in f.date.unique():
            m = (f.date == dd).values
            mf[m] = seg_med(f[~m], f[m])
        return add_est(f, mf), add_est(a, seg_med(f, a))

    # LightGBM 모델 설정 (기준 wd_lgb와 완전 동일)
    LGB_PARAMS = dict(
        objective='regression_l1',
        learning_rate=0.02,
        num_leaves=127,
        min_child_samples=10,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        n_estimators=1500,
        random_state=0,
        verbose=-1,
        n_jobs=-1,
    )

    def save_ckpt(ckpt_dir, name, row_ids, preds):
        np.savez_compressed(ckpt_dir / f"{name}.npz",
                            row_id=np.asarray(row_ids, dtype=str),
                            pred=np.asarray(preds, dtype=np.float32))

    def load_ckpt(ckpt_dir, name, expected_ids):
        p = ckpt_dir / f"{name}.npz"
        if not p.exists():
            return None
        with np.load(p) as data:
            if not np.array_equal(data['row_id'], np.asarray(expected_ids, dtype=str)):
                print(f"   [경고] {name} row_id 불일치 → 재학습", flush=True)
                return None
            return data['pred'].copy()

    # ─── 3. 날짜별 OOF 학습 및 검증 ──────────────────────────────────────
    WDm = pd.to_datetime(TRAIN.date).dt.day_of_week.values < 5
    val_dates = [
        ('2024-09-23', '월요일'),
        ('2024-09-24', '화요일'),
        ('2024-09-25', '수요일'),
    ]

    print("\n3. 날짜별 OOF 모델 학습 및 비교 (기준: 평일 전용 wd_lgb)", flush=True)
    all_records = []
    grid_stats_rows = []

    for d, day_name in val_dates:
        print(f"\n{'='*64}", flush=True)
        print(f">>> 검증 날짜: {d} ({day_name})", flush=True)
        print(f"{'='*64}", flush=True)

        val_mask     = (TRAIN.date == d).values
        val_df       = TRAIN[val_mask].copy()
        fit_wd_raw   = TRAIN[WDm & ~val_mask]
        expected_ids = val_df['row_id'].values
        y_val        = val_df['tt_sec'].values
        print(f"   학습(fit_wd) 행 수: {len(fit_wd_raw):,} | 검증(val) 행 수: {len(val_df):,}", flush=True)

        # 기준(wd_lgb) 체크포인트 확인 (일치하면 재사용, 없거나 불일치 시 재학습)
        pred_base = load_ckpt(REF_CKPT, f"wd_lgb_{d}", expected_ids)
        pred_grid = load_ckpt(CKPT_DIR, f"wd_lgb_grid_{d}", expected_ids)

        need_prep = (pred_base is None) or (pred_grid is None)
        if need_prep:
            t_prep = time.time()
            print("   prep() 전처리 수행 중...", flush=True)
            f_wd, v_wd = prep(fit_wd_raw, val_df)
            print(f"   prep 완료 ({time.time() - t_prep:.1f}초)", flush=True)

            if pred_base is None:
                t_m = time.time()
                print("   [기준] wd_lgb 체크포인트 없음/불일치 → 기준 모델 재학습 중...", flush=True)
                m_base = lgb.LGBMRegressor(**LGB_PARAMS)
                m_base.fit(to_X_base(f_wd), f_wd.tt_sec)
                pred_base = m_base.predict(to_X_base(v_wd)).astype(np.float32)
                save_ckpt(CKPT_DIR, f"wd_lgb_base_{d}", expected_ids, pred_base)
                print(f"   [기준] 재학습 완료 ({time.time() - t_m:.1f}초)", flush=True)
            else:
                print(f"   [기준] 기존 wd_lgb_{d} 체크포인트 검증 일치 → 재사용", flush=True)
                save_ckpt(CKPT_DIR, f"wd_lgb_base_{d}", expected_ids, pred_base)

            if pred_grid is None:
                # 새 범주형 인코딩(from_grid_id, to_grid_id)은 각 학습 폴드(f_wd)에서만 적합!
                grid_dtypes = {
                    c: pd.CategoricalDtype(sorted(f_wd[c].unique()))
                    for c in GRID_CATS
                }
                unseen_from = len(set(v_wd['from_grid_id']) - set(f_wd['from_grid_id']))
                unseen_to   = len(set(v_wd['to_grid_id'])   - set(f_wd['to_grid_id']))
                print(f"   [격자] 폴드 내 적합 고유 격자 수: from={f_wd['from_grid_id'].nunique()}, "
                      f"to={f_wd['to_grid_id'].nunique()} | 검증 미등장 격자: from={unseen_from}, to={unseen_to}", flush=True)
                grid_stats_rows.append({
                    'date': d, 'day_name': day_name,
                    'fit_rows': len(f_wd), 'val_rows': len(v_wd),
                    'fit_from_grids': int(f_wd['from_grid_id'].nunique()),
                    'fit_to_grids':   int(f_wd['to_grid_id'].nunique()),
                    'val_unseen_from': int(unseen_from),
                    'val_unseen_to':   int(unseen_to),
                })

                t_m = time.time()
                print("   [격자] wd_lgb_grid (38피처) 학습 중...", flush=True)
                X_tr_g = to_X_grid(f_wd, grid_dtypes)
                X_va_g = to_X_grid(v_wd, grid_dtypes)
                m_grid = lgb.LGBMRegressor(**LGB_PARAMS)
                m_grid.fit(X_tr_g, f_wd.tt_sec)
                pred_grid = m_grid.predict(X_va_g).astype(np.float32)
                save_ckpt(CKPT_DIR, f"wd_lgb_grid_{d}", expected_ids, pred_grid)
                print(f"   [격자] 학습 완료 ({time.time() - t_m:.1f}초)", flush=True)
        else:
            print(f"   [기준] 기존 wd_lgb_{d} 체크포인트 재사용", flush=True)
            print(f"   [격자] 기존 wd_lgb_grid_{d} 체크포인트 재사용", flush=True)

        m_b, r_b = mae(y_val, pred_base), rmse(y_val, pred_base)
        m_g, r_g = mae(y_val, pred_grid), rmse(y_val, pred_grid)
        print(f"\n   결과 ({d} {day_name}, {len(y_val):,}행):", flush=True)
        print(f"     기준 (wd_lgb, 36피처)     : MAE {m_b:.3f} | RMSE {r_b:.3f}", flush=True)
        print(f"     격자 추가 (+500m, 38피처) : MAE {m_g:.3f} ({m_g - m_b:+.3f}) | RMSE {r_g:.3f} ({r_g - r_b:+.3f})", flush=True)

        all_records.append(pd.DataFrame({
            'date':         d,
            'day_name':     day_name,
            'row_id':       expected_ids,
            'route_nm':     val_df['route_nm'].values,
            'arrive_hour':  val_df['arrive_hour'].values,
            'from_grid_id': val_df['from_grid_id'].values,
            'to_grid_id':   val_df['to_grid_id'].values,
            'y_true':       y_val,
            'pred_base':    pred_base,
            'pred_grid':    pred_grid,
        }))

    # ─── 4. 전체·날짜별·노선별 비교 저장 ──────────────────────────────────
    oof = pd.concat(all_records, ignore_index=True)
    oof.to_parquet(EXP_DIR / "oof_grid_predictions.parquet", index=False)

    y_all  = oof['y_true'].values
    p_base = oof['pred_base'].values
    p_grid = oof['pred_grid'].values

    print(f"\n{'='*72}", flush=True)
    print(">>> [전체 평일 통합 결과 (120,769행, 기준: wd_lgb)]", flush=True)
    print(f"{'='*72}", flush=True)
    mb_all, rb_all = mae(y_all, p_base), rmse(y_all, p_base)
    mg_all, rg_all = mae(y_all, p_grid), rmse(y_all, p_grid)
    print(f"   기준 (wd_lgb, 36피처)     : MAE {mb_all:.3f} | RMSE {rb_all:.3f}", flush=True)
    print(f"   격자 추가 (+500m, 38피처) : MAE {mg_all:.3f} ({mg_all - mb_all:+.3f}) | RMSE {rg_all:.3f} ({rg_all - rb_all:+.3f})", flush=True)

    # 날짜별 메트릭
    daily_rows = []
    for d, day_name in val_dates:
        m = (oof.date == d).values
        y = y_all[m]
        daily_rows.append({
            'date': d, 'day_name': day_name, 'rows': int(m.sum()),
            'MAE_base': mae(y, p_base[m]),   'MAE_grid': mae(y, p_grid[m]),
            'dMAE': mae(y, p_grid[m]) - mae(y, p_base[m]),
            'RMSE_base': rmse(y, p_base[m]), 'RMSE_grid': rmse(y, p_grid[m]),
            'dRMSE': rmse(y, p_grid[m]) - rmse(y, p_base[m]),
        })
    df_daily = pd.DataFrame(daily_rows)
    df_daily.to_csv(EXP_DIR / "daily_metrics.csv", index=False, encoding='utf-8-sig')
    print("\n[날짜별 비교]", flush=True)
    print(df_daily.to_string(index=False), flush=True)

    # 전체 요약 메트릭
    summary_rows = [
        {
            '모델': '기준 (wd_lgb, 36피처)',
            'Pooled_MAE': mb_all, 'Pooled_RMSE': rb_all,
            'Macro_MAE': float(df_daily['MAE_base'].mean()),
            '월_MAE': float(df_daily.loc[df_daily.date == '2024-09-23', 'MAE_base'].values[0]),
            '화_MAE': float(df_daily.loc[df_daily.date == '2024-09-24', 'MAE_base'].values[0]),
            '수_MAE': float(df_daily.loc[df_daily.date == '2024-09-25', 'MAE_base'].values[0]),
            '기준대비_MAE': 0.0, '기준대비_RMSE': 0.0,
        },
        {
            '모델': '격자 추가 (+500m, 38피처)',
            'Pooled_MAE': mg_all, 'Pooled_RMSE': rg_all,
            'Macro_MAE': float(df_daily['MAE_grid'].mean()),
            '월_MAE': float(df_daily.loc[df_daily.date == '2024-09-23', 'MAE_grid'].values[0]),
            '화_MAE': float(df_daily.loc[df_daily.date == '2024-09-24', 'MAE_grid'].values[0]),
            '수_MAE': float(df_daily.loc[df_daily.date == '2024-09-25', 'MAE_grid'].values[0]),
            '기준대비_MAE': mg_all - mb_all, '기준대비_RMSE': rg_all - rb_all,
        },
    ]
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(EXP_DIR / "summary_metrics.csv", index=False, encoding='utf-8-sig')

    # 노선별 메트릭
    route_rows = []
    for r in sorted(oof.route_nm.unique()):
        m = (oof.route_nm == r).values
        y = y_all[m]
        route_rows.append({
            'route_nm': r, 'rows': int(m.sum()),
            'MAE_base': mae(y, p_base[m]),   'MAE_grid': mae(y, p_grid[m]),
            'dMAE': mae(y, p_grid[m]) - mae(y, p_base[m]),
            'RMSE_base': rmse(y, p_base[m]), 'RMSE_grid': rmse(y, p_grid[m]),
            'dRMSE': rmse(y, p_grid[m]) - rmse(y, p_base[m]),
        })
    df_route = pd.DataFrame(route_rows)
    df_route.to_csv(EXP_DIR / "route_metrics.csv", index=False, encoding='utf-8-sig')
    print("\n[노선별 비교 (3일 통합)]", flush=True)
    print(df_route.to_string(index=False), flush=True)

    # 날짜 x 노선별 메트릭
    dr_rows = []
    for d, day_name in val_dates:
        for r in sorted(oof.route_nm.unique()):
            m = ((oof.date == d) & (oof.route_nm == r)).values
            y = y_all[m]
            dr_rows.append({
                'date': d, 'day_name': day_name, 'route_nm': r, 'rows': int(m.sum()),
                'MAE_base': mae(y, p_base[m]), 'MAE_grid': mae(y, p_grid[m]),
                'dMAE': mae(y, p_grid[m]) - mae(y, p_base[m]),
                'RMSE_base': rmse(y, p_base[m]), 'RMSE_grid': rmse(y, p_grid[m]),
                'dRMSE': rmse(y, p_grid[m]) - rmse(y, p_base[m]),
            })
    pd.DataFrame(dr_rows).to_csv(EXP_DIR / "date_route_metrics.csv", index=False, encoding='utf-8-sig')

    if grid_stats_rows:
        pd.DataFrame(grid_stats_rows).to_csv(EXP_DIR / "grid_stats.csv", index=False, encoding='utf-8-sig')

    elapsed = time.time() - overall_start
    print(f"\n총 소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)", flush=True)
    print("결과 저장: dayun/grid_feature_experiment", flush=True)
    print("GRID_EXPERIMENT_DONE", flush=True)


if __name__ == "__main__":
    main()
