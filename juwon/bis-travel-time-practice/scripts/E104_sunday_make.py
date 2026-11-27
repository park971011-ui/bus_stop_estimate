"""E104 일요일 최종: E73과 같음(LGB 0.8 + CB 0.2, 배차 다른 노선 일요일 x2) 단 LGB = (기본 + num_leaves 511) 평균
  검증(E104 실습 노선 빼기, LGB만): 기본 27.340 → 평균 27.238 (3/4 노선)
결과: experiments/E104_sunday_pred.parquet (row_id, tt_sec)
"""
import sys
ROOT = '/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice'
src = open(f'{ROOT}/scripts/E73_sunday_px.py').read().split("\nFX = F0 + PX")[0]
sys.argv = [sys.argv[0], 'make']
exec(src)
FX = F0 + PX
from catboost import CatBoostRegressor
P_all = pd.concat([PTR, PTE])
ALL = with_px([FTR, PTR, PTE], P_all)
sun = dow(FTE) == 6; FS = with_px([FTE[sun]], P_all)
diff = (FS.interval_sun != FS.interval_sat).values
BIG = dict(num_leaves=511, min_child_samples=20)
t0 = time.time()


def fit(ws):
    w = np.where(dow(ALL) == 6, ws, 1.0)
    m1 = lgb.LGBMRegressor(**LGB).fit(Xf(ALL, FX), ALL.tt_sec, sample_weight=w)
    m2 = lgb.LGBMRegressor(**{**LGB, **BIG}).fit(Xf(ALL, FX), ALL.tt_sec, sample_weight=w)
    mc = CatBoostRegressor(**CB, cat_features=['route_id', 'seg_id']).fit(ALL[FX], ALL.tt_sec, sample_weight=w)
    print(f'fit ws={ws} ({time.time() - t0:.0f}초)', flush=True)
    return lambda X: 0.8 * 0.5 * (m1.predict(Xf(X, FX)) + m2.predict(Xf(X, FX))) + 0.2 * mc.predict(X[FX])


p = fit(1.0)(FS); p[diff] = fit(2.0)(FS[diff])
out = pd.DataFrame({'row_id': FS.row_id.values, 'tt_sec': p})
out.to_parquet(f'{ROOT}/experiments/E104_sunday_pred.parquet', index=False)
e89 = pd.read_csv(f'{ROOT}/submissions/final_E89_wd_lpMD_E73sun.csv').set_index('row_id').tt_sec
d = p - e89.loc[FS.row_id.values].values
print(f'== 일요일 저장 {len(out)}행 | E89(E73) 대비 평균 {d.mean():+.2f}초, |차이| {np.abs(d).mean():.2f}초', flush=True)
