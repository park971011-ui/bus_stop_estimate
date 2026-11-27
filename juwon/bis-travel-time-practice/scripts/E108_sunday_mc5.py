"""E108: E73 일요일 모델(피처 FX) 그대로, LightGBM 크기만 키움. 실습 노선 빼기 검증 (E73 cv와 같은 틀)
  E103: 평일에서 num_leaves 127→255(min_child 20)가 4일 LGB -0.13, 평일 전용 -0.05
실행: python scripts/E104_sunday_big.py cv
"""
import sys
ROOT = '/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice'
src = open(f'{ROOT}/scripts/E73_sunday_px.py').read().split("\nFX = F0 + PX")[0]
sys.argv = [sys.argv[0], 'cv']
exec(src)
FX = F0 + PX
VAR = {'mc5_255': dict(num_leaves=255, min_child_samples=5), 'mc5_511': dict(num_leaves=511, min_child_samples=5), 'mc5_127': dict(min_child_samples=5)}
t0 = time.time(); L = []
for r in sorted(PTE.route_nm.unique()):
    m = (PTE.route_nm == r).values
    P_all = pd.concat([PTR, PTE[~m]])
    tr = with_px([FTR, PTR, PTE[~m]], P_all)
    v = with_px([PTE[m & (dow(PTE) == 6)]], P_all)
    out = pd.DataFrame({'row_id': v.row_id.values, 'route': r, 'y': v.tt_sec.values, 'is_turn2': v.is_turn2.values if 'is_turn2' in v else 0})
    for k, ex in VAR.items():
        out[k] = lgb.LGBMRegressor(**{**LGB, **ex}).fit(Xf(tr, FX), tr.tt_sec).predict(Xf(v, FX))
    print(f'노선 {r}: ' + ' | '.join(f'{k} {(out.y - out[k]).abs().mean():.3f}' for k in VAR) + f' ({time.time() - t0:.0f}초)', flush=True)
    L.append(out)
L = pd.concat(L); L.to_parquet(f'{ROOT}/experiments/E108_loro.parquet')
print('== 일요일 전체 ' + ' | '.join(f'{k} {(L.y - L[k]).abs().mean():.3f}' for k in VAR), flush=True)
