"""E52 E46 평일 예측에 "구간(seg_id) × 시간대(arrive_hour) tt_sec 중앙값" 피처 한 열만 추가 (평일 전용 LightGBM에)
  피처 seg_hour_med = (n × 구간·시간대 중앙값 + K × 구간 전체 중앙값) / (n + K),  n = 그 구간·시간대 표본 수, K = 20 (미리 고정)
    구간·시간대 표본이 없으면 구간 전체 중앙값, 구간도 없으면 결측
  날짜 누출 방지: 값을 계산할 행의 날짜는 계산에서 뺌
    - 검증 날짜 행: 학습 날짜 전체로 계산 (예: 수요일 검증 → 월·화)
    - 학습 날짜 행: 다른 학습 날짜로만 계산 (월요일 행 → 화요일, 화요일 행 → 월요일). 구간 전체 중앙값도 같은 규칙
  나머지(E41 피처·하이퍼파라미터·seed 0·혼합 E41 0.5 + 평일 전용 0.5·일요일 E46 값)는 그대로.
  E4와 다른 점: E4는 피처 3개(구간·구간×시간·속도), 키 = 노선·방향·정류장, 표본 5개 미만은 결측(보완 없음), 4일 학습 모델(E3 기반)

  final cv      : 월(화·수로 학습)·화(월·수)·수(월·화) 검증 + 그래프 experiments/E52_segment_errors.png
  practice check: 연습 월·화·수로 학습 → 연습 목·금에서 E46과 한 번 비교
  final make    : 목·금 = 새 예측, 일요일 = E46 → submissions/final_E52_seghour.csv
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E52_seg_hour_median.py final cv
주의: 노트북 cv_feature_experiments.ipynb 의 셀(2,4,6,8,10,27,29,41번)을 실행해서 피처·모델 함수를 가져옴. E46 파일은 안 건드림.
"""
import json, os, re, sys, time
import numpy as np, pandas as pd, lightgbm as lgb
os.chdir('/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice/notebooks')
PH, MODE = sys.argv[1], sys.argv[2]
nb = json.load(open('cv_feature_experiments.ipynb')); G = {}
for i in [2, 4, 6, 8, 10, 27, 29, 41]:
    src = ''.join(nb['cells'][i]['source'])
    if i == 2:
        src = re.sub(r"PHASE = '\w+'", f"PHASE = '{PH}'", src, count=1); assert f"PHASE = '{PH}'" in src
    exec(src, G)
TRAIN, TEST, test, to_X, DAY = G['TRAIN'], G['TEST'], G['test'], G['to_X'], G['DAY']
assert G['PHASE'] == PH
SEGK = ['route_id', 'direction', 'from_stop', 'to_stop']


def turn_of(fit):
    """학습 날짜(fit)만으로 회차 구간: 노선마다 상행 구간 중 중앙값 최대"""
    med = fit[fit.direction == 0].groupby(SEGK).tt_sec.median().reset_index()
    return med.loc[med.groupby('route_id').tt_sec.idxmax(), SEGK].assign(is_turn2=1)


TRIP = ['date', 'vehicle_id', 'route_id', 'direction', 'trip_no']
VD = ['date', 'vehicle_id']          # 차량 하루 (확인: 차량 하루에 노선은 항상 1개)
H = ['t_prev_gap', 't_blk_n', 't_blk_dist', 't_blk_before', 't_blk_after',
     'v_blk_n', 'v_prev_n', 'v_next_n', 'v_next_trip_gap', 'v_next_trip_n0']


def add_turn_hint(df, TURN):
    """회차 구간은 상행의 마지막(또는 끝에서 둘째) 구간이라, 바로 뒤 행은 반대 방향 다음 운행에 있음 → 차량 하루 단위로 봄."""
    df = df.merge(TURN, on=SEGK, how='left').set_index(df.index)
    df['is_turn2'] = df.is_turn2.fillna(0).astype(int)
    s = df.sort_values(TRIP + ['stop_seq']).copy()
    s['t_prev_gap'] = s.arrive_hour - s.groupby(TRIP).arrive_hour.shift(1)     # 같은 운행 바로 앞 구간과 시각 차이
    b = s.groupby(TRIP + ['arrive_hour'])
    s['t_blk_n'] = b.stop_seq.transform('size')                                # 이 운행에서 같은 시에 든 구간 수
    s['t_blk_dist'] = b.dist_m.transform('sum')
    s['t_blk_before'] = s.stop_seq - b.stop_seq.transform('min')
    s['t_blk_after'] = b.stop_seq.transform('max') - s.stop_seq
    # 차량 하루 전체에서 시간별 행 수 (방향 상관없이)
    vc = s.groupby(VD + ['arrive_hour']).size()
    def vcount(shift):
        k = pd.MultiIndex.from_arrays([s.date, s.vehicle_id, s.arrive_hour + shift])
        return vc.reindex(k).fillna(0).values
    s['v_blk_n'], s['v_prev_n'], s['v_next_n'] = vcount(0), vcount(-1), vcount(1)
    # 다음 운행: 이 운행이 끝난 시각 이후에 시작하는 같은 차량의 운행 중 가장 이른 것
    tr = s.groupby(TRIP).agg(start=('arrive_hour', 'min'), end=('arrive_hour', 'max'), first_seq=('stop_seq', 'min')).reset_index()
    first_n = s.merge(tr[TRIP + ['start']], on=TRIP).query('arrive_hour == start').groupby(TRIP).size().rename('n0').reset_index()
    tr = tr.merge(first_n, on=TRIP, how='left')
    nxt = []
    for _, g in tr.groupby(VD):
        g = g.sort_values(['start', 'end'])
        for r in g.itertuples():
            c = g[(g.start >= r.end) & ~((g.direction == r.direction) & (g.trip_no == r.trip_no) & (g.route_id == r.route_id))]
            c = c[c.direction != r.direction] if (c.direction != r.direction).any() else c
            nxt.append((r.date, r.vehicle_id, r.route_id, r.direction, r.trip_no,
                        c.start.min() if len(c) else np.nan, c.sort_values('start').n0.iloc[0] if len(c) else np.nan))
    nxt = pd.DataFrame(nxt, columns=TRIP + ['next_start', 'v_next_trip_n0'])
    s = s.merge(nxt, on=TRIP, how='left').set_index(s.index)
    s['v_next_trip_gap'] = s.next_start - s.arrive_hour                         # 다음 운행 시작까지 몇 시간
    s.loc[s.is_turn2 == 0, H] = np.nan                                           # 회차 구간 행에만 값
    return s.loc[df.index]

F = G['feats_of'](['base', 'seg', 'time', 'trip']) + H
K = 20
LGB = {**G['PARAMS'], 'min_child_samples': 10, 'learning_rate': 0.02, 'n_estimators': 1500}
dow = lambda df: pd.to_datetime(df.date).dt.day_of_week.values
WD = TRAIN[dow(TRAIN) < 5]
print(f'[{PH}] 학습 평일 날짜: {sorted(WD.date.unique())}', flush=True)


def seg_hour_med(src, apply):
    """src 행으로 계산한 (구간×시간 중앙값, 구간 중앙값)을 K로 섞어 apply에 붙임"""
    g = src.groupby(['seg_id', 'arrive_hour']).tt_sec.agg(['median', 'size'])
    s = src.groupby('seg_id').tt_sec.median()
    k = pd.MultiIndex.from_arrays([apply.seg_id, apply.arrive_hour])
    m, n = g['median'].reindex(k).values, g['size'].reindex(k).fillna(0).values
    sm = s.reindex(apply.seg_id).values
    return np.where(n > 0, (n * np.nan_to_num(m) + K * sm) / (n + K), sm)


def add_feat(fit, apply):
    f, a = fit.copy(), apply.copy()
    a['seg_hour_med'] = seg_hour_med(fit, apply)
    f['seg_hour_med'] = np.nan
    for d in f.date.unique():                       # 학습 행은 자기 날짜를 뺀 다른 학습 날짜로
        m = (f.date == d).values
        f.loc[m, 'seg_hour_med'] = seg_hour_med(fit[fit.date != d], fit[m])
    return f, a


def fit_pred(fit, apply):
    turn = turn_of(fit)
    f, a = add_turn_hint(fit, turn), add_turn_hint(apply, turn)
    f, a = add_feat(f, a)
    feats = F + ['seg_hour_med']
    return lgb.LGBMRegressor(**LGB).fit(to_X(f, feats), f.tt_sec).predict(to_X(a, feats)), a


def plot(R, turn_all):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'AppleGothic'; plt.rcParams['axes.unicode_minus'] = False
    R = R.copy(); R['ae'] = (R.tt_sec - R.old).abs()
    tk = set(map(tuple, turn_all[SEGK].values)); R['turn'] = [tuple(x) in tk for x in R[SEGK].values]
    mt = R[R.day.isin(['월', '화'])]
    s = mt.groupby(['seg_id', 'route_nm', 'direction', 'turn']).ae.sum().sort_values(ascending=False).reset_index()
    pick = list(s[~s.turn].head(2).itertuples()) + list(s[s.turn].head(1).itertuples())
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharey='row')
    for i, p in enumerate(pick):
        for j, day in enumerate(['월', '화', '수']):
            ax = axes[i, j]; x = R[(R.seg_id == p.seg_id) & (R.route_nm == p.route_nm) & (R.direction == p.direction) & (R.day == day)]
            g = x.groupby('arrive_hour').agg(n=('tt_sec', 'size'), q1=('tt_sec', lambda v: v.quantile(.25)), med=('tt_sec', 'median'),
                                              q3=('tt_sec', lambda v: v.quantile(.75)), old=('old', 'median'), new=('new', 'median'))
            ax.fill_between(g.index, g.q1, g.q3, color='tab:blue', alpha=.15, label='실제 25~75%')
            ax.plot(g.index, g.med, 'o-', color='tab:blue', label='실제 중앙값')
            ax.plot(g.index, g.old, 's--', color='tab:red', label='E46 예측 중앙값')
            ax.plot(g.index, g.new, '^:', color='tab:green', label='E52 예측 중앙값')
            for h, r in g.iterrows():
                ax.annotate(f'{int(r.n)}', (h, r.q3), textcoords='offset points', xytext=(0, 4), ha='center', fontsize=7, color='gray')
            kind = '회차 구간' if p.turn else '일반 구간'
            ax.set_title(f'{p.route_nm}번 {"상행" if p.direction == 0 else "하행"} {kind} ({day})\n월·화 절대오차 합 {p.ae:,.0f}초', fontsize=10)
            ax.set_xlabel('도착 시간대 (시)'); ax.grid(alpha=.3)
            if j == 0: ax.set_ylabel('통행시간 (초)')
            if i == 0 and j == 0: ax.legend(fontsize=8)
    fig.suptitle('E46이 반복해서 틀리는 구간 (회색 숫자 = 시간대별 표본 수, 고른 기준: 월·화 절대오차 합)', fontsize=13)
    fig.tight_layout(); out = '../experiments/E52_segment_errors.png'; fig.savefig(out, dpi=110); print(f'[{PH}] 그래프 저장 {out}', flush=True)
    for p in pick: print(f'[{PH}] 고른 구간: {p.route_nm}번 방향 {p.direction} seg {p.seg_id} 회차={p.turn} 월·화 절대오차 합 {p.ae:,.0f}', flush=True)


if MODE == 'cv':
    old41 = pd.read_parquet(f'../experiments/oof_{PH}_E41.parquet').set_index('row_id')
    old44 = pd.read_parquet(f'../experiments/oof_{PH}_E44.parquet').set_index('row_id')
    rows, parts = [], []
    t0 = time.time()
    for d in sorted(WD.date.unique()):
        fit, val = WD[WD.date != d], WD[WD.date == d]
        pN, a = fit_pred(fit, val)
        ids = val.row_id.values; y = val.tt_sec.values
        old = 0.5 * old41.loc[ids].e41.values + 0.5 * old44.loc[ids].wd_lgb.values
        new = 0.5 * old41.loc[ids].e41.values + 0.5 * pN
        tr = a.is_turn2.values == 1; name = DAY[pd.Timestamp(d).day_of_week]
        ao, an = np.abs(y - old), np.abs(y - new)
        rows.append({'날짜': name, '행 수': len(val), '기존 MAE': ao.mean(), '새 MAE': an.mean(), '개선폭(기존−새)': ao.mean() - an.mean(),
                     '일반 구간 개선폭': ao[~tr].mean() - an[~tr].mean(), '회차 구간 개선폭': ao[tr].mean() - an[tr].mean(),
                     '피처 결측 %': a.seg_hour_med.isna().mean() * 100})
        parts.append(val.assign(day=name, old=old, new=new, turn2=tr))
        print(f'[{PH}] {name} 끝 (학습 {sorted(fit.date.unique())}, {time.time() - t0:.0f}초)', flush=True)
    R = pd.concat(parts)
    r = pd.DataFrame(rows).set_index('날짜')
    ao, an = (R.tt_sec - R.old).abs(), (R.tt_sec - R.new).abs(); tr = R.turn2.values
    r.loc['세 날짜 평균'] = r.mean(numeric_only=True); r.loc['세 날짜 평균', '행 수'] = np.nan
    r.loc['전체 평일 행'] = [len(R), ao.mean(), an.mean(), ao.mean() - an.mean(), ao[~tr].mean() - an[~tr].mean(), ao[tr].mean() - an[tr].mean(), np.nan]
    print(r.round(3).to_string(), flush=True)
    imp = r.loc['세 날짜 평균', '개선폭(기존−새)']; days = int((r.iloc[:3]['개선폭(기존−새)'] > 0).sum())
    share = (dow(TEST) < 5).mean()
    print(f'[{PH}] 기준(E46 평일) 세 날짜 평균 {r.loc["세 날짜 평균", "기존 MAE"]:.3f} (기존 기록 23.261) | 개선 날짜 {days}/3', flush=True)
    print(f'[{PH}] 전체 MAE 효과 추정: 개선폭 {imp:+.3f} × 대회 test 평일 비율 {share:.4f} = {imp * share:+.3f}초 (검증 기반 추정치, + = 좋아짐)', flush=True)
    r.to_csv('../experiments/E52_cv_table.csv')
    plot(R, turn_of(WD))
else:
    pN, a = fit_pred(WD, TEST)
    e41 = pd.read_csv(f'../submissions/{PH}_E41_turnhint_sunmix.csv').set_index('row_id').tt_sec
    e46 = pd.read_csv(f'../submissions/{PH}_E46_wd_route_sunmix.csv').set_index('row_id').tt_sec
    TE = TEST.assign(N=pN, E41=e41.loc[TEST.row_id.values].values, E46=e46.loc[TEST.row_id.values].values)
    wd = dow(TE) < 5
    new = np.where(wd, np.round(0.5 * TE.E41 + 0.5 * TE.N, 1), TE.E46)
    if MODE == 'check':
        y = TE.tt_sec.values
        for name, p in [('E46', TE.E46.values), ('E52', new)]:
            aa = np.abs(y - p)
            print(f'[{PH}] {name} 연습 목·금 {aa[wd].mean():.3f} | 목 {aa[dow(TE) == 3].mean():.3f} | 금 {aa[dow(TE) == 4].mean():.3f}', flush=True)
        print(f'[{PH}] 개선폭 (E46 − E52, 목·금) = {np.abs(y - TE.E46.values)[wd].mean() - np.abs(y - new)[wd].mean():+.3f}', flush=True)
    elif MODE == 'make':
        sub = pd.DataFrame({'row_id': TE.row_id.values, 'tt_sec': new}).set_index('row_id').loc[test.row_id].reset_index()
        ss = pd.read_csv(f'../data/{PH}/sample_submission.csv')
        assert len(sub) == len(ss) and (sub.row_id.values == ss.row_id.values).all(), 'row_id 순서·행 수'
        assert sub.tt_sec.notna().all() and not sub.row_id.duplicated().any() and (sub.tt_sec >= 0).all(), '결측·중복·음수'
        sid = TE.row_id[~wd]; assert (sub.set_index('row_id').tt_sec.loc[sid].values == e46.loc[sid].values).all(), '일요일 = E46'
        path = f'../submissions/{PH}_E52_seghour.csv'; assert not os.path.exists(path)
        sub.to_csv(path, index=False)
        print(f'[{PH}] 저장 {path} {sub.shape} | row_id 순서·행 수 = sample_submission | 결측·중복·음수 0 | 일요일 {(~wd).sum():,}행 E46과 같음', flush=True)
