"""E68 회차 시간 양쪽 경계 추정 (탐색용 확인 스크립트)
  회차 행 r (상행 운행 T의 회차 구간, 출발 = a, 도착 = b, 정답 = b - a):
   - a 추정 (왼쪽): 운행 T 안에서 r과 같은 시(h)인 첫 행 f. f 바로 앞 행이 h-1시면 h:00은 그 앞 행 구간 중간쯤 →
                    a ≈ h:00 + 0.5 × 앞 행 평소 시간 + (f ~ r 직전) 평소 시간 합
   - b 추정 (오른쪽): 같은 차량의 다음 운행 N (회차 도착 정류장에서 출발). N의 첫 행과 같은 시(h')인 마지막 행 l, l 다음 행이 h'+1시면
                    b ≈ (h'+1):00 − 0.5 × l 평소 시간 − (N 첫 행 ~ l 직전) 평소 시간 합
   - est = b − a (양쪽 다 있을 때만)
  평소 시간 = 구간 중앙값. 학습 행은 자기 날짜를 뺀 다른 날짜로 (정답 누출 없음)
"""
import json, os, re
import numpy as np, pandas as pd
ROOT = '/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice'
SEGK = ['route_id', 'direction', 'from_stop', 'to_stop']
TRIP = ['date', 'vehicle_id', 'route_id', 'direction', 'trip_no']


def load(ph):
    os.chdir(f'{ROOT}/notebooks')
    nb = json.load(open('cv_feature_experiments.ipynb')); G = {}
    for i in [2, 4, 6]:
        src = ''.join(nb['cells'][i]['source'])
        if i == 2: src = re.sub(r"PHASE = '\w+'", f"PHASE = '{ph}'", src, count=1)
        exec(src, G)
    return G['TRAIN'], G['TEST']


def turn_of(fit):
    med = fit[fit.direction == 0].groupby(SEGK).tt_sec.median().reset_index()
    return med.loc[med.groupby('route_id').tt_sec.idxmax(), SEGK].assign(is_turn2=1)


def boundary_est(df, med, turn):
    """df: trip_no 있는 행들, med: 행마다 평소 시간(구간 중앙값). 회차 행마다 (a_est, b_est, tb_est).
    한 바퀴(상행→하행)는 stop_seq가 이어짐 → 같은 날·차량·trip_no 안에서 stop_seq 순으로 줄 세우고, 순번이 1씩 이어지는 행만 씀."""
    df = df.copy(); df['_m'] = med
    df = df.merge(turn, on=SEGK, how='left').set_index(df.index); df['is_turn2'] = df.is_turn2.fillna(0)
    s = df.sort_values(['date', 'vehicle_id', 'route_id', 'trip_no', 'stop_seq'])
    out = {}
    for _, g in s.groupby(['date', 'vehicle_id', 'route_id', 'trip_no'], sort=False):
        if not (g.is_turn2.values == 1).any():
            continue
        seq, hr, m, ix = g.stop_seq.values, g.arrive_hour.values, g._m.values, g.index.values
        for i in np.where(g.is_turn2.values == 1)[0]:
            h = hr[i]; a = np.nan; la = np.nan; lb = np.nan; acc = 0.0; j = i - 1
            while j >= 0 and seq[j] == seq[j + 1] - 1:            # 왼쪽: 같은 시 구간 평소 시간 더하며 시 경계 찾기
                if hr[j] == h:
                    acc += m[j]; j -= 1; continue
                if hr[j] == h - 1:
                    a = h * 3600 + 0.5 * m[j] + acc; la = 0.5 * m[j] + acc
                break
            b = np.nan
            if i + 1 < len(g) and seq[i + 1] == seq[i] + 1:
                H = hr[i + 1]; acc = 0.0; k = i + 1
                while k + 1 < len(g) and seq[k + 1] == seq[k] + 1:   # 오른쪽: 회차 다음 행부터 같은 시 구간 더하며 다음 시 경계 찾기
                    if hr[k + 1] == H:
                        acc += m[k]; k += 1; continue
                    if hr[k + 1] == H + 1:
                        b = (H + 1) * 3600 - 0.5 * m[k] - acc; lb = 0.5 * m[k] + acc
                    break
            tb = b - a
            if not (0 < tb < 4 * 3600): tb = np.nan          # 순서 매칭 실패(음수·4시간 넘음)는 버림
            out[ix[i]] = (a, b, tb, la, lb)
    return pd.DataFrame.from_dict(out, orient='index', columns=['a_est', 'b_est', 'tb_est', 'tb_left', 'tb_right'])


def seg_med(src, apply):
    md = src.groupby(SEGK).tt_sec.median()
    return apply[SEGK].join(md.rename('x'), on=SEGK).x.values


if __name__ == '__main__':
    TR, TE = load('final')
    turn = turn_of(TR)
    mf = np.full(len(TR), np.nan)
    for dd in TR.date.unique():
        m = (TR.date == dd).values; mf[m] = seg_med(TR[~m], TR[m])
    E = boundary_est(TR, mf, turn)
    d = TR.join(E, how='inner')
    o = pd.read_parquet(f'{ROOT}/experiments/oof_final_E56.parquet').set_index('row_id').e56_41part
    d['p'] = d.row_id.map(o); d['dow'] = pd.to_datetime(d.date).dt.day_of_week
    print(f'회차 행 {len(d)} | a 있음 {d.a_est.notna().mean():.0%} | b 있음 {d.b_est.notna().mean():.0%} | 둘 다 {d.tb_est.notna().mean():.0%}')
    v = d[d.tb_est.notna()]
    print(f'둘 다 있는 행 {len(v)}: 추정값 그대로 MAE {(v.tt_sec - v.tb_est).abs().mean():.1f} | 지금 모델(OOF) MAE {(v.tt_sec - v.p).abs().mean():.1f} | 반반 {(v.tt_sec - 0.5*v.p - 0.5*v.tb_est).abs().mean():.1f}')
    print(f'상관: 실제~추정 {np.corrcoef(v.tt_sec, v.tb_est)[0,1]:.3f} | 모델 잔차~(추정-모델) {np.corrcoef(v.tt_sec - v.p, v.tb_est - v.p)[0,1]:.3f}')
    print(v.groupby(['route_nm']).apply(lambda g: pd.Series({'n': len(g), 'est_mae': (g.tt_sec - g.tb_est).abs().mean(), 'model_mae': (g.tt_sec - g.p).abs().mean(),
          'half': (g.tt_sec - 0.5*g.p - 0.5*g.tb_est).abs().mean(), 'bias_est': (g.tt_sec - g.tb_est).median()})).round(1))
    print(v.groupby('dow').apply(lambda g: pd.Series({'n': len(g), 'est_mae': (g.tt_sec - g.tb_est).abs().mean(), 'model_mae': (g.tt_sec - g.p).abs().mean()})).round(1))
    v[['row_id', 'tt_sec', 'p', 'a_est', 'b_est', 'tb_est']].to_parquet(f'{ROOT}/experiments/E68_check.parquet')
