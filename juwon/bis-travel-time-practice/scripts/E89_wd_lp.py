"""E89 목·금 = E79 + E87 피처 (일요일 = E73 그대로, E88에서 일요일엔 채택 안 됨)
  변형 D : + lp_D (회차 역산)          E87 대회 25.080 → 25.060 (평일 2/3), 연습 목·금 26.076 → 26.060
  변형 MD: + lp_min, lp_tod, lp_D      E87 대회 25.080 → 25.071 (평일 2/3), 연습 목·금 26.076 → 25.984
  방식은 E79와 같음: 0.5 × (LGB 0.8 + CB 0.2, 4일 학습) + 0.5 × 평일 전용 LGB
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E89_wd_lp.py final D   (또는 MD)
"""
import sys
VAR = sys.argv[2]
ROOT = '/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice'
src = open(f'{ROOT}/scripts/E79_cross_wd.py').read()
head, body = src.split('\nt0 = time.time()')
exec(head)
sys.path.insert(0, f'{ROOT}/scripts')
from lp_dwell import lp_dwell
FE = FE + {'D': ['lp_D'], 'MD': ['lp_min', 'lp_tod', 'lp_D']}[VAR]


def add_lp(d, med):
    r = lp_dwell(d, med, 0.3, 0.05); d = d.copy()
    d['lp_min'] = r.lp_min.values; d['lp_tod'] = d.arrive_hour + d.lp_min / 60; d['lp_D'] = r.lp_D.values
    return d


def prep(fit_raw, apply_raw):
    turn = turn_of(fit_raw)
    f, a = add_turn_hint(fit_raw, turn), add_turn_hint(apply_raw, turn)
    mf = np.full(len(f), np.nan)
    for dd in f.date.unique():
        m = (f.date == dd).values; mf[m] = seg_med(f[~m], f[m])
    ma = seg_med(f, a)
    return add_lp(add_cross(add_est(f, mf)), mf), add_lp(add_cross(add_est(a, ma)), ma)


body = body.replace("'../submissions/final_E79_cross_wd_E73sun.csv'", f"'../submissions/final_E89_wd_lp{VAR}_E73sun.csv'")
assert f'final_E89_wd_lp{VAR}' in body
exec('t0 = time.time()' + body)
