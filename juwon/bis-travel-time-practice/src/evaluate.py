"""제출 파일 채점.

사용 (연습용 데이터: 정답이 들어 있는 data/practice/test.csv 로 직접 채점)
    python src/evaluate.py --submission submissions/my_submission.csv --labels data/practice/test.csv

사용 (대회: 담당자가 비공개 정답·구분 파일로 채점)
    python src/evaluate.py --submission submissions/team1.csv --labels final_private/test_labels.csv \
        --test final/test.csv --split-key final_private/split_key.csv

- submission: row_id, tt_sec
- labels:     row_id, tt_sec 가 있는 파일 (다른 컬럼이 있어도 됨)
- test:       (선택) 평가 입력 파일. 있으면 첨두시간 MAE, 노선별 MAE도 계산
- split-key:  (선택) row_id, split(public/private). 있으면 퍼블릭/프라이빗 점수를 따로 계산
"""
import argparse
import sys

import numpy as np
import pandas as pd

PEAK_HOURS = {7, 8, 9, 17, 18, 19}


def check(sub, lab):
    if not {'row_id', 'tt_sec'} <= set(sub.columns):
        sys.exit('오류: 제출 파일에 row_id, tt_sec 컬럼이 있어야 함.')
    if sub.row_id.duplicated().any():
        sys.exit(f'오류: row_id가 중복됨 ({int(sub.row_id.duplicated().sum())}건).')
    missing = set(lab.row_id) - set(sub.row_id)
    if missing:
        sys.exit(f'오류: 예측이 빠진 row_id가 {len(missing)}건 있음.')
    extra = set(sub.row_id) - set(lab.row_id)
    if extra:
        sys.exit(f'오류: 평가에 없는 row_id가 {len(extra)}건 있음.')
    tt = pd.to_numeric(sub.tt_sec, errors='coerce')
    bad = sub.tt_sec.notna() & tt.isna()
    if bad.any():
        ex = sub.loc[bad, 'row_id'].head(3).tolist()
        sys.exit(f'오류: tt_sec에 숫자가 아닌 값이 있음 ({int(bad.sum())}건, 예: {ex}).')
    if tt.isna().any() or not np.isfinite(tt).all():
        sys.exit('오류: tt_sec에 결측이나 무한대 값이 있음.')
    sub['tt_sec'] = tt


def _mae(d):
    return float((d.tt_sec_pred - d.tt_sec_true).abs().mean())


def _rmse(d):
    return float(np.sqrt(((d.tt_sec_pred - d.tt_sec_true) ** 2).mean()))


def score(sub, lab, test=None, split_key=None):
    """점수 계산. 반환: dict (전체, 첨두, 노선별, 퍼블릭/프라이빗)."""
    lab = lab[['row_id', 'tt_sec']]
    d = lab.merge(sub[['row_id', 'tt_sec']], on='row_id', suffixes=('_true', '_pred'))
    out = {'n': len(d), 'MAE': _mae(d), 'RMSE': _rmse(d)}
    if test is not None:
        d = d.merge(test[['row_id', 'route_nm', 'arrive_hour']], on='row_id')
        peak = d.arrive_hour.isin(PEAK_HOURS)
        out['첨두 MAE'] = _mae(d[peak])
        out['비첨두 MAE'] = _mae(d[~peak])
        out['노선별 MAE'] = {str(k): _mae(g) for k, g in d.groupby('route_nm')}
    if split_key is not None:
        d0 = lab.merge(sub[['row_id', 'tt_sec']], on='row_id', suffixes=('_true', '_pred')).merge(split_key, on='row_id')
        for name in ['public', 'private']:
            g = d0[d0.split == name]
            out[f'{name} MAE'] = _mae(g)
            out[f'{name} RMSE'] = _rmse(g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--submission', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--test')
    ap.add_argument('--split-key')
    a = ap.parse_args()
    sub = pd.read_csv(a.submission)
    lab = pd.read_csv(a.labels)
    test = pd.read_csv(a.test) if a.test else None
    if test is None and 'arrive_hour' in lab.columns:        # 연습용 데이터: 정답 파일이 입력 컬럼도 가짐
        test = lab
    key = pd.read_csv(a.split_key) if a.split_key else None
    check(sub, lab)
    r = score(sub, lab, test, key)
    print(f"평가 행 수  : {r['n']:,}")
    print(f"MAE         : {r['MAE']:.3f} 초   (주 지표)")
    print(f"RMSE        : {r['RMSE']:.3f} 초")
    if '첨두 MAE' in r:
        print(f"첨두시간 MAE: {r['첨두 MAE']:.3f} 초")
        print(f"비첨두 MAE  : {r['비첨두 MAE']:.3f} 초")
        print('노선별 MAE  : ' + ', '.join(f'{k}번 {v:.2f}' for k, v in r['노선별 MAE'].items()))
    if 'public MAE' in r:
        print(f"퍼블릭 MAE  : {r['public MAE']:.3f} 초")
        print(f"프라이빗 MAE: {r['private MAE']:.3f} 초   (마감 후 공개)")


if __name__ == '__main__':
    main()
