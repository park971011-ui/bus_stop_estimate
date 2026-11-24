"""E34 회차 구간 규칙 수정(상행 구간 중 train 중앙값이 가장 긴 구간) + 회차 피처 3개 추가. E32와 검증 비교

실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E34_turn_rule.py final   (또는 practice)
결과: submissions/{final|practice}_E34_turn.csv (단계마다 약 5~8분)
주의: 노트북 cv_feature_experiments.ipynb 의 셀(2,4,6,8,10,27,29,41번)을 순서대로 실행해서 피처·모델 함수를 가져옴.
      노트북 셀 순서가 바뀌면 이 스크립트도 고쳐야 함.
"""
# 회차 구간 = 상행(방향 0) 구간 중 train 중앙값이 가장 긴 구간. E32 + 회차 피처(E34) 비교
import json, os, sys, time
import numpy as np, pandas as pd
os.chdir('/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice/notebooks')
PH = sys.argv[1]
nb = json.load(open('cv_feature_experiments.ipynb')); G = {}
for i in [2, 4, 6, 8, 10, 27, 29, 41]:
    src = ''.join(nb['cells'][i]['source'])
    if i == 2: src = __import__('re').sub(r"PHASE = '\w+'", f"PHASE = '{PH}'", src, count=1)
    exec(src, G)
TRAIN, TEST, test = G['TRAIN'], G['TEST'], G['test']
old = G['TURN_SEQ'].to_dict()
d0 = G['train'][G['train'].direction == 0]
NEW = d0.groupby(['route_id', 'stop_seq']).tt_sec.median().reset_index().sort_values('tt_sec').groupby('route_id').tail(1).set_index('route_id')
nm = G['train'].drop_duplicates('route_id').set_index('route_id').route_nm
print(f'[{PH}] 예전 규칙:', {nm[k]: v for k, v in old.items()})
print(f'[{PH}] 새 규칙  :', {nm[k]: (int(r.stop_seq), round(r.tt_sec)) for k, r in NEW.iterrows()}, flush=True)
TS = NEW.stop_seq
for df in (TRAIN, TEST):
    df['turn_seq'] = df.route_id.map(TS)
    df['is_turn'] = ((df.direction == 0) & (df.stop_seq == df.turn_seq)).astype(int)
    df['turn_interval'] = df.is_turn * df.interval
    df['turn_hour'] = np.where(df.is_turn == 1, df.arrive_hour, -1)
LGB = {**G['PARAMS'], 'min_child_samples': 10, 'learning_rate': 0.02, 'n_estimators': 1500}
CB = dict(loss_function='MAE', iterations=2000, learning_rate=0.08, depth=10, l2_leaf_reg=3, random_seed=0, verbose=0, thread_count=-1)
E5 = G['feats_of'](['base', 'seg', 'time', 'trip']); E34 = E5 + ['is_turn', 'turn_interval', 'turn_hour']
cv_of, oof_of = G['cv_of'], G['oof_of']
out = {}
for name, f in [('E32', E5), ('E34', E34)]:
    G['feats'] = f
    if name == 'E32' and PH == 'final' and os.path.exists('../experiments/oof_final_E32.parquet'):
        o = pd.read_parquet('../experiments/oof_final_E32.parquet'); o.index = TRAIN.index; oof = o.e32
    elif name == 'E32' and PH == 'practice':
        o = pd.read_parquet('../experiments/oof_E32.parquet'); o.index = TRAIN.index; oof = o.e32
    else:
        oof = 0.8 * oof_of(G['lgb_fp'], LGB) + 0.2 * oof_of(G['cb_fp'], CB)
    r = cv_of(oof); t = TRAIN.is_turn == 1
    out[name] = r
    print(f"[{PH}] {name} 검증 예상 {r['예상']:.3f} | 평일 {r['평일']:.3f} | 토 {r['토']:.3f} | 회차행 MAE {np.abs(TRAIN.tt_sec[t] - oof[t]).mean():.1f}", flush=True)
    if name == 'E34':
        p = 0.8 * G['lgb_fp'](TRAIN, TEST, LGB) + 0.2 * G['cb_fp'](TRAIN, TEST, CB)
        sub = pd.DataFrame({'row_id': TEST.row_id.values, 'tt_sec': np.round(p, 1)}).set_index('row_id').loc[test.row_id].reset_index()
        assert sub.tt_sec.notna().all() and not sub.row_id.duplicated().any() and len(sub) == len(test)
        sub.to_csv(f'../submissions/{PH}_E34_turn.csv', index=False); print(f'[{PH}] 저장 ../submissions/{PH}_E34_turn.csv', flush=True)
        if 'tt_sec' in test.columns:
            sys.path.append('../src'); from evaluate import score
            for f2 in [f'{PH}_E32.csv', f'{PH}_E34_turn.csv']:
                s = pd.read_csv('../submissions/' + f2); tt = test.assign(p=s.tt_sec.values); D = {3: '목', 4: '금', 6: '일'}
                print(f"[{PH}] 연습 test {f2}: {score(s.copy(), test[['row_id', 'tt_sec']], test)['MAE']:.3f}",
                      {D[pd.Timestamp(d).day_of_week]: round((g.tt_sec - g.p).abs().mean(), 2) for d, g in tt.groupby('date')})
print(f"[{PH}] E34 - E32 = {out['E34']['예상'] - out['E32']['예상']:+.3f}")
