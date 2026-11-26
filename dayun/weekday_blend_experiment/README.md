# 평일 혼합 비율 실험

E56 기반 평일 혼합 모델의 두 가지 가중치 선택을 저장된 예측으로 비교한 실험입니다. 검증 기준은 **2024년 9월 월·화·수 날짜별 MAE**이며, 대회 Public MAE가 아닙니다.

기준 모델은 전체 날짜 조합과 평일 전용 LightGBM을 50:50으로 섞은 기존 구성입니다. 전체 날짜 조합 내부는 LightGBM:CatBoost 80:20입니다.

## 핵심 결과와 결론

- 전체 날짜 조합:평일 전용 모델 비율 비교에서 30:70은 50:50보다 통합 MAE가 약 **0.001초** 낮았지만, 날짜별로는 화요일만 개선됐고 월·수요일은 악화됐습니다. RMSE는 약 0.062초 높았습니다. 70:30은 3일 MAE 모두 악화했습니다.
- 전체 날짜 조합 내부의 LightGBM:CatBoost 비율 비교에서는 90:10의 통합 MAE가 기존 80:20보다 약 **0.002초** 낮았습니다. 그러나 월요일은 악화했고, RMSE도 약 0.050초 높았습니다. 100:0도 차이는 약 0.001초였고 날짜별 결과가 달랐습니다.
- 관측된 MAE 이득은 작고 날짜별 방향이 일관되지 않아 **기존 외부 50:50, 내부 80:20 설정을 유지**했습니다. 이는 시험한 조건의 결과에 따른 실무 선택이며, 어떤 비율이 최적이라고 입증한 것은 아닙니다.

[두 실험의 조건·날짜별 결과·해석을 담은 상세 보고서](./혼합비율_실험_보고서.md)

## 비율의 뜻

1. **외부 혼합 비율**은 `전체 날짜 학습 조합 : 평일 전용 LightGBM`의 가중치입니다. 전체 날짜 조합은 평일 전용이 아닌 전체 학습 날짜로 만든 모델 묶음이며, 그 안의 LGB:CB 비율은 80:20입니다.
2. **내부 혼합 비율**은 전체 날짜 조합 안의 `LightGBM : CatBoost` 가중치입니다. 이 실험에서는 외부 비율을 50:50으로 고정했습니다.

## 코드·입력·실행

- 기존 모델 학습 및 외부 비율 결과 생성 코드: `run_weekday_blend.py`
- 저장 예측에서 내부 비율을 집계하는 코드: `eval_inner_ratios.py` (모델 학습 없음)
- 외부 비율 재계산 입력: `oof_blend_predictions.parquet`
- 내부 비율 재계산 입력: `data/final/train.parquet`의 정답·행 ID와 `checkpoints/`의 `all_lgb`, `all_cb`, `wd_lgb` 날짜별 예측 9개
- 모델 예측을 새로 만드는 실행은 프로젝트 루트에서 `python dayun/weekday_blend_experiment/run_weekday_blend.py`입니다. 필요한 학습 데이터는 `data/final/train.parquet`, `data/final/test.parquet`이며, 체크포인트가 없거나 행 ID가 맞지 않으면 학습할 수 있습니다.
- 저장 예측으로 내부 비율을 다시 집계할 때는 `python dayun/weekday_blend_experiment/eval_inner_ratios.py`를 사용합니다. 이 스크립트는 `inner_ratio_summary.csv`를 다시 씁니다. 두 스크립트의 `BASE_DIR`은 작성 환경의 저장소 위치를 가리키므로, 다른 clone에서 실행하려면 해당 값을 그 clone의 저장소 루트로 맞춰야 합니다. 필요한 Python 패키지는 NumPy, Pandas, PyArrow, LightGBM, CatBoost입니다.

상세 수치와 검증 절차는 보고서를 참조하세요.
