# ExtraTrees 평일 예측 혼합 실험

## 목적과 비교 방법

기준은 **E56 기반 평일 혼합 모델**입니다. 외부 전체 날짜 조합:평일 전용 모델은 **50:50**, 내부 LightGBM:CatBoost는 **80:20**입니다. 기준 예측은 `0.5 × (0.8 × 전체 날짜 LightGBM + 0.2 × 전체 날짜 CatBoost) + 0.5 × 평일 전용 LightGBM`으로 구성합니다. ExtraTrees 단독과 기준 예측에 ExtraTrees를 10% 또는 20% 섞은 경우를 저장된 예측으로 비교했습니다.

점수는 Phase 1의 월·화·수 날짜별 OOF 검증 및 Phase 2의 연습 데이터 목·금 평가에서 나온 **내부 점수**입니다. **대회 Public MAE가 아닙니다.**

## 핵심 결과

- **Phase 1** (기준 36피처, ExtraTrees 35피처): 기준 MAE 23.235초, 기준 90%+ExtraTrees 10% 23.267초, 기준 80%+ExtraTrees 20% 23.337초였습니다. 혼합은 기준보다 각각 **0.032초**, **0.102초 악화**했습니다.
- **Phase 2 별도 추가 실험** (두 모델 모두 25피처, 연습용 목·금): 기준 26.267초, 기준 90%+ExtraTrees 10% 26.308초로 **0.041초 악화**했습니다. 이 데이터는 이전 실험에서도 사용됐습니다.
- 두 단계에서 혼합 MAE는 기준보다 높았습니다. RMSE는 Phase 1 10% 혼합에서 0.274초, 20% 혼합에서 0.488초, Phase 2 10% 혼합에서 0.277초 낮게 관측됐습니다.

## 결론

이번 설정에서는 혼합 MAE가 악화되어 ExtraTrees 혼합을 채택하지 않았습니다. RMSE 감소는 함께 관측된 결과로 기록하며, 이를 특정 모델 구조의 효과라고 단정하지 않습니다. Phase 1은 기준 36개와 ExtraTrees 35개 피처 및 전처리가 달라 모델 종류만의 효과를 분리할 수 없습니다. Phase 2도 25개 피처를 사용한 별도 실험이므로 Phase 1 설정의 재검증이 아닙니다.

[조건·날짜별 결과·한계가 담긴 상세 보고서](./ExtraTrees_실험_보고서.md)

## 파일 및 재현 입력

- 분석 노트북: `extratrees_experiment.ipynb` (학습 코드와 기존 실행 출력 포함)
- Phase 1 스크립트: `run_et_weekday.py`
- Phase 2 별도 스크립트: `run_et_practice.py`
- Phase 1 저장 예측: `oof_et_predictions.parquet`; 지표 CSV: `summary_metrics.csv`, `daily_metrics.csv`, `route_metrics.csv`, `peak_metrics.csv`
- Phase 2 저장 예측: `practice_thufri_predictions.parquet`; 지표 CSV: `practice_thufri_summary.csv`, `practice_thufri_daily_metrics.csv`, `practice_thufri_route_metrics.csv`, `practice_thufri_peak_metrics.csv`
- 기준 예측 입력: `../weekday_blend_experiment/checkpoints/` 아래의 `all_lgb`, `all_cb`, `wd_lgb` 날짜별 체크포인트 9개
- 데이터 입력: Phase 1 `data/final/train.parquet`, `data/final/test.parquet`; Phase 2 `data/practice/train.parquet`, `data/practice/test.parquet`

저장소 루트에서 Python 환경과 의존성을 준비한 뒤 `python dayun/extratrees_experiment/run_et_weekday.py` 및 `python dayun/extratrees_experiment/run_et_practice.py`로 각 단계를 실행할 수 있습니다. 체크포인트가 없거나 행 ID가 일치하지 않으면 실행 스크립트가 학습할 수 있습니다. 결과 확인은 `jupyter notebook dayun/extratrees_experiment/extratrees_experiment.ipynb`로 합니다. 필요한 패키지는 저장소 루트 `requirements.txt`를 참조하세요.
