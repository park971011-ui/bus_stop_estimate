# ExtraTrees 학습 기준 비교

## 목적

ExtraTrees의 분할 기준(`criterion`)을 `squared_error`에서 `absolute_error`로 바꿨을 때 내부 검증 예측이 달라지는지 비교했다. 두 조건 모두 100개 트리와 같은 35개 피처·전처리·날짜별 검증 폴드를 사용했다.

## 비교 방법

기준 예측은 E56 기반 평일 혼합 모델이다.

```text
0.5 * (0.8 * all_lgb + 0.2 * all_cb) + 0.5 * wd_lgb
```

각 criterion별로 ExtraTrees 단독, 기준 예측에 ET를 10% 섞은 결과, ET를 20% 섞은 결과를 비교했다. 비율은 `기준 예측:ExtraTrees`의 비중이다. 점수는 월·화·수 날짜별 내부 검증 행을 모아 계산한 MAE와 RMSE이며 Public MAE가 아니다.

## 핵심 결과

| ExtraTrees criterion | 비교 예측 | MAE (초) | 기준 대비 MAE 변화 (초) | RMSE (초) | 기준 대비 RMSE 변화 (초) |
|---|---|---:|---:|---:|---:|
| 기준 | E56 평일 혼합 | 23.234862 | 0.000000 | 40.994560 | 0.000000 |
| squared_error | ET 단독 | 24.800603 | +1.565742 | 40.954949 | -0.039612 |
| squared_error | 기준 90% + ET 10% | 23.267000 | +0.032138 | 40.718761 | -0.275799 |
| squared_error | 기준 80% + ET 20% | 23.336918 | +0.102056 | 40.502061 | -0.492499 |
| absolute_error | ET 단독 | 23.572038 | +0.337176 | 42.226837 | +1.232277 |
| absolute_error | 기준 90% + ET 10% | 23.223780 | -0.011081 | 40.915050 | -0.079510 |
| absolute_error | 기준 80% + ET 20% | 23.221988 | -0.012874 | 40.880983 | -0.113577 |

이번 검증에서 `absolute_error` ET 20% 혼합은 기준보다 MAE가 약 0.0129초 낮았다. 차이는 작고, 이 내부 검증 결과만으로 Public 점수 개선이나 일반화 성능 향상을 주장할 수 없다. MAE 개선은 관측됐지만 `absolute_error` 단독 예측의 MAE는 기준보다 높았다.

날짜별·노선별 상세 수치는 [실험 보고서](ExtraTrees_학습기준_비교_실험_보고서.md)에 정리했다.

## 기존 코드와 이번 비교

- 기존 ExtraTrees 실험의 35개 피처 구성·전처리 흐름과 E56 기준 예측을 사용했다.
- 이번 비교에서 변경한 모델 설정은 ExtraTrees의 `criterion`이며 두 조건의 트리 수는 모두 100개다.
- 저장 예측 및 지표: `oof_criterion_predictions.parquet`, `summary_metrics.csv`, `daily_metrics.csv`, `route_metrics.csv`
- 설정 및 재현 스크립트: `run_config.json`, `run_criterion_experiment.py`

## 실행 및 입력

재현 스크립트는 저장소 루트에서 다음처럼 실행한다.

```bash
python dayun/extratrees_mae_experiment/run_criterion_experiment.py --n-estimators 100
```

실행에는 `data/final/train.parquet`, `data/final/test.parquet`, 기존 평일 혼합 실험의 기준 예측 체크포인트 9개와 `run_weekday_blend.py`, 기존 ExtraTrees 실험의 `run_et_weekday.py`와 `oof_et_predictions.parquet`가 필요하다. 실행 명령은 해당 파일들이 로컬 프로젝트에 준비되어 있다는 전제다. 스크립트는 설정이 일치하는 체크포인트를 확인해 재사용한다. 실제 파일 위치와 전처리 세부사항은 스크립트의 base_dir, data_dir 및 build_features를 참조한다.
