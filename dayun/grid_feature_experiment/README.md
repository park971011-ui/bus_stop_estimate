# 정류장 500m 격자 지역 피처 실험

정류장 좌표에서 만든 고정 500m 격자 ID(`from_grid_id`, `to_grid_id`)를 평일 전용 LightGBM에 추가한 실험입니다.

> **비교 기준은 전체 혼합 모델이 아니라 평일 전용 LightGBM(`wd_lgb`, 36피처) 자체입니다.** 저장 예측 120,769행 기준, 격자 피처를 추가한 38피처 모델의 Pooled MAE는 **23.266초에서 23.274초로 0.008초 증가**, RMSE는 **41.239초에서 41.324초로 0.084초 증가**했습니다. 이 설정에서는 전체 성능이 소폭 악화됐습니다.

## 구현 범위

- **기존 코드·설정 활용**: 평일 데이터 필터링, 기존 36개 피처와 전처리, LightGBM 설정 및 날짜별 검증 구조를 따랐습니다. 저장된 평일 전용 `wd_lgb` 기준 체크포인트 예측을 재사용했습니다. 기준은 전체 혼합 모델이나 ExtraTrees가 아닙니다.
- **이번 실험에서 추가**: 고정 좌표 변환과 500m 격자 ID 생성, `from_grid_id`·`to_grid_id` 두 범주형 피처의 폴드 내 매핑, 격자 모델 체크포인트와 OOF 예측 저장 및 결과 집계를 추가했습니다.

## 실험 설정

- 격자 기준점: 위도 36.35°, 경도 127.35°, 지구 반지름 6,371,000m
- 국소 미터 좌표 변환 후 `floor(x / 500)`, `floor(y / 500)`으로 격자 생성 (고유 격자 177개)
- 검증일: 2024-09-23(월), 2024-09-24(화), 2024-09-25(수), 하루씩 제외하는 날짜별 3-fold
- 전체 검증 행: 120,769
- 새 범주형 값은 학습 폴드의 범주 목록으로만 매핑했습니다. 각 검증일 미등장 격자는 0개입니다.

## 파일 역할, 입력 파일 및 실행 방법

### 결과 분석 노트북

- 파일: `grid_feature_experiment.ipynb`
- 역할: 기존 `oof_grid_predictions.parquet`을 읽어 전체·날짜·노선·날짜×노선 결과표를 계산합니다. **학습이나 재학습은 수행하지 않습니다.**
- 필요한 입력: `dayun/grid_feature_experiment/oof_grid_predictions.parquet`
- 실행: 프로젝트 루트에서 Jupyter로 이 노트북을 열고 셀을 실행합니다. 선택한 Python 환경에는 `pandas`, `numpy`, `pyarrow`가 필요합니다.

### 실제 격자 실험 학습 스크립트

- 파일: **`run_grid_experiment.py`** (`dayun/grid_feature_experiment/run_grid_experiment.py`)
- 역할: 기존 전처리·평일 전용 `wd_lgb` 설정으로 기준 모델과 격자 추가 모델의 날짜별 3-fold 예측을 만들고, 체크포인트·OOF 예측·결과 CSV를 저장합니다. 저장된 체크포인트가 없거나 `row_id`가 맞지 않으면 해당 모델을 학습합니다.
- 필요한 입력: `data/final/train.parquet`, `data/final/test.parquet`. 기준 예측 체크포인트(`dayun/weekday_blend_experiment/checkpoints/wd_lgb_2024-09-23.npz` 등)가 있으면 검증 후 기준 예측에 재사용하며, 없거나 불일치하면 기준 모델도 학습합니다. LightGBM을 포함해 저장소 루트의 `requirements.txt`에 명시된 Python 패키지가 필요합니다.
- 실행 방법 (PowerShell, 프로젝트 루트에서):

  ```powershell
  conda run -n py312 python dayun/grid_feature_experiment/run_grid_experiment.py
  ```

  이 명령은 누락·불일치한 체크포인트가 있을 때 실제 학습을 수행할 수 있습니다.
### 최종 과제 제출용 재현성

현재 `grid_feature_experiment.ipynb`는 **결과 분석 전용**입니다. 최종 과제 제출에서 학습부터 결과까지 재현해야 하므로, 학습·전처리·검증을 처음부터 수행하는 **별도의 재현용 노트북이 필요합니다.** 현재 학습 로직은 위 Python 스크립트에 보존돼 있습니다.
