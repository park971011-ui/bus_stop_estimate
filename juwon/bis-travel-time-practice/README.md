# 대전광역시 BIS 데이터를 활용한 노선별 버스 도착시간 예측(ETA)

## 1. 개요

1) 목표: 버스가 한 정류장에 도착한 뒤 다음 정류장에 도착할 때까지 걸린 시간(`tt_sec`, 초)을 예측함

2) 범위
    - 기간: 2024-09-23(월) ~ 09-29(일), 7일. 학습은 월·화·수·토, 평가는 목·금·일
    - 예측 단위: 정류장 → 바로 다음 정류장 한 구간
    - 노선: 대회 5개 노선. 연습용 데이터는 이와 겹치지 않는 4개 노선
    - 모델: 앙상블 계열만 사용(XGBoost, LightGBM, CatBoost, RandomForest 등)

3) 데이터 소개
    - 한 행: **버스 한 대가 정류장 A에 도착해서 다음 정류장 B에 도착하기까지**의 기록 하나임
    - 파생 변수: 요일, 주말 여부 같은 파생 변수는 없으므로 필요하면 직접 만들어야 함
    - 정답: `tt_sec`는 원본의 두 도착 시각(초 단위) 차이로 계산함

## 2. 파일 구성

1) 연습용 데이터 (`data/practice/`): 노트북 사용법을 익히는 용도
    - `train.csv` / `train.parquet`: 학습용. 입력 + 정답(`tt_sec`), 109,891행
    - `test.csv` / `test.parquet`: 평가용. 입력 + 정답(스스로 채점해 보는 용도), 79,942행
    - 노선: **102, 511, 605, 1번**
    - 형식: CSV와 parquet은 내용이 같음. parquet이 작고 빨라서 노트북은 parquet을 읽음

2) 대회 데이터 (`data/final/`, 대회 시작 때 공개)
    - `train`: 학습용. 입력 + 정답, 154,327행
    - `test`: 평가용. 입력만(**정답 없음**), 111,563행
    - `sample_submission.csv`: 제출 형식 예시(모든 행에 학습 평균을 채움), 111,563행
    - 노선: 5개. 연습용 노선과 겹치지 않으며 유형·길이가 서로 다름
    - 제출 형식: `row_id`, `tt_sec` 두 컬럼(평가 행마다 예측한 통행시간, 초)

## 3. 컬럼 설명

| 컬럼 | 의미 |
| --- | --- |
| `row_id` | 행 ID (연습용 `PTR..`/`PTE..`, 대회 `FTR..`/`FTE..`) |
| `route_id` | 노선 ID |
| `route_nm` | 노선 번호 |
| `route_type` | 노선 유형 코드: 1 급행, 2 간선, 3 지선, 4 외곽, 5 마을, 6 첨단, 7 광역 |
| `vehicle_id` | 차량 ID |
| `direction` | 상하행 구분: 0 상행, 1 하행 |
| `interval_wd`, `interval_sat`, `interval_sun` | 배차간격(분): 평일, 토요일, 일요일 |
| `first_time`, `last_time` | 평일 기점 기준 첫차·막차 출발 시각(HHMM, 예: 545 = 05:45) |
| `from_stop`, `from_name`, `from_lat`, `from_lon` | 출발 정류장 ID, 이름, 위도, 경도 |
| `to_stop`, `to_name`, `to_lat`, `to_lon` | 도착 정류장 ID, 이름, 위도, 경도 |
| `stop_seq` | 출발 정류장이 노선(해당 방향)에서 몇 번째 정류장인지 |
| `dist_m` | 두 정류장 사이 거리(m) |
| `date` | 출발 정류장 도착 날짜 (YYYY-MM-DD) |
| `arrive_hour` | 출발 정류장 도착 **시**(0~23) |
| **`tt_sec`** | **정답.** 출발 정류장 도착 → 도착 정류장 도착 소요 시간(초). 출발 정류장 정차시간 포함 |

## 4. 시작하기

```bash
git clone https://github.com/camuslab/bis-travel-time-practice
cd bis-travel-time-practice
```

- **Python 3.12 환경이 필요함.** 3.11 이하이거나 3.15에서는 패키지 설치가 실패함. 수업용 콘다 환경을 새로 만들어 씀:
    ```bash
    conda create -n bis python=3.12 -y
    ```
    VS Code에서 노트북을 열고 오른쪽 위 커널 선택에서 `bis`를 고름
- `notebooks/baseline_lightgbm.ipynb`를 VS Code에서 열고, 사용할 콘다 환경을 커널로 고른 뒤 위에서부터 실행함
    - 첫 코드 셀(1-0)이 `requirements.txt`의 패키지를 설치함. 처음 한 번만 실행하고 커널을 재시작함
    - 다음 셀(1-1)이 허깅페이스에서 데이터를 `data/practice/`로 내려받음
- 데이터는 용량 때문에 이 레포에 포함하지 않고 허깅페이스에서 받음: [연습용 데이터](https://huggingface.co/datasets/CAMUS-LAB/bis-travel-time-practice)
- 맥에서 LightGBM을 불러올 때 `libomp.dylib` 오류가 나면 `brew install libomp`를 한 번 실행함
- 노트북 구성
    - **Part 1. 데이터**: 패키지 설치 → 허깅페이스에서 받기 → 불러오기 → 둘러보기
    - **Part 2. 모델**: 날짜 단위 검증 → 평균 예측 → 피처 만들기 → LightGBM → 제출 파일 만들기와 채점
- 연습용 데이터로 한 번 끝까지 돌려 본 뒤, 1-1 셀의 `PHASE`를 `'final'`로 바꾸면 대회 데이터로 같은 노트북이 돌아감

## 5. 폴더 구성

```
bis-travel-time-practice/
├─ README.md            과제 안내 (이 문서)
├─ EVALUATION.md        평가 기준, 제출 방법, 리더보드 규칙
├─ requirements.txt
├─ data/                데이터 폴더 (내려받은 파일이 들어감, git에는 올리지 않음)
│  ├─ practice/         연습용 데이터 (train, test)
│  └─ final/            대회 데이터 (대회 시작 때 받음)
├─ src/
│  └─ evaluate.py       제출 파일 채점 (MAE, RMSE, 첨두시간, 노선별)
├─ notebooks/
│  └─ baseline_lightgbm.ipynb   실습 코드 (Part 1 데이터, Part 2 모델)
└─ submissions/         제출 파일 저장 위치
```

## 6. 제출 방법

1) 예측 결과를 `row_id`, `tt_sec` 두 컬럼의 CSV로 저장함

2) `test`의 모든 `row_id`에 예측값이 하나씩 있어야 함. `row_id`가 빠지거나 같은 `row_id`가 두 번 나오면 채점되지 않음 (제출 횟수에는 세지 않음)

3) 연습용 데이터: `test`에 정답이 들어 있으니 직접 채점해 봄
    ```bash
    python src/evaluate.py --submission submissions/my_submission.csv --labels data/practice/test.csv
    ```

4) 대회 데이터: `test`에 정답이 없음. 제출 방식과 리더보드 규칙은 [EVALUATION.md](EVALUATION.md)를 참고함