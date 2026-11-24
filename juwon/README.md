# 주원 작업 폴더

버스 통행시간 예측 대회에서 제가 한 실험, 코드, 정리 문서를 모아둔 폴더예요.

## 먼저 볼 파일

- **[2026-11-24_모둠공유_최고모델_정리.md](2026-11-24_모둠공유_최고모델_정리.md)**: 지금까지 정리. 최고 점수는 E58, public MAE 24.309
- [제출메모_final_E60_sunday_refine.md](제출메모_final_E60_sunday_refine.md): 가장 최근 제출 후보 (E58 일요일 모델에 CatBoost 섞음)
- [2026-11-19_전체정리.md](2026-11-19_전체정리.md): 실험 흐름 전체 기록

## 폴더 구조

| 위치 | 내용 |
|---|---|
| `*.md` (이 폴더) | 날짜별 실험기록, 분석, 제출메모, 개념 정리 |
| `그래프/` | 데이터·실험 결과 그래프(png)와 그래프 만드는 코드 |
| `bis-travel-time-practice/scripts/` | 실험 코드 (파일 이름 = 실험 번호, 예: `E60_sunday_refine.py`) |
| `bis-travel-time-practice/src/` | 공통 함수 |
| `bis-travel-time-practice/notebooks/` | 베이스라인, CV 실험 노트북 |
| `bis-travel-time-practice/experiments/` | CV 결과표(csv), 모델 선택 기록(json) |
| `bis-travel-time-practice/submissions/` | 제출 파일 (`final_*` = 대회용, `practice_*` = 실습용) |

## 올리지 않은 것

- 원본 데이터 (`대회_학생용/`, `실습용/`, `bis-travel-time-practice/data/`): 공개 저장소라서 뺐어요. 코드를 다시 돌리려면 데이터를 `bis-travel-time-practice/data/final/`, `bis-travel-time-practice/data/practice/`에 넣으면 돼요.
- 실험 중간 결과 (`*.parquet`), `catboost_info/`, 백업 폴더: 용량이 크고 코드로 다시 만들 수 있어요.
