from pathlib import Path
import pandas as pd
import gc


# ============================================================
# 1. 경로 설정
# ============================================================

# 현재 파일 위치:
# LS Project 1/src/data_test.py
#
# parents[0] = src
# parents[1] = LS Project 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = PROJECT_ROOT / "data" / "train"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# processed 폴더가 없으면 생성
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT :", PROJECT_ROOT)
print("TRAIN_DIR    :", TRAIN_DIR)
print("PROCESSED_DIR:", PROCESSED_DIR)

assert TRAIN_DIR.exists(), f"TRAIN_DIR이 존재하지 않습니다: {TRAIN_DIR}"


# ============================================================
# 2. 01~10 / M01~M02 데이터 처리
# ============================================================

for i in range(1, 11):

    num = f"{i:02d}"

    for machine in ["M01", "M02"]:

        key = f"{num}_{machine}"

        print()
        print("=" * 60)
        print(f"[START] {key}")
        print("=" * 60)


        # ----------------------------------------------------
        # 3. 파일 경로
        # ----------------------------------------------------

        sensor_path = (
            TRAIN_DIR
            / f"{num}_{machine}_DC_train.csv"
        )

        ttf_path = (
            TRAIN_DIR
            / "train_ttf"
            / f"{num}_{machine}_DC_train_ttf.csv"
        )

        output_path = (
            PROCESSED_DIR
            / f"{key}_sensor_ttf.parquet"
        )


        # ----------------------------------------------------
        # 4. 이미 만들어진 파일이면 건너뛰기
        # ----------------------------------------------------

        if output_path.exists():
            print(f"[SKIP] 이미 존재함: {output_path.name}")
            continue


        # ----------------------------------------------------
        # 5. 파일 존재 여부 확인
        # ----------------------------------------------------

        if not sensor_path.exists():
            raise FileNotFoundError(
                f"센서 파일이 없습니다:\n{sensor_path}"
            )

        if not ttf_path.exists():
            raise FileNotFoundError(
                f"TTF 파일이 없습니다:\n{ttf_path}"
            )


        # ----------------------------------------------------
        # 6. 데이터 읽기
        # ----------------------------------------------------

        print("[LOAD] Sensor")

        sensor = pd.read_csv(sensor_path)

        print(
            f"       shape = {sensor.shape}"
        )


        print("[LOAD] TTF")

        ttf = pd.read_csv(ttf_path)

        print(
            f"       shape = {ttf.shape}"
        )


        # ----------------------------------------------------
        # 7. 행 수 검증
        # ----------------------------------------------------

        if len(sensor) != len(ttf):

            raise ValueError(
                f"{key}: Sensor와 TTF의 행 수가 다릅니다.\n"
                f"Sensor = {len(sensor):,}\n"
                f"TTF    = {len(ttf):,}"
            )


        # ----------------------------------------------------
        # 8. time 컬럼 존재 여부 확인
        # ----------------------------------------------------

        if "time" not in sensor.columns:
            raise ValueError(
                f"{key}: Sensor에 time 컬럼이 없습니다."
            )

        if "time" not in ttf.columns:
            raise ValueError(
                f"{key}: TTF에 time 컬럼이 없습니다."
            )


        # ----------------------------------------------------
        # 9. Sensor와 TTF의 time이 행별로 동일한지 검증
        # ----------------------------------------------------

        sensor_time = sensor["time"].reset_index(drop=True)
        ttf_time = ttf["time"].reset_index(drop=True)

        if not sensor_time.equals(ttf_time):

            raise ValueError(
                f"{key}: Sensor와 TTF의 time 순서가 일치하지 않습니다."
            )

        print("[CHECK] 행 수 일치")
        print("[CHECK] time 전체 일치")


        # ----------------------------------------------------
        # 10. concat
        #
        # Sensor의 time은 유지
        # TTF의 time은 중복되므로 제거
        # ----------------------------------------------------

        sensor = sensor.reset_index(drop=True)

        ttf = (
            ttf
            .drop(columns=["time"])
            .reset_index(drop=True)
        )

        df = pd.concat(
            [
                sensor,
                ttf
            ],
            axis=1
        )


        # ----------------------------------------------------
        # 11. 결과 확인
        # ----------------------------------------------------

        print(
            f"[CONCAT] {key}: "
            f"{sensor.shape} + "
            f"{ttf.shape} "
            f"-> {df.shape}"
        )

        if len(df) != len(sensor):

            raise ValueError(
                f"{key}: concat 후 행 수가 변경되었습니다."
            )


        # ----------------------------------------------------
        # 12. 바로 Parquet 저장
        # ----------------------------------------------------

        print("[SAVE] 저장 중...")

        df.to_parquet(
            output_path,
            index=False
        )

        print(
            f"[DONE] {output_path}"
        )


        # ----------------------------------------------------
        # 13. 메모리 정리
        # ----------------------------------------------------

        del sensor
        del ttf
        del sensor_time
        del ttf_time
        del df

        gc.collect()


print()
print("=" * 60)
print("모든 Sensor + TTF 데이터 생성 완료")
print("=" * 60)