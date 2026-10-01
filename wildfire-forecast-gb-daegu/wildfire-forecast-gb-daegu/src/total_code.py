#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### DWI(기상지수) 산출코드

# 날짜×좌표별로 ERA5 기반 기후변수(Tmean, RH, WSPD, TP_mm)와 DEM을 추출하는 코드
# 입력: CSV (필수 컬럼: date(YYYY-MM-DD), lat, lon  / 순서는 무관)
# 출력: fri_inputs_by_row.csv
#   - 컬럼: date, pid, lon, lat, Tmean(℃), RH(%), WSPD(m/s), TP_mm(mm/day), DEM(m)
# 특징:
#   - 날짜별 배치 처리로 getInfo 호출 수를 제어
#   - ERA5-Land에 값이 없을 때 ERA5(Global)로 자동 보완(unmask)
#   - ERA5 해상도(~9km)에 맞춘 기본 scale/버퍼 사용(필요시 조정 가능)
#   - 간단한 QC 로그로 NaN 비율, 좌표 범위 등을 확인

!pip -q install earthengine-api pandas tqdm

# 1) CSV 업로드: 좌표·날짜 파일 불러오기
from google.colab import files
uploaded = files.upload()   # 예: latlon_with_date.csv
CSV_FILENAME = next(iter(uploaded))
CSV_PATH = f"/content/{CSV_FILENAME}"

# 2) 기본 설정: 샘플 반경, 스케일, 날짜 보정 옵션
# ERA5 격자 크기가 약 9km이므로, 기본 반경과 스케일을 9000m로 설정
SAMPLE_RADIUS_M = 9000      # 포인트 주변 평균을 낼 원 반경(m)
SAMPLE_SCALE_M  = 9000      # reduceRegions에 사용될 스케일(m)
OUT_CSV = "/content/fri_inputs_by_row.csv"

# KST 기준 일평균을 맞추고 싶은 경우 날짜를 하루 이동시키는 등 보정 가능
# 기본은 0 (보정 없음)
DATE_SHIFT_DAYS = 0

# 3) GEE 초기화: 프로젝트 설정 및 인증
import ee, pandas as pd, datetime as dt
from tqdm import tqdm

PROJECT_ID = "YOUR_GEE_PROJECT_ID"  # 본인 GEE 프로젝트 ID
try:
    ee.Initialize(project=PROJECT_ID)
except Exception:
    ee.Authenticate(project=PROJECT_ID)
    ee.Initialize(project=PROJECT_ID)
print("[GEE] initialized")

# 4) 입력 로드 및 정규화: 헤더 정리, 형 변환, 필수 컬럼 체크
# - BOM 제거, 소문자 헤더 통일, date/lat/lon 형식 정리, 결측/중복 제거
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig", engine="python")
df.columns = [c.strip().lower().replace("\ufeff", "") for c in df.columns]

need = {"lat", "lon", "date"}
if not (need <= set(df.columns)):
    raise ValueError(f"필수 컬럼 누락. 현재 헤더: {list(df.columns)} (필요: {sorted(need)})")

df["lon"]  = pd.to_numeric(df["lon"], errors="coerce")
df["lat"]  = pd.to_numeric(df["lat"], errors="coerce")
df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

# 날짜 보정이 필요할 때만 적용 (UTC/KST 경계 이슈 완화용)
if DATE_SHIFT_DAYS != 0:
    df["date"] = df["date"].apply(
        lambda d: (dt.date.fromisoformat(str(d)) + dt.timedelta(days=DATE_SHIFT_DAYS))
        if pd.notnull(d) else d
    )

df = df.dropna(subset=["lon","lat","date"]).drop_duplicates().reset_index(drop=True)
df["pid"] = df.index.astype(int)

print(f"[INFO] rows after cleaning: {len(df)}")
print(df.head())

# 5) GEE 데이터셋 정의: ERA5(육지/전지구)와 DEM(SRTM)
ERA5_LAND   = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
ERA5_GLOBAL = ee.ImageCollection("ECMWF/ERA5/HOURLY")       # 바다 포함
SRTM        = ee.Image("USGS/SRTMGL1_003").select("elevation").rename("DEM")

# 6) 시간별 ERA5 이미지를 일 단위 요약으로 변환
def _per_hour_to_vars(im):
    # 기온(K)을 섭씨(℃)로 변환
    T  = im.select("temperature_2m").subtract(273.15).rename("Tmean")
    Td = im.select("dewpoint_temperature_2m").subtract(273.15)
    # 상대습도(%): Magnus 공식 기반 근사
    a, b = 17.625, 243.04
    RH = Td.expression(
        "100*exp(a*Td/(b+Td) - a*T/(b+T))",
        {"a": a, "b": b, "Td": Td, "T": T}
    ).rename("RH")
    # 풍속(m/s): u, v 성분으로부터 계산
    U = im.select("u_component_of_wind_10m")
    V = im.select("v_component_of_wind_10m")
    WSPD = U.pow(2).add(V.pow(2)).sqrt().rename("WSPD")
    # 강수량: 누적 강수(m)를 mm로 환산
    TP = im.select("total_precipitation").multiply(1000).rename("TP_mm")
    return T.addBands([RH, WSPD, TP])

def _daily_from(ic, date_str):
    # 주어진 날짜 [d0, d1) 구간의 시간 자료를 일 평균/일합계로 집계
    d0 = ee.Date(date_str); d1 = d0.advance(1, "day")
    hourly = ic.filterDate(d0, d1).map(_per_hour_to_vars)
    Tmean = hourly.select("Tmean").mean()
    RH    = hourly.select("RH").mean()
    WSPD  = hourly.select("WSPD").mean()
    TP    = hourly.select("TP_mm").sum()
    return Tmean.addBands([RH, WSPD, TP])

def daily_era5(date_str):
    # ERA5-Land 값을 우선 사용, 마스크된 픽셀은 ERA5(Global)로 보완
    land = _daily_from(ERA5_LAND, date_str)
    globe = _daily_from(ERA5_GLOBAL, date_str)
    fused = land.unmask(globe)
    # DEM 밴드와 날짜 속성 추가
    return fused.addBands(SRTM).set({"date": date_str})

# 7) reduceRegions 결과에서 원하는 키를 안전하게 추출하는 헬퍼
def pick(p, *names):
    for k in names:
        v = p.get(k)
        if v is not None:
            return v
    return None

# 8) 날짜별로 배치 샘플링: 같은 날짜의 포인트를 한 번에 reduceRegions
rows_out = []
unique_dates = sorted({d.isoformat() for d in df["date"].unique()})
print(f"[INFO] unique dates in CSV: {len(unique_dates)} → {unique_dates[:5]}{' ...' if len(unique_dates)>5 else ''}")

for d in tqdm(unique_dates, desc="[sampling by date]"):
    # 현재 날짜에 해당하는 행만 선택
    sub = df[df["date"] == dt.date.fromisoformat(d)].copy()

    # 각 포인트를 버퍼(원영역)로 확장한 FeatureCollection 생성
    fc = ee.FeatureCollection([
        ee.Feature(
            ee.Geometry.Point([float(r["lon"]), float(r["lat"])]).buffer(SAMPLE_RADIUS_M),
            {"pid": int(r["pid"]), "lon": float(r["lon"]), "lat": float(r["lat"])}
        )
        for _, r in sub.iterrows()
    ])

    # 해당 날짜의 일일 ERA5+DEM 이미지 생성
    img = daily_era5(d)

    # 버퍼 영역에 대해 픽셀 평균(reduceRegions) 수행
    reduced = img.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=SAMPLE_SCALE_M
    ).map(lambda f: f.set({"date": d}))

    # 결과를 클라이언트로 가져와 리스트에 누적
    feats = reduced.getInfo().get("features", [])
    for f in feats:
        p = f.get("properties", {})
        rows_out.append({
            "date": p.get("date"),
            "pid":  p.get("pid"),
            "lon":  p.get("lon"),
            "lat":  p.get("lat"),
            "Tmean": pick(p, "Tmean_mean", "Tmean"),
            "RH":    pick(p, "RH_mean", "RH"),
            "WSPD":  pick(p, "WSPD_mean", "WSPD"),
            "TP_mm": pick(p, "TP_mm_mean", "TP_mm"),
            "DEM":   pick(p, "DEM_mean", "DEM", "elevation_mean", "elevation"),
        })

# 9) 결과 저장, 간단 QC, 파일 다운로드
df_out = pd.DataFrame(rows_out).sort_values(["date","pid"]).reset_index(drop=True)
df_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

# 결측치 QC: 모든 기상변수가 NaN인 행과 일부만 NaN인 행 개수 확인
num_total   = len(df_out)
num_all_nan = df_out[["Tmean","RH","WSPD","TP_mm"]].isna().all(axis=1).sum()
num_any_nan = df_out[["Tmean","RH","WSPD","TP_mm"]].isna().any(axis=1).sum()
print(f"[SAVED] {OUT_CSV}  rows={num_total}")
print(f"[QC] all-NaN rows = {num_all_nan} / any-NaN rows = {num_any_nan}")

# 한국 대략 경계(위도 33~39, 경도 124~132) 벗어난 좌표가 있는지 빠르게 확인
bad = df[(df["lat"]<33)|(df["lat"]>39)|(df["lon"]<124)|(df["lon"]>132)]
if len(bad):
    print("[WARN] KR bounds outliers (first 5):")
    print(bad[["pid","date","lat","lon"]].head())

from google.colab import files
files.download(OUT_CSV)

### FMI(임상지수) 산출코드

# 임상도 코드(forest)를 산림 유형별 FMI로 변환하는 코드
# 입력: 엑셀/CSV 1개 (forest 코드 칼럼 포함)
# 출력: 원본 + FMI, FMI_missing 컬럼이 추가된 CSV

# 0) 준비: 라이브러리 임포트
import io, os, re
import numpy as np
import pandas as pd
from google.colab import files

# 1) 파일 업로드: 엑셀 또는 CSV 1개 선택
uploaded = files.upload()  # .xlsx, .xls, .csv 모두 허용
assert len(uploaded) == 1, "파일을 하나만 업로드하세요."
in_fname = list(uploaded.keys())[0]
raw = uploaded[in_fname]

# 2) 파일 로딩 헬퍼: 확장자와 인코딩을 자동 판별하여 DataFrame으로 로드
def load_table(name: str, buf: bytes) -> pd.DataFrame:
    lower = name.lower()
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(buf))  # 첫 시트 로드
    elif lower.endswith(".csv"):
        # CSV는 utf-8-sig 우선, 실패 시 cp949 시도
        try:
            return pd.read_csv(io.StringIO(buf.decode("utf-8-sig")))
        except Exception:
            return pd.read_csv(io.StringIO(buf.decode("cp949")))
    else:
        raise ValueError("지원하지 않는 확장자입니다. .xlsx/.xls/.csv 파일을 올려주세요.")

df = load_table(in_fname, raw)

# 3) forest 코드 컬럼명 설정 (파일에 맞게 필요 시 여기만 수정)
FOREST_COL = "forest"

if FOREST_COL not in df.columns:
    raise KeyError(f"업로드한 파일에 '{FOREST_COL}' 칼럼이 없습니다. 칼럼명을 확인하세요.")

# 4) FMI 매핑 함수: 산림청 코드 구간을 FMI 점수(숫자)로 변환
def map_code_to_fmi(raw_code):
    try:
        if pd.isna(raw_code):
            return np.nan
        code = int(raw_code)

        # 침엽수림(10~21) → FMI=10
        if 10 <= code <= 21:
            return 10
        # 활엽수림(30~49, 60~68) → FMI=2
        elif (30 <= code <= 49) or (60 <= code <= 68):
            return 2
        # 혼효림(77) → FMI=3
        elif code == 77:
            return 3
        # 죽림(78) → FMI=8
        elif code == 78:
            return 8
        # 무립목지(81,82) → FMI=5
        elif code in (81, 82):
            return 5
        # 비산림(91~99, -1) → FMI=0
        elif (91 <= code <= 99) or (code == -1):
            return 0
        # 정의되지 않은 코드 → NaN
        else:
            return np.nan
    except Exception:
        return np.nan

# 5) FMI 계산 및 보조 플래그 생성
df["FMI"] = df[FOREST_COL].apply(map_code_to_fmi)
# FMI_missing: 매핑 실패 또는 결측(1이면 FMI가 비어있음)
df["FMI_missing"] = df["FMI"].isna().astype(int)

# (선택) forest_type(범주형)까지 필요하면 아래 블록 주석 해제
# def map_forest_type(code):
#     try:
#         if pd.isna(code): return "unknown"
#         c = int(code)
#         if 10 <= c <= 21: return "conifer"
#         if (30 <= c <= 49) or (60 <= c <= 68): return "broadleaf"
#         if c == 77: return "mixed"
#         if c == 78: return "bamboo"
#         if c in (81, 82): return "grassland"
#         if (91 <= c <= 99) or (c == -1): return "nonforest"
#         return "unknown"
#     except Exception:
#         return "unknown"
# df["forest_type"] = df[FOREST_COL].apply(map_forest_type)

# 6) 결과 저장 및 다운로드: 원 파일명에 _with_fmi 붙여 CSV 저장
base = re.sub(r"\.(xlsx|xls|csv)$", "", in_fname, flags=re.IGNORECASE)
out_csv = f"{base}_with_fmi.csv"
df.to_csv(out_csv, index=False, encoding="utf-8-sig")
files.download(out_csv)

# (선택) 엑셀로도 내보내고 싶다면 아래 주석 해제
# out_xlsx = f"{base}_with_fmi.xlsx"
# df.to_excel(out_xlsx, index=False)
# files.download(out_xlsx)

### TMI(지형지수) 산출코드

# DEM을 이용해 각 좌표의 Slope(경사도)와 TMI(간이 지형지수)를 계산하는 코드
# 입력: CSV (필수 컬럼: lon, lat)
# 출력: 원본 + DEM, Slope_deg, TMI가 추가된 points_with_topography.csv

!pip -q install earthengine-api pandas chardet

import ee, pandas as pd, math, io, chardet
from google.colab import files

# 0) 설정: GEE 프로젝트, DEM 자산, TMI 커널 반경, 출력 파일명
PROJECT_ID = 'YOUR_GEE_PROJECT_ID'         # 본인 GEE 프로젝트 ID
DEM_ASSET  = 'USGS/SRTMGL1_003'            # 다른 DEM 자산이 있다면 교체 가능
TMI_KERNEL_M = 300                          # TMI 계산에 사용할 주변 반경(미터)
OUTPUT_CSV = 'points_with_topography.csv'

# 1) GEE 초기화: 프로젝트 기반 인증 및 연결
try:
    ee.Initialize(project=PROJECT_ID)
    print(f'[GEE] Initialized with project = {PROJECT_ID}')
except Exception:
    print('[GEE] OAuth required. Follow the popup…')
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)

# 2) CSV 업로드: 좌표 파일 선택 및 인코딩 자동 판별
print('[Upload] CSV 파일을 선택하세요 (lon, lat 필수).')
uploaded = files.upload()
assert len(uploaded) == 1, '한 개의 CSV만 업로드하세요.'
fname, raw = next(iter(uploaded.items()))

# 인코딩 자동 판별 후, 실패 시 흔한 인코딩으로 재시도
enc = chardet.detect(raw)['encoding'] or 'utf-8'
try:
    df = pd.read_csv(io.BytesIO(raw), encoding=enc)
except Exception:
    for alt in ['utf-8-sig','cp949','euc-kr']:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=alt)
            enc = alt
            break
        except Exception:
            pass
print(f'[Upload] Loaded "{fname}" with encoding={enc}')

# 필수 좌표 컬럼 확인
assert {'lon','lat'}.issubset(df.columns), "CSV에 lon, lat 컬럼이 필요합니다."

# 각 행에 고유 row_id를 부여하여 GEE 결과와 다시 매칭
df = df.reset_index().rename(columns={'index':'row_id'})

# 3) 포인트를 GEE FeatureCollection으로 변환
def row_to_feature(row):
    return ee.Feature(
        ee.Geometry.Point([float(row['lon']), float(row['lat'])]),
        {'row_id': int(row['row_id'])}
    )
fc = ee.FeatureCollection([row_to_feature(r) for r in df.to_dict('records')])

# 4) DEM, 경사도, TMI 계산
dem   = ee.Image(DEM_ASSET).select(['elevation']).rename('DEM')
slope = ee.Terrain.slope(dem).rename('Slope_deg')
slope_rad = slope.multiply(math.pi/180.0)

# 주변 집수면적 근사(A_local): 반경 TMI_KERNEL_M 내 픽셀 면적 합
pixel_area = ee.Image.pixelArea()
kernel     = ee.Kernel.circle(TMI_KERNEL_M, 'meters', normalize=False)
A_local    = pixel_area.reduceNeighborhood(ee.Reducer.sum(), kernel).rename('A_local_m2')

# TMI ≈ ln(A_local) - ln(tan(slope)), 급경사는 tan(slope) 값이 커짐
eps = ee.Number(1e-6)  # 0 회피용 작은 값
tmi = A_local.add(1).log().subtract(slope_rad.tan().add(eps).log()).rename('TMI')

# DEM, Slope_deg, TMI를 하나의 이미지로 결합
topo_img = ee.Image.cat([dem, slope, tmi])

# 5) sampleRegions로 각 포인트 위치에서 DEM, Slope_deg, TMI 추출
samples = topo_img.sampleRegions(
    collection=fc,
    properties=['row_id'],
    scale=30,           # DEM 해상도(약 30m)에 맞춰 샘플링
    geometries=False
)

# 결과를 클라이언트로 내려받아 DataFrame으로 변환
res = samples.getInfo()
rows = [f['properties'] for f in res['features']]
topo_df = pd.DataFrame(rows)  # row_id, DEM, Slope_deg, TMI

# 원본 df와 계산 결과를 row_id 기준으로 병합
out = df.merge(topo_df, on='row_id', how='left').drop(columns=['row_id'])

# 6) CSV 저장 및 다운로드
out.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
files.download(OUTPUT_CSV)
print(f'[DONE] Saved & Downloaded → {OUTPUT_CSV}')


### 좌표조정

# fmi == -1 인 좌표를 ERA5-Land 온도장을 이용해
# 같은 시간·날짜 주변의 "최고 기온 픽셀" 위치로 이동시키는 스크립트
# 입력:
#   - CSV (필수 컬럼: lat, lon, fmi, date 또는 datetime)
#   - date: YYYY-MM-DD, datetime: YYYY-MM-DD HH:MM:SS
# 사용 데이터:
#   - ECMWF/ERA5_LAND/HOURLY 의 temperature_2m
# 출력:
#   - /content/updated_coords.csv        : 좌표 보정 결과 (utf-8-sig, Excel 호환)
#   - /content/update_summary.txt        : 전체 갱신 통계 요약
#   - /content/update_failures.csv       : 대체 실패 행 (있을 때만 저장)

!pip -q install earthengine-api pandas tqdm

import ee, pandas as pd, numpy as np, datetime as dt
from tqdm import tqdm
from google.colab import files

# 0) 설정: GEE 프로젝트, 탐색 반경, 시간창, 출력 옵션
PROJECT_ID        = 'YOUR_GEE_PROJECT_ID'   # 본인 GEE 프로젝트 ID
BUFFER_KM_STEPS   = [5, 8, 12]              # 주변 탐색 반경(km), 단계적으로 확장
SCALE_M           = 2000                    # 샘플 해상도(m), 작을수록 정밀·느림
HOUR_MARGIN       = 1                       # datetime 기준 ±HOUR_MARGIN 시간창
VERBOSE           = True                    # 중간 로그 출력 여부

# 1) GEE 초기화: 프로젝트 기반 인증 및 연결
try:
    ee.Initialize(project=PROJECT_ID)
except Exception:
    ee.Authenticate(project=PROJECT_ID)
    ee.Initialize(project=PROJECT_ID)
print('[GEE] Initialized')

# 2) 파일 업로드: 좌표·날짜·fmi 정보가 담긴 CSV 불러오기
print("🔹 CSV 파일을 선택하세요 (예: input_points.csv)")
uploaded = files.upload()
CSV_PATH = list(uploaded.keys())[0]
print(f"업로드 완료: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# 필수 컬럼 체크: lat, lon, fmi 와 date 또는 datetime 필요
need = {'lat','lon','fmi'}
assert need.issubset(df.columns), f"CSV에 {need} 컬럼이 필요합니다."
has_datetime = 'datetime' in df.columns
has_date     = 'date' in df.columns
assert has_datetime or has_date, "CSV에 datetime 또는 date 컬럼 중 하나가 필요합니다."

# datetime/date 파싱: Excel 호환을 위해 datetime은 나중에 문자열로 다시 저장
if has_datetime:
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
if has_date:
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date

# 3) ERA5-Land 온도 컬렉션 정의
ERA = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").select('temperature_2m')

def parse_time_window(row):
    """행에서 시간창(ee.Date start, end) 계산: datetime 있으면 ±HOUR_MARGIN, 없으면 하루"""
    if has_datetime and pd.notnull(row.get('datetime')):
        t0 = pd.to_datetime(row['datetime'])
        start = ee.Date(t0.tz_localize('UTC').to_pydatetime()).advance(-HOUR_MARGIN, 'hour')
        end   = ee.Date(t0.tz_localize('UTC').to_pydatetime()).advance( HOUR_MARGIN+1, 'hour')
    else:
        d0 = pd.to_datetime(row['date']).date()
        start = ee.Date(dt.datetime(d0.year, d0.month, d0.day))
        end   = start.advance(1, 'day')
    return start, end

def find_hottest_point_latlon(lat, lon, start, end):
    """
    입력/외부 인터페이스는 (lat, lon).
    내부 GEE 포인트는 (lon, lat).
    반환: (new_lat, new_lon, tmax_K) 또는 None
    """
    p = ee.Geometry.Point([float(lon), float(lat)])
    img_time_max = ERA.filterDate(start, end).max()

    for buf_km in BUFFER_KM_STEPS:
        region = p.buffer(buf_km * 1000)

        # 1) 지정 반경 내 최대 온도 계산
        max_obj = img_time_max.reduceRegion(
            reducer=ee.Reducer.max(),
            geometry=region,
            scale=SCALE_M,
            bestEffort=True
        )
        max_val = ee.Number(max_obj.get('temperature_2m'))

        # 값이 없으면 더 큰 반경으로 재시도
        try:
            _ = max_val.getInfo()
        except Exception:
            if VERBOSE: print(f"  • no value @ {buf_km}km → retry")
            continue

        # 2) 최대 온도를 가지는 픽셀의 위경도 추출
        lonlat = ee.Image.pixelLonLat()
        max_mask = img_time_max.eq(max_val)
        lonlat_masked = lonlat.updateMask(max_mask)

        coord = lonlat_masked.reduceRegion(
            reducer=ee.Reducer.firstNonNull(),
            geometry=region,
            scale=SCALE_M,
            bestEffort=True
        )

        try:
            new_lon = ee.Number(coord.get('longitude')).getInfo()
            new_lat = ee.Number(coord.get('latitude')).getInfo()
            tmax_k  = max_val.getInfo()
            return new_lat, new_lon, tmax_k
        except Exception:
            if VERBOSE: print(f"  • no coord @ {buf_km}km → retry")
            continue

    return None

# 4) 메인 루프: 각 행에 대해 fmi==-1 이면 주변 최고 기온 픽셀로 좌표 이동
updated_lat, updated_lon, picked_tempK = [], [], []
updated_mask, fail_rows = [], []

records = df.to_dict('records')
for i, row in enumerate(tqdm(records, desc='Processing')):
    lat, lon = float(row['lat']), float(row['lon'])
    fmi = row['fmi']
    start, end = parse_time_window(row)

    if fmi == -1:
        try:
            res = find_hottest_point_latlon(lat, lon, start, end)
            if res is None:
                # 대체 실패: 원래 좌표를 유지
                updated_lat.append(lat)
                updated_lon.append(lon)
                picked_tempK.append(np.nan)
                updated_mask.append(False)
                fail_rows.append(i)
                if VERBOSE:
                    print(f"[{i}] fallback keep original (no hottest pixel found)")
            else:
                nl, nlo, tmaxk = res
                updated_lat.append(nl)
                updated_lon.append(nlo)
                picked_tempK.append(tmaxk)
                updated_mask.append(True)
                if VERBOSE:
                    print(f"[{i}] fmi=-1 → ({nl:.5f}, {nlo:.5f}) Tmax(K)={tmaxk:.2f}")
        except Exception as e:
            updated_lat.append(lat)
            updated_lon.append(lon)
            picked_tempK.append(np.nan)
            updated_mask.append(False)
            fail_rows.append(i)
            if VERBOSE:
                print(f"[{i}] error → keep original: {e}")
    else:
        # fmi != -1 인 좌표는 그대로 유지
        updated_lat.append(lat)
        updated_lon.append(lon)
        picked_tempK.append(np.nan)
        updated_mask.append(False)

df['new_lat'] = updated_lat
df['new_lon'] = updated_lon
df['hottest_temp_K'] = picked_tempK
df['was_updated'] = updated_mask

# 실제 좌표 교체: fmi==-1 이고 대체에 성공한 행만 lat, lon을 new_lat/new_lon으로 변경
mask_replace = (df['fmi'] == -1) & (df['was_updated'])
df.loc[mask_replace, 'lat'] = df.loc[mask_replace, 'new_lat']
df.loc[mask_replace, 'lon'] = df.loc[mask_replace, 'new_lon']

# 5) 결과 저장 및 요약 파일 생성
# datetime 컬럼은 Excel에서 보기 좋게 문자열로 변환
if has_datetime:
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

OUT_MAIN = '/content/updated_coords.csv'          # utf-8-sig (BOM 포함)
OUT_FAIL = '/content/update_failures.csv'
OUT_SUMM = '/content/update_summary.txt'

df.to_csv(OUT_MAIN, index=False, encoding='utf-8-sig', float_format='%.6f')

# 대체 실패 행만 따로 저장 (참고용)
if fail_rows:
    df.iloc[fail_rows].to_csv(OUT_FAIL, index=False, encoding='utf-8-sig', float_format='%.6f')

# 요약 리포트 작성: 전체 행 수, fmi==-1 개수, 실제로 갱신된 행 수, 실패 행 수
total = len(df)
n_minus1 = int((df['fmi'] == -1).sum())
n_updated = int(mask_replace.sum())
n_failed  = len(fail_rows)
with open(OUT_SUMM, 'w', encoding='utf-8') as f:
    f.write(f"Total rows           : {total}\n")
    f.write(f"fmi == -1 rows       : {n_minus1}\n")
    f.write(f"Updated (-1→hot)     : {n_updated}\n")
    f.write(f"Failed (kept original): {n_failed}\n")
print(open(OUT_SUMM, encoding='utf-8').read())

print(f"저장 완료: {OUT_MAIN}")
if fail_rows:
    print(f"실패 목록: {OUT_FAIL}")

# 최종 출력 파일 다운로드 (좌표 결과, 실패 목록, 요약 텍스트)
files.download(OUT_MAIN)
if fail_rows:
    files.download(OUT_FAIL)
files.download(OUT_SUMM)

### 산불 비발생 지역 랜덤 샘플링

# 양성 화점(pos.csv)을 기준으로, 동일 날짜(±K_DAYS)·동일 클러스터(지역+계절)에서
# 화점 주변 BUFFER_M 바깥에서 음성(비발생) 좌표를 무작위로 샘플링하는 스크립트
# 입력:
#   - pos.csv (date, lon, lat, region, season, label 포함, label=1)
# 출력:
#   - sampling_plan_for_extraction.csv (양성+음성 통합 샘플링 계획표)

import os, sys, math, io, datetime as dt
import numpy as np, pandas as pd
from shapely.geometry import Point
from shapely.strtree import STRtree
from pyproj import Transformer

# 0) 파일 업로드: Colab 또는 로컬 환경에서 pos.csv 선택
def pick_file_interactive():
    # Colab 환경: files.upload 사용
    try:
        from google.colab import files
        up = files.upload()
        fname = next(iter(up.keys()))
        with open(fname, 'wb') as f:
            f.write(up[fname])
        print("[Colab] Uploaded:", fname)
        return fname
    except Exception:
        pass
    # 일반 Jupyter/로컬 환경: Tkinter 파일 선택창 사용
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        fname = filedialog.askopenfilename(title="Select pos.csv", filetypes=[("CSV","*.csv"),("All","*.*")])
        if not fname: raise RuntimeError("No file selected")
        print("[Local] Selected:", fname)
        return fname
    except Exception as e:
        raise RuntimeError("파일 선택 실패. Colab이면 google.colab.files.upload를, 로컬이면 Tkinter 사용 가능 환경인지 확인하세요.") from e

POS_CSV = pick_file_interactive()

# 1) 샘플링 파라미터: 음성 비율, 화점 배제 버퍼, 날짜 범위, 난수 시드
RATIO_NEG_PER_POS = 1.5    # 음성/양성 목표 비율
BUFFER_M          = 1500   # 화점 주변 배제 거리(m)
K_DAYS            = 0       # 같은 날짜(±K_DAYS) 범위
SEED              = 42

# 좌표계 변환기: 경위도(WGS84) ↔ 투영좌표(EPSG:5179, 한국 중부권 예시)
to_m  = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True).transform
to_deg= Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True).transform

rng = np.random.default_rng(SEED)

# 2) 양성 데이터 로드 및 컬럼 이름 정리
pos = pd.read_csv(POS_CSV)
need_cols = {'date','lon','lat','region','season','label'}
missing = need_cols - set(map(str.lower, pos.columns))

# 원본 컬럼명 → 소문자 키로 매핑
cols_map = {c.lower(): c for c in pos.columns}
def col(name): return cols_map[name]

if missing:
    raise ValueError(f"필수 컬럼 누락: {missing}. 필요한 컬럼: {sorted(list(need_cols))}")

pos = pos.rename(columns={col('date'):'date', col('lon'):'lon', col('lat'):'lat',
                          col('region'):'region', col('season'):'season', col('label'):'label'})
pos['label'] = 1
pos['date']  = pd.to_datetime(pos['date']).dt.date

# region+season 조합을 하나의 클러스터 키로 사용
pos['cluster'] = pos['region'].astype(str) + '|' + pos['season'].astype(str)
print(f"[INFO] Loaded positives: {len(pos)} rows, clusters={pos['cluster'].nunique()}")

# 3) 날짜별 화점 버퍼(투영좌표계에서 BUFFER_M 만큼) 생성
def to_point_m(lon, lat):
    x, y = to_m(lon, lat)
    return Point(x, y)

pos['date_dt'] = pd.to_datetime(pos['date'])
buf_rows = []
for d, df_d in pos.groupby('date_dt'):
    # 해당 날짜의 모든 화점을 BUFFER_M 만큼 버퍼링
    geoms_m = [to_point_m(lon, lat).buffer(BUFFER_M) for lon,lat in zip(df_d['lon'], df_d['lat'])]
    buf_rows.append((d, geoms_m, STRtree(geoms_m)))

def get_buffer_union_for_date(d):
    # d 기준 ±K_DAYS 날짜의 모든 화점 버퍼를 합집합으로 계산
    dates = [d + dt.timedelta(days=k) for k in range(-K_DAYS, K_DAYS+1)]
    geoms = []
    for dd, geoms_m, _ in buf_rows:
        if dd.date() in [x.date() for x in dates]:
            geoms.extend(geoms_m)
    if not geoms:
        return None
    u = geoms[0]
    for g in geoms[1:]:
        u = u.union(g)
    return u

# 4) 클러스터×날짜별 목표 음성 개수 계산
targets = (pos.groupby(['cluster','date'])
             .size().rename('n_pos').reset_index())
targets['n_neg_target'] = np.ceil(RATIO_NEG_PER_POS * targets['n_pos']).astype(int)

# 5) 랜덤 샘플링 박스 설정: 최소/최대 경위도 박스 안에서 음성 위치 생성
#   (필요하면 이 박스를 행정경계(GeoJSON 등)로 교체 가능)
min_lon, max_lon = pos['lon'].min()-0.5, pos['lon'].max()+0.5
min_lat, max_lat = pos['lat'].min()-0.5, pos['lat'].max()+0.5

def random_points_in_box(n):
    lons = rng.uniform(min_lon, max_lon, n)
    lats = rng.uniform(min_lat, max_lat, n)
    return list(zip(lons, lats))

# 6) 음성 샘플링: 화점 버퍼 밖에서 목표 개수만큼 음성 좌표 생성
neg_rows = []
for (cluster, date), row in targets.set_index(['cluster','date']).iterrows():
    need = int(row['n_neg_target'])
    bu = get_buffer_union_for_date(pd.to_datetime(date))
    got = 0
    trials = 0
    cap = max(need * 300, 5000)  # 최대 시도 횟수 상한

    while got < need and trials < cap:
        batch_n = min(need - got, 500)
        for lon, lat in random_points_in_box(batch_n):
            trials += 1
            pt_m = to_point_m(lon, lat)
            # 화점 버퍼 안이면 음성 후보에서 제외
            if bu is not None and bu.contains(pt_m):
                continue
            neg_rows.append(dict(
                date=date, lon=lon, lat=lat,
                region=cluster.split('|')[0],
                season=cluster.split('|')[1],
                label=0
            ))
            got += 1
            if got >= need: break

    if got < need:
        # 지정한 버퍼/날짜 범위에서 음성이 부족할 때 경고 메시지 출력
        print(f"[WARN] 부족: cluster={cluster}, date={date}, target_neg={need}, got={got}. "
              f"K_DAYS 확대 또는 BUFFER_M 완화 고려.")

neg = pd.DataFrame(neg_rows)

# 7) 최종 샘플링 계획 병합 및 저장
pos_out = pos[['date','lon','lat','region','season','label']].copy()
plan = pd.concat([pos_out, neg], ignore_index=True)

out_csv = "sampling_plan_for_extraction.csv"
plan.to_csv(out_csv, index=False)
print(f"[DONE] saved → {out_csv} (rows={len(plan)})")

# 요약 리포트: 전체 label 비율 및 region·season·label별 개수 테이블
print("\n[Summary]")
print(plan['label'].value_counts())
print(plan.groupby(['region','season','label']).size().unstack(fill_value=0))

from google.colab import files
files.download("sampling_plan_for_extraction.csv")


### 분류

# 산불 발생 여부(label)를 예측하는 이진 분류 스크립트
# 입력:
#   - CSV 1개 (필수 컬럼: label, date, lon, lat, TP_mm, Tmax, Tmin, RH, WSPD, FMI, DEM, Slope)
#   - 선택 컬럼: Tmean, TMI, FFDRI, FRI_norm 등은 있으면 사용
# 처리:
#   - 좌표별로 시계열 정렬 후 이동평균·누적강수·건조일수(DrySpell) 파생
#   - 2015–2022년을 학습, 2023/2024년을 홀드아웃 평가
#   - RandomForest/ExtraTrees 학습, RF 기준 OOF로 임계값 최적화(F2 최대, P 하한 0.35)
#   - season_cluster별로 다른 임계값 적용
# 출력:
#   - /content/artifacts/thresholds.json      : 계절별 임계값
#   - /content/artifacts/feature_list.json    : 사용 피처 리스트
#   - /content/artifacts/clf_rf.joblib        : RF 모델 리스트
#   - /content/pred_cls_2023.csv, pred_cls_2024.csv : 연도별 예측/경보 결과

# 패키지 설치 (Colab 기준)
!pip -q install scikit-learn imbalanced-learn lightgbm joblib pandas

# CSV 업로드 (Colab 위젯으로 직접 선택)
from google.colab import files
up = files.upload()  # CSV 직접 선택

import io, pandas as pd, numpy as np, json, joblib, os, warnings
warnings.filterwarnings("ignore")

CSV_PATH = next(iter(up))  # 업로드한 첫 번째 파일명
print("Loaded:", CSV_PATH)

# CSV 로드 (여러 인코딩을 순차적으로 시도)
enc_trials = ["utf-8", "cp949", "euc-kr"]
for enc in enc_trials:
    try:
        df = pd.read_csv(io.BytesIO(up[CSV_PATH]), encoding=enc, parse_dates=["date"])
        print(f"[INFO] Read CSV with encoding={enc}, rows={len(df)}")
        break
    except Exception as e:
        last_err = e
else:
    raise last_err

# 필수/선택 컬럼 점검 및 Tmin 보정
must_cols = ["label","date","lon","lat","TP_mm","Tmax","Tmin","RH","WSPD","FMI","DEM","Slope"]
miss = [c for c in must_cols if c not in df.columns]
if miss:
    # Tmin만 없는 경우: Tmean을 이용해 근사 생성
    if miss == ["Tmin"]:
        if "Tmean" in df.columns:
            df["Tmin"] = 2*df["Tmean"] - df["Tmax"]
            print("[INFO] 'Tmin' 미존재 → Tmean,Tmax로 근사 생성.")
        else:
            raise ValueError(f"필수 칼럼 누락: {miss} (또는 Tmean 제공)")
    else:
        raise ValueError(f"필수 칼럼 누락: {miss}")

opt_cols = ["Tmean","TMI","FFDRI","FRI_norm","FRI_grade","range","RNE","EH","pDWI","DWI_by_month","DWI","season_cluster"]
print("[INFO] Optional present:", [c for c in opt_cols if c in df.columns])

# 연도 파생 변수 추가
df["year"] = df["date"].dt.year

# 일교차(DTR) 파생
df["DTR"] = df["Tmax"] - df["Tmin"]

# season_cluster가 없으면 단순 규칙으로 봄/여름/가을겨울 분리
if "season_cluster" not in df.columns:
    m = df["date"].dt.month
    season = np.where(m.isin([3,4,5]), "spring",
              np.where(m.isin([6,7,8]), "summer", "fall_winter"))
    df["season_cluster"] = season
    print("[INFO] season_cluster 생성 완료 (rule-based: spring/summer/fall_winter)")

# 좌표 단위 시계열 그룹 키 (필요하면 season_cluster를 포함하는 것도 가능)
group_keys = ["lat","lon"]

def add_memory_feats(g):
    # 각 좌표별로 시간순 정렬 이후 이동특징을 계산한다고 가정
    # 3일, 7일 이동평균 (기온, 습도, 풍속, 강수, 일교차)
    for c in ["Tmax","RH","WSPD","TP_mm","DTR"]:
        g[f"{c}_ma3"] = g[c].rolling(3, min_periods=1).mean()
        g[f"{c}_ma7"] = g[c].rolling(7, min_periods=1).mean()
    # 최근 3/7일 누적강수 (강수량 누적)
    g["TP_3obs_sum"] = g["TP_mm"].rolling(3, min_periods=1).sum()
    g["TP_7obs_sum"] = g["TP_mm"].rolling(7, min_periods=1).sum()
    # DrySpell: 0.1mm 이하 무강수일 연속 길이
    dry = (g["TP_mm"].fillna(0) <= 0.1).astype(int)
    g["DrySpell"] = dry.groupby((dry==0).cumsum()).cumcount() * dry
    return g

# 좌표별·날짜순으로 정렬 후 메모리 피처 생성
df = df.sort_values(["lat","lon","date"]).groupby(group_keys, group_keys=False).apply(add_memory_feats).reset_index(drop=True)

# 분류 모델 입력 피처 구성
FEATURES = [
    "Tmax","RH","WSPD","TP_mm","DTR",
    "Tmax_ma3","Tmax_ma7","RH_ma3","RH_ma7","WSPD_ma3","WSPD_ma7","DTR_ma3","DTR_ma7",
    "TP_3obs_sum","TP_7obs_sum","DrySpell",
    "Slope","FMI","DEM"
]
TARGET = "label"
CLUSTER = "season_cluster"

# 무한대 값 제거 및 수치 결측을 중앙값으로 대체
df[FEATURES] = df[FEATURES].replace([np.inf,-np.inf], np.nan)
df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())

# 학습/평가 기간 분할
train = df[(df["year"]>=2015) & (df["year"]<=2022)].copy()
test23 = df[df["year"]==2023].copy()
ext24  = df[df["year"]==2024].copy()

print("Shapes →", "train:", train.shape, "test23:", test23.shape, "ext24:", ext24.shape)

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from scipy.optimize import minimize_scalar
import numpy as np, os, json, joblib

SEED = 123
N_SPLITS = 5

def get_model(name="RF"):
    # RandomForest와 ExtraTrees 두 가지 트리 기반 분류기 정의
    if name=="RF":
        return RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, random_state=SEED, class_weight="balanced"
        )
    if name=="ET":
        return ExtraTreesClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, random_state=SEED, class_weight="balanced"
        )
    raise ValueError

def cv_fit_predict(train_df, feats, target, groups, model_name="RF"):
    # 연도(year)를 그룹으로 하는 StratifiedGroupKFold 교차검증 수행
    # 반환:
    #   - fold별 학습된 모델 리스트
    #   - 전체 학습 데이터에 대한 OOF 예측 확률
    #   - PR-AUC, ROC-AUC 성능 지표
    X = train_df[feats].values
    y = train_df[target].values
    g = train_df[groups].values
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train_df))
    models = []
    for tr, va in cv.split(X, y, groups=g):
        clf = get_model(model_name)
        clf.fit(X[tr], y[tr])
        p  = clf.predict_proba(X[va])[:,1]
        oof[va] = p
        models.append(clf)
    pr, roc = average_precision_score(y, oof), roc_auc_score(y, oof)
    return models, oof, {"PR-AUC":pr, "ROC-AUC":roc}

# RF, ET 두 모델에 대해 교차검증 기반 성능 비교
models_rf, oof_rf, metr_rf = cv_fit_predict(train, FEATURES, TARGET, "year", "RF")
models_et, oof_et, metr_et = cv_fit_predict(train, FEATURES, TARGET, "year", "ET")
print("[RF] ", metr_rf)
print("[ET] ", metr_et)

# 임계값 탐색: F2-score 최대 + Precision 하한(≥0.35) 조건을 만족하는 threshold 선택
def find_threshold_f2_with_precision(y_true, y_prob, min_prec=0.35):
    # 0~1 구간에서 촘촘한 격자(threshold)를 스캔하여 안정적으로 탐색
    best_t, best = 0.0, (-1,-1,-1)  # (P,R,F2)
    for thr in np.linspace(0,1,2001):
        yp = (y_prob>=thr).astype(int)
        p,r,f2,_ = precision_recall_fscore_support(y_true, yp, beta=2.0,
                                                   average="binary", zero_division=0)
        if p>=min_prec and f2>best[2]:
            best_t, best = float(thr), (p,r,f2)
    if best[2] >= 0:
        return best_t, {"precision":best[0], "recall":best[1], "f2":best[2]}
    # Precision 하한을 만족하는 값이 없으면 F2만 최대가 되는 임계값으로 대체
    best_t2, best2 = 0.0, (-1,-1,-1)
    for thr in np.linspace(0,1,2001):
        yp = (y_prob>=thr).astype(int)
        p,r,f2,_ = precision_recall_fscore_support(y_true, yp, beta=2.0,
                                                   average="binary", zero_division=0)
        if f2>best2[2]:
            best_t2, best2 = float(thr), (p,r,f2)
    return best_t2, {"precision":best2[0], "recall":best2[1], "f2":best2[2]}

# 전체 학습기간 OOF 확률을 이용해 전역 임계값 계산
y_true_tr = train[TARGET].values
t_star, stats = find_threshold_f2_with_precision(y_true_tr, oof_rf, min_prec=0.35)
print(f"[Global t*] {t_star:.4f}  stats={stats}")

# OOF 결과를 Series로 만들어 index 기반으로 안전하게 서브셋 선택
oof_series = pd.Series(oof_rf, index=train.index)

# season_cluster별로 개별 임계값 계산 (데이터가 전부 0 또는 1이면 전역 임계값 사용)
thresholds_by_cluster = {}
for cid, gdf in train.groupby("season_cluster"):
    idx = gdf.index
    y_c = gdf[TARGET].values
    p_c = oof_series.loc[idx].values
    if (y_c.sum()==0) or (y_c.sum()==len(y_c)):
        t, s = t_star, {"precision":np.nan,"recall":np.nan,"f2":np.nan}
    else:
        t, s = find_threshold_f2_with_precision(y_c, p_c, 0.35)
    thresholds_by_cluster[str(cid)] = float(t)

thresholds_by_cluster["__global__"] = float(t_star)
print("thresholds_by_cluster =", thresholds_by_cluster)

# RF 모델 및 임계값, 피처 리스트를 아티팩트로 저장
os.makedirs("/content/artifacts", exist_ok=True)
with open("/content/artifacts/thresholds.json","w") as f: json.dump(thresholds_by_cluster, f, indent=2)
with open("/content/artifacts/feature_list.json","w") as f: json.dump(FEATURES, f, indent=2)
joblib.dump(models_rf, "/content/artifacts/clf_rf.joblib")
print("Saved artifacts → /content/artifacts")

def predict_mean_proba(models, X):
    # fold별 RF 확률 예측을 평균내어 최종 확률로 사용
    return np.mean([m.predict_proba(X)[:,1] for m in models], axis=0)

def evaluate_binary(y, p, thr):
    # 주어진 임계값(thr)에 대해 PR-AUC, ROC-AUC, Precision/Recall/F2 등 산출
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support
    yhat = (p>=thr).astype(int)
    pr = average_precision_score(y, p)
    roc = roc_auc_score(y, p)
    P,R,F2,_ = precision_recall_fscore_support(y, yhat, beta=2.0, average="binary", zero_division=0)
    return {"PR-AUC":pr, "ROC-AUC":roc, "P":P, "R":R, "F2":F2, "N_pred_pos":int(yhat.sum()), "N":int(len(yhat))}

# 저장된 임계값 로드
with open("/content/artifacts/thresholds.json") as f:
    THR = json.load(f)

# 2023, 2024 데이터에 대해 예측 및 경보 CSV 생성
for name, hold in {"2023":test23, "2024":ext24}.items():
    if len(hold)==0:
        print(f"[{name}] no rows."); continue
    Xh = hold[FEATURES].values
    ph = predict_mean_proba(models_rf, Xh)

    # 참고용: 고정 임계값 0.5 기준 성능 출력
    rep05 = evaluate_binary(hold[TARGET].values, ph, 0.5)
    print(f"[{name}] metrics@0.5 →", rep05)

    # 실제 경보는 season_cluster별 임계값 적용
    thr_vec = np.array([ THR.get(str(c), THR["__global__"]) for c in hold["season_cluster"].astype(str) ])
    yhat = (ph >= thr_vec).astype(int)

    out = hold[["date","lon","lat","season_cluster",TARGET]].copy()
    out["y_prob"] = ph
    out["y_pred"] = yhat
    out_path = f"/content/pred_cls_{name}.csv"
    out.to_csv(out_path, index=False)
    print("Saved:", out_path, "| 경보량:", yhat.sum(), "/", len(yhat))

    from google.colab import files

# 생성된 예측 결과 CSV 파일 다운로드 (2023, 2024)
files.download("/content/pred_cls_2023.csv")
files.download("/content/pred_cls_2024.csv")



### 회귀

# FFDRI 또는 FRI_norm과 같은 연속형 위험지수를 회귀로 예측하는 스크립트
# 특징:
#   - 업로드된 원본 CSV에서 타깃(FFDRI/FRI_norm)을 자동 탐지
#   - 좌표·계절별 일 단위로 조밀화(asfreq('D')) 후 짧은 결측만 보간
#   - 각 시점에서 향후 L일(0~7일) 리드 타깃을 생성(row-lead 구조)
#   - LightGBM 우선, 데이터가 작거나 깊은 리드에서는 ElasticNet으로 폴백
#   - 2015–2022 학습, 2023/2024에 대한 리드별 성능표 및 예측 CSV, 등급컷 저장
# 출력:
#   - /content/artifacts_reg/reg_L0_7_rowlead.joblib     : 리드별 회귀 모델 dict
#   - /content/artifacts_reg/reg_features_used.json      : 리드별 실제 사용 피처 목록
#   - /content/artifacts_reg/grades_reg.json             : 위험도 등급 구간(Low/Moderate/High/VeryHigh)
#   - /content/pred_reg_2023_rowlead.csv, _2024_rowlead.csv : 리드별 예측값/등급
#   - /content/artifacts_reg_bundle_rowlead.zip          : 회귀 관련 아티팩트 압축본

# 회귀에 필요한 모듈 임포트
import os, json, shutil, joblib, numpy as np, pandas as pd, re
from scipy.stats import pearsonr

# Colab/로컬 공용 다운로드 헬퍼
try:
    from google.colab import files
except Exception:
    class _Files:
        def upload(self): raise RuntimeError("로컬 환경: files.upload() 미지원. 대신 경로 지정 로드 필요.")
        def download(self, path): print(f"[local] 파일 저장됨: {path}")
    files = _Files()

# 원본 데이터 CSV 업로드 (예: wildfire_dataset.csv)
print("원본 데이터 CSV 업로드 (예: wildfire_dataset.csv)")
up = files.upload()
PATH = list(up.keys())[0]

# 인코딩 자동 시도 (utf-8-sig → utf-8 → cp949 → euc-kr 순서)
_encs = ["utf-8-sig","utf-8","cp949","euc-kr"]
for enc in _encs:
    try:
        df = pd.read_csv(PATH, encoding=enc)
        print(f"로드 성공: {PATH} (encoding={enc}) shape={df.shape}")
        break
    except Exception as e:
        last_err = e
else:
    raise RuntimeError(f"CSV 로드 실패: {last_err}")

# 날짜 컬럼 표준화 (date가 없으면 dt/날짜/ymd 후보를 date로 리네임)
if "date" not in df.columns:
    cand = [c for c in df.columns if c.lower() in ("dt","날짜","ymd")]
    if cand:
        df = df.rename(columns={cand[0]: "date"})
    else:
        raise ValueError("date 컬럼이 필요합니다 (YYYY-MM-DD).")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# 연도 파생 변수 (없으면 생성)
if "year" not in df.columns:
    df["year"] = df["date"].dt.year

# season_cluster가 없으면 간단 규칙으로 생성 (봄: 2~4, 가을·겨울: 10~12, 나머지는 summer)
if "season_cluster" not in df.columns:
    m = df["date"].dt.month
    season = np.select([m.isin([2,3,4]), m.isin([10,11,12])],
                       ["spring","fall_winter"], default="summer")
    df["season_cluster"] = season

# 위경도 이름 정리 (소문자화 + longitude/latitude → lon/lat 치환)
def _scrub_cols(cols):
    fixed = {}
    for c in cols:
        cl = c.strip().lower()
        cl = cl.replace("longitude","lon").replace("latitude","lat")
        fixed[c] = cl
    return fixed
df = df.rename(columns=_scrub_cols(df.columns))

# lon/lat 필수: 없으면 유사 이름을 찾아서 리네임
for need in ["lon","lat"]:
    if need not in df.columns:
        cand = [c for c in df.columns if re.fullmatch(rf".*\b{need}\b.*", c.lower())]
        if cand:
            df = df.rename(columns={cand[0]: need})
        else:
            raise ValueError(f"{need} 컬럼이 필요합니다.")

# 타깃 컬럼 자동 탐지 (FFDRI → FRI_norm 순으로 우선, 없으면 패턴/숫자형 후보 안내)
def pick_target_column(_df):
    cols = list(_df.columns)
    lowmap = {c.lower(): c for c in cols}
    for key in ["ffdri","fri_norm"]:
        if key in lowmap: return lowmap[key]
    for pat in [r"ffdr[iy]?", r"fri[_\s-]*norm", r"^fri$"]:
        for c in cols:
            if re.search(pat, c.lower()):
                return c
    num_candidates = [c for c in cols if np.issubdtype(_df[c].dropna().dtype, np.number)]
    print("타깃(FFDRI/FRI_norm) 자동탐지 실패. 숫자형 후보:", num_candidates)
    user = input("타깃 칼럼명 입력: ").strip()
    if user in cols: return user
    raise ValueError("회귀 타깃 칼럼을 찾지 못했습니다.")

TARGET_CONT = pick_target_column(df)
TARGET_CONT = TARGET_CONT.strip()

# 기본 기상·지형 피처 스키마 (표준 이름)
base_feats_std = [
    "tmax","rh","wspd","tp_mm","dtr",
    "tmax_ma3","tmax_ma7","rh_ma3","rh_ma7",
    "wspd_ma3","wspd_ma7","dtr_ma3","dtr_ma7",
    "tp_3obs_sum","tp_7obs_sum","dryspell","slope","fmi","dem"
]

# 데이터셋 내 다양한 이름을 표준 이름으로 매핑하기 위한 별칭
ALIASES = {
    "tmax":       ["tmax","t_max","tasmax","tmax_c","tmax_deg","tmax_℃","tmax_celsius"],
    "rh":         ["rh","relhum","humidity","relative_humidity","rh_%"],
    "wspd":       ["wspd","ws","wind","wind_speed","wspd_mps"],
    "tp_mm":      ["tp_mm","tp","precip","precip_mm","prcp_mm","rain_mm","ppt_mm"],
    "dtr":        ["dtr","diurnal_range","tmax_tmin_diff","tdiff"],
    "tmax_ma3":   ["tmax_ma3","tmax_3ma","tmax_ma_3"],
    "tmax_ma7":   ["tmax_ma7","tmax_7ma","tmax_ma_7"],
    "rh_ma3":     ["rh_ma3","rh_3ma","rh_ma_3"],
    "rh_ma7":     ["rh_ma7","rh_7ma","rh_ma_7"],
    "wspd_ma3":   ["wspd_ma3","wspd_3ma","ws_3ma"],
    "wspd_ma7":   ["wspd_ma7","wspd_7ma","ws_7ma"],
    "dtr_ma3":    ["dtr_ma3","dtr_3ma"],
    "dtr_ma7":    ["dtr_ma7","dtr_7ma"],
    "tp_3obs_sum":["tp_3obs_sum","tp_3d_sum","precip_3d_sum","rain_3d_sum"],
    "tp_7obs_sum":["tp_7obs_sum","tp_7d_sum","precip_7d_sum","rain_7d_sum"],
    "dryspell":   ["dryspell","dry_spell","drydays","dry_days"],
    "slope":      ["slope","dem_slope","srtm_slope"],
    "fmi":        ["fmi","forest_moisture_index","fuel_moisture_index"],
    "dem":        ["dem","elev","elevation","altitude","srtm"]
}

present = set(df.columns)
for std, aliases in ALIASES.items():
    found = None
    for a in aliases:
        if a in present:
            found = a; break
    if found and (found != std) and (std not in df.columns):
        df = df.rename(columns={found: std})

# 실제 사용 가능한 회귀 피처 리스트 (데이터에 존재하는 것만 사용)
FEATURES = [c for c in base_feats_std if c in df.columns]
REG_FEATURES = FEATURES[:]
print(f"기본 피처 수: {len(REG_FEATURES)} / 후보 {len(base_feats_std)}")

# 회귀용 파라미터 정의
DATE = "date"
ID_COLS = ["season_cluster","lat","lon"]
DAILY_FFILL_LIMIT = 3
INTERP_LIMIT      = 3
LAG_DAYS = 7
NA_RATIO_CUT = 0.7
MIN_VAR = 1e-10
MIN_FEATS = 5
MIN_Y = 8
MAX_L = 7

# 데일리화 및 짧은 결측 보간에 사용할 컬럼
interp_feats = [c for c in (REG_FEATURES + [TARGET_CONT]) if c in df.columns]

# 좌표·계절별로 날짜 인덱스를 "매일"로 확장하고
# 짧은 간격의 결측만 시간 보간/앞뒤 채우기로 메우는 함수
def densify_daily(df_in):
    out_list = []
    for keys, g in df_in.sort_values(DATE).groupby(ID_COLS, dropna=False):
        g = g.set_index(DATE).sort_index()

        # 최대 예측 리드(MAX_L)일 이후까지 인덱스를 확장
        end_ext = g.index.max() + pd.Timedelta(days=MAX_L)
        full_idx = pd.date_range(g.index.min(), end_ext, freq="D")
        g = g.reindex(full_idx)

        # ID 컬럼 값 유지
        for i, col in enumerate(ID_COLS):
            g[col] = keys[i]

        # 수치형 컬럼에 대해 시간 보간/앞뒤 채우기를 적용
        num_cols = [c for c in interp_feats if c in g.columns]

        # 기상/지형 피처: 앞뒤 짧은 결측에만 보간/채우기 허용
        g[num_cols] = g[num_cols].interpolate(method="time", limit=INTERP_LIMIT, limit_direction="both")
        g[num_cols] = g[num_cols].ffill(limit=DAILY_FFILL_LIMIT).bfill(limit=DAILY_FFILL_LIMIT)

        # 타깃은 미래 리드 라벨 생성을 위해 오른쪽(미래 방향)으로 조금 더 허용
        if TARGET_CONT in g.columns:
            g[TARGET_CONT] = (
                g[TARGET_CONT]
                  .interpolate(method="time", limit=INTERP_LIMIT, limit_direction="forward")
                  .ffill(limit=MAX_L)
            )

        g = g.reset_index().rename(columns={"index": DATE})
        out_list.append(g)
    return pd.concat(out_list, ignore_index=True)

# 좌표·계절별 데일리 시계열 생성
df_dense = densify_daily(df)
print("Dense shape:", df_dense.shape)

# 타깃의 과거값(래그) 피처 추가 (0일전, 1일전, ..., LAG_DAYS-1일전)
grp = df_dense.sort_values(ID_COLS + [DATE]).groupby(ID_COLS, group_keys=False)
for k in range(0, LAG_DAYS):
    lag_col = f"{TARGET_CONT}_lag{k}"
    df_dense[lag_col] = grp[TARGET_CONT].shift(k)

lag_feats = [f"{TARGET_CONT}_lag{k}" for k in range(0, LAG_DAYS)]
REG_FEATURES = REG_FEATURES + [c for c in lag_feats if c in df_dense.columns]
print(f"래그 포함 피처 수: {len(REG_FEATURES)}")

# date+L 매칭 대신, 그룹 내에서 단순 시프트로 리드 타깃 생성
# 각 L에 대해 현재 시점에서 L일 뒤의 값을 타깃으로 사용
def make_leads_by_shift(df_in, id_cols, date_col, target_col, max_l=7):
    out = df_in.sort_values(id_cols + [date_col]).copy()
    g = out.groupby(id_cols, group_keys=False)
    for L in range(0, max_l+1):
        out[f"{target_col}_L{L}"] = g[target_col].shift(-L)
    return out

df_reg = make_leads_by_shift(df_dense, ID_COLS, DATE, TARGET_CONT, max_l=MAX_L)

# 회귀용 학습/평가 기간 분할
train_r = df_reg[(df_reg["year"]>=2015) & (df_reg["year"]<=2022)].copy()
test23_r = df_reg[df_reg["year"]==2023].copy()
ext24_r  = df_reg[df_reg["year"]==2024].copy()
print("Shapes (reg):", "train:", train_r.shape, "2023:", test23_r.shape, "2024:", ext24_r.shape)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor

# 소규모 데이터에 적합한 LightGBM 회귀기 생성 함수
def make_lgbm_smalldata():
    params = dict(
        n_estimators=1200, learning_rate=0.02,
        num_leaves=31, max_depth=6,
        subsample=0.9, colsample_bytree=0.9,
        min_child_samples=5, random_state=123, n_jobs=-1
    )
    try:
        params.update(dict(min_data_in_bin=1, feature_pre_filter=False, force_col_wise=True))
        _ = LGBMRegressor(**params)
    except TypeError:
        for k in ["min_data_in_bin","feature_pre_filter","force_col_wise"]:
            params.pop(k, None)
    return LGBMRegressor(**params)

# 리드 L에 대해 학습/평가용 X, y를 만드는 함수
def prepare_xy(sub, L, feats, target_col, na_ratio_cut=NA_RATIO_CUT):
    y = sub[f"{target_col}_L{L}"].values
    X = sub[feats].replace([np.inf,-np.inf], np.nan)
    ok = ~np.isnan(y)
    X, y = X.loc[ok], y[ok]
    # 컬럼별 결측 비율이 너무 높으면 제거
    keep = X.isna().mean() <= na_ratio_cut
    X = X.loc[:, keep.values]
    # 중앙값 대체 후 분산이 거의 없는 컬럼 제거
    X = X.fillna(X.median())
    var = X.var(ddof=0)
    X = X.loc[:, var > MIN_VAR]
    # 피처 수가 너무 적으면 기본 기상 코어 피처를 추가로 보장
    if X.shape[1] < MIN_FEATS:
        core = [c for c in ["tmax","rh","wspd","tp_mm","dtr"] if c in sub.columns]
        add = sub.loc[ok, core].replace([np.inf,-np.inf], np.nan).fillna(sub[core].median())
        X = pd.concat([X, add], axis=1)
        var = X.var(ddof=0)
        X = X.loc[:, var > MIN_VAR]
    return X, y, list(X.columns)

# 소규모 데이터용 회귀 모델 학습 (LGBM 우선, 필요 시 ElasticNet으로 폴백)
def train_regressor_small_data(X, y, L=None):
    # 리드가 크거나 표본이 적으면 ENet을 우선 사용
    if L is None or len(y) < 80 or (isinstance(L, int) and L >= 5):
        model = Pipeline([("scaler", StandardScaler()),
                          ("enet", ElasticNet(alpha=0.05, l1_ratio=0.2, max_iter=10000, random_state=123))])
        model.fit(X, y);  return model, "ENet"
    lgb = make_lgbm_smalldata(); lgb.fit(X, y)
    # LightGBM이 사실상 학습되지 않은 경우 ENet으로 대체
    if getattr(lgb, "best_iteration_", 1) <= 1 or (np.asarray(getattr(lgb, "feature_importances_", []))==0).all():
        model = Pipeline([("scaler", StandardScaler()),
                          ("enet", ElasticNet(alpha=0.03, l1_ratio=0.15, max_iter=10000, random_state=123))])
        model.fit(X, y);  return model, "ENet(fallback)"
    return lgb, "LGBM"

# 회귀 평가 함수 (MAE, RMSE, 상관계수 r, 상대 RMSE 등 계산)
def eval_reg(hold, L, model, feats):
    y = hold[f"{TARGET_CONT}_L{L}"].values
    X = hold[feats].replace([np.inf,-np.inf], np.nan)
    ok = ~np.isnan(y)
    y, X = y[ok], X[ok]
    if len(y)==0 or X.shape[1]==0:
        return {"MAE":np.nan,"RMSE":np.nan,"r":np.nan,"rRMSE":np.nan}, np.array([]), np.array([])
    ph = model.predict(X.fillna(X.median()))
    mae  = mean_absolute_error(y, ph)
    rmse = np.sqrt(mean_squared_error(y, ph))
    r    = pearsonr(y, ph)[0] if (len(y)>2 and np.std(ph)>0 and np.std(y)>0) else np.nan
    rrmse = rmse / (np.nanmean(y)+1e-9)
    return {"MAE":mae,"RMSE":rmse,"r":r,"rRMSE":rrmse}, ph, y

# 데이터가 거의 없거나 타깃이 모두 NaN인 경우 사용할 안전한 평균값 계산
def safe_mean_constant(sub_df, target_col, L):
    mu = np.nanmean(sub_df[f"{target_col}_L{L}"].values)
    if not np.isfinite(mu): mu = np.nanmean(sub_df[target_col].values)
    if not np.isfinite(mu): mu = 0.0
    return float(mu)

# 회귀 모델 및 사용 피처 저장 경로 생성
os.makedirs("/content/artifacts_reg", exist_ok=True)
reg_models = {}; feature_used_by_L = {}

# 리드 L=0~MAX_L까지 순회하며 개별 회귀 모델 학습
for L in range(0, MAX_L+1):
    Xtr, ytr, used_feats = prepare_xy(train_r, L, REG_FEATURES, TARGET_CONT)
    if len(ytr) < MIN_Y or len(used_feats)==0:
        mu = safe_mean_constant(train_r, TARGET_CONT, L)
        mdl, tag = DummyRegressor(strategy="constant", constant=mu), "Mean"
        mdl.fit(pd.DataFrame({"c":[0.0]}), [mu]); used_feats=[]
    else:
        mdl, tag = train_regressor_small_data(Xtr, ytr)
    reg_models[L] = mdl; feature_used_by_L[L] = used_feats
    print(f"[L={L}] model={tag}, n={len(ytr)}, used_feats={len(used_feats)}")

# 리드별 모델 dict와 사용 피처 dict 저장
joblib.dump(reg_models, "/content/artifacts_reg/reg_L0_7_rowlead.joblib")
with open("/content/artifacts_reg/reg_features_used.json","w") as f: json.dump(feature_used_by_L, f, indent=2)
print("Saved: /content/artifacts_reg/reg_L0_7_rowlead.joblib, reg_features_used.json")

# 훈련 데이터의 피처별 중앙값 (결측 대체에 일관되게 사용)
train_medians = train_r[REG_FEATURES].replace([np.inf,-np.inf], np.nan).median()

# 2023, 2024에 대해 리드별 회귀 성능표 출력
for name, hold in {"2023":test23_r, "2024":ext24_r}.items():
    rows=[]
    for L in range(0, MAX_L+1):
        featsL = feature_used_by_L[L]
        if len(featsL)==0:
            metr = {"MAE":np.nan,"RMSE":np.nan,"r":np.nan,"rRMSE":np.nan}
        else:
            y = hold[f"{TARGET_CONT}_L{L}"].values
            X = hold[featsL].replace([np.inf,-np.inf], np.nan)
            ok = ~np.isnan(y); y, X = y[ok], X[ok]
            if len(y)==0 or X.shape[1]==0:
                metr = {"MAE":np.nan,"RMSE":np.nan,"r":np.nan,"rRMSE":np.nan}
            else:
                X = X.fillna(train_medians.reindex(featsL))
                ph = reg_models[L].predict(X)
                mae  = mean_absolute_error(y, ph)
                rmse = np.sqrt(mean_squared_error(y, ph))
                r    = pearsonr(y, ph)[0] if (len(y)>2 and np.std(ph)>0 and np.std(y)>0) else np.nan
                rrmse = rmse / (np.nanmean(y)+1e-9)
                metr = {"MAE":mae,"RMSE":rmse,"r":r,"rRMSE":rrmse}
        rows.append({"L":L, **metr})
    print(f"[{name}]")
    try:
        from IPython.display import display
        display(pd.DataFrame(rows))
    except:
        print(pd.DataFrame(rows).to_string(index=False))

# 훈련 데이터 분포를 기준으로 FFDRI/FRI_norm 등급 구간 계산
score = train_r[TARGET_CONT].values
valid_n = np.isfinite(score).sum()
q = np.nanquantile(score, [0.5, 0.65, 0.85]) if valid_n>10 else [np.nanquantile(score,p) for p in [0.25,0.5,0.75]]
grades = {"Low":[-1e18,float(q[0])],"Moderate":[float(q[0]),float(q[1])],
          "High":[float(q[1]),float(q[2])],"VeryHigh":[float(q[2]),1e18]}
with open("/content/artifacts_reg/grades_reg.json","w") as f: json.dump(grades, f, indent=2)
print("Saved: /content/artifacts_reg/grades_reg.json")

# 연속 점수를 등급 문자열로 변환하는 헬퍼
def score_to_grade(x):
    for k,(a,b) in grades.items():
        if a<=x<b: return k
    return "VeryHigh"

# 2023, 2024에 대해 각 리드별 예측값과 L0 등급을 포함한 CSV 저장
for name, hold in {"2023":test23_r, "2024":ext24_r}.items():
    if len(hold)==0:
        print(f"[{name}] no rows.")
        continue
    out = hold[["date","lon","lat","season_cluster"]].copy()
    for L in range(0, MAX_L+1):
        featsL = feature_used_by_L[L]
        if len(featsL)==0:
            mu = safe_mean_constant(train_r, TARGET_CONT, L)
            yhat = np.full(len(out), mu, dtype=float)
        else:
            Xh = hold[featsL].replace([np.inf,-np.inf], np.nan)
            Xh = Xh.fillna(train_medians.reindex(featsL))
            yhat = reg_models[L].predict(Xh)
        out[f"{TARGET_CONT}_hat_L{L}"] = yhat
    out["grade_hat_L0"] = [score_to_grade(x) for x in out[f"{TARGET_CONT}_hat_L0"].values]
    p = f"/content/pred_reg_{name}_rowlead.csv"; out.to_csv(p, index=False)
    print("Saved:", p)
    try: files.download(p)
    except: pass

# 회귀 아티팩트 폴더를 ZIP으로 묶어 저장
shutil.make_archive("/content/artifacts_reg_bundle_rowlead", "zip", "/content/artifacts_reg")
try: files.download("/content/artifacts_reg_bundle_rowlead.zip")
except: print("[local] ZIP 저장 완료: /content/artifacts_reg_bundle_rowlead.zip")

# 디버그용: 리드별 실제 사용 피처 목록 일부 출력
for L in range(0, MAX_L+1):
    feats = feature_used_by_L.get(L, [])
    print(f"[DEBUG] L={L} used_feats={len(feats)} -> {feats[:10]}")



### 클러스터링

# NDVI, 누적 일사량(DSR_total_MJm^2), FFDRI를 이용해 산림 상태/위험 유형을 군집화하는 코드
# 입력:
#   - df_out_removed (전처리 후 이상치 제거된 데이터프레임, NDVI/DSR/FFDRI 포함)
# 처리:
#   - RobustScaler로 이상치에 강건한 스케일링
#   - KMeans(k=2)로 두 개의 클러스터로 나눔
#   - PCA 2차원으로 차원 축소 후 군집을 시각화
# 출력:
#   - df["cluster_k2"]에 군집 레이블 추가
#   - PCA 좌표(_PCA1, _PCA2) 및 산점도 플롯

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

RANDOM_STATE = 42
FEATURES = ['NDVI', 'DSR_total_MJm^2','FFDRI']

# 이전 단계에서 생성된 df_out_removed를 복사해서 사용
df = df_out_removed.copy()

# 원본 피처만 추출
X_raw = df[FEATURES].copy()

# 이상치에 덜 민감한 RobustScaler로 스케일링
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_raw)

# 스케일링된 값을 별도 DataFrame으로 보관 (필요 시 추가 분석용)
X_scaled_df = pd.DataFrame(X_scaled, columns=[f"scaled_{c}" for c in FEATURES], index=df.index)

# KMeans(k=2) 군집화 수행
kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10)
labels = kmeans.fit_predict(X_scaled)
df["cluster_k2"] = labels

# PCA를 이용해 2차원으로 투영 (시각화용)
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
df["_PCA1"], df["_PCA2"] = X_pca[:, 0], X_pca[:, 1]

# PCA 공간에서 군집별 산점도 시각화
plt.figure(figsize=(7, 6))
sns.scatterplot(data=df, x="_PCA1", y="_PCA2", hue="cluster_k2",
                s=50, alpha=0.85, edgecolor="none")
plt.title("K-Means (k=2) with Robust Scaling")
plt.xlabel(f"PCA1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PCA2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()


### 좌표 추출 및 위험 후보 지역 선정

# 좌표별(2019-01-01 ~ 2024-12-31)로 ERA5 기상 일평균을 추출하는 스크립트
# 입력:
#   - /content/sites_by_type.csv  (필수 컬럼: lat, lon)
# 출력:
#   - /content/fri_inputs_by_row.csv
# 포함 변수:
#   - Tmean  (일 평균 기온, ℃)
#   - RH     (상대습도, %)
#   - WSPD   (풍속, m/s)
#   - TP_mm  (일강수량, mm/day)
# 특징:
#   - ERA5-Land 우선 사용, 결측 시 ERA5-Global로 보완
#   - 타임아웃 완화 옵션(tileScale) 및 지수 백오프를 이용한 재시도
#   - 일정 간격마다 CSV를 저장하는 체크포인트 방식으로 이어서 실행 가능

!pip -q install earthengine-api pandas tqdm

import ee, pandas as pd, datetime as dt, time
from google.colab import files
from tqdm import tqdm

# GEE 프로젝트 및 경로, 샘플링 관련 파라미터 설정
PROJECT_ID = "YOUR_GEE_PROJECT_ID"     # 본인 GEE 프로젝트 ID
CSV_PATH   = "/content/sites_by_type.csv"
OUT_CSV    = "/content/fri_inputs_by_row.csv"

# ERA5 해상도(~9km) 기준 권장 반경/스케일
SAMPLE_RADIUS_M = 9000      # 포인트 주변 평균을 낼 반경
SAMPLE_SCALE_M  = 9000      # reduceRegions 스케일
TILESCALE       = 4         # 서버 메모리/타임아웃 조절용 (필요 시 8까지 조정 가능)

# 추출 날짜 범위
DATE_START = dt.date(2019, 1, 1)
DATE_END   = dt.date(2024,12,31)

# 필요 시 날짜를 통째로 이동시킬 보정 값 (일 단위, 보통 0 유지)
DATE_SHIFT_DAYS = 0

# 체크포인트 저장 주기(일 단위)
CHECKPOINT_EVERY = 50

# getInfo 재시도 횟수와 백오프 계수
MAX_RETRY   = 6
BACKOFF_BASE= 1.5

# GEE 초기화
try:
    ee.Initialize(project=PROJECT_ID)
except Exception:
    ee.Authenticate(project=PROJECT_ID)
    ee.Initialize(project=PROJECT_ID)
print("[GEE] initialized")
ee.data.setDeadline(120000)  # 요청 타임리밋(ms)

# 입력 CSV 존재 여부 확인 후 필요 시 업로드
try:
    _ = open(CSV_PATH, "r")
except FileNotFoundError:
    print("[UPLOAD] sites_by_type.csv 파일을 업로드하세요")
    uploaded = files.upload()
    if "sites_by_type.csv" not in uploaded:
        name = next(iter(uploaded))
        with open(CSV_PATH, "wb") as f:
            f.write(uploaded[name])
        print(f"[INFO] 업로드 파일을 sites_by_type.csv로 저장: {name} → sites_by_type.csv")
    else:
        print("[INFO] sites_by_type.csv 업로드 완료")

# 좌표 목록 로드 및 전처리
df_sites = pd.read_csv(CSV_PATH, encoding="utf-8-sig", engine="python")
df_sites.columns = [c.strip().lower().replace("\ufeff","") for c in df_sites.columns]

need = {"lat","lon"}
if not (need <= set(df_sites.columns)):
    raise ValueError(f"필수 컬럼 누락. 현재 헤더: {list(df_sites.columns)} (필요: {sorted(need)})")

df_sites["lon"] = pd.to_numeric(df_sites["lon"], errors="coerce")
df_sites["lat"] = pd.to_numeric(df_sites["lat"], errors="coerce")
df_sites = df_sites.dropna(subset=["lon","lat"]).drop_duplicates(subset=["lon","lat"]).reset_index(drop=True)
df_sites["pid"] = df_sites.index.astype(int)

print(f"[INFO] sites: {len(df_sites)} rows (예상: 15)")
print(df_sites.head())

# 날짜 리스트 생성
def daterange(d0, d1):
    cur = d0
    while cur <= d1:
        yield cur
        cur = cur + dt.timedelta(days=1)

unique_dates = [d for d in daterange(DATE_START, DATE_END)]
if DATE_SHIFT_DAYS != 0:
    unique_dates = [d + dt.timedelta(days=DATE_SHIFT_DAYS) for d in unique_dates]
unique_dates_str = [d.isoformat() for d in unique_dates]
print(f"[INFO] date range: {unique_dates_str[0]} ~ {unique_dates_str[-1]}  (days={len(unique_dates_str)})  # 예상=2192")

# ERA5-Land, ERA5-Global 컬렉션 정의
ERA5_LAND   = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
ERA5_GLOBAL = ee.ImageCollection("ECMWF/ERA5/HOURLY")

# 시간당 이미지를 일평균/일합산 변수로 변환하는 함수
def _per_hour_to_vars(im):
    # 온도(K)를 ℃로 변환
    T  = im.select("temperature_2m").subtract(273.15).rename("Tmean")
    Td = im.select("dewpoint_temperature_2m").subtract(273.15)
    # Magnus 공식을 이용한 상대습도 계산
    a, b = 17.625, 243.04
    RH = Td.expression(
        "100*exp(a*Td/(b+Td) - a*T/(b+T))",
        {"a": a, "b": b, "Td": Td, "T": T}
    ).rename("RH")
    # 10m 바람성분에서 풍속(m/s) 계산
    U = im.select("u_component_of_wind_10m")
    V = im.select("v_component_of_wind_10m")
    WSPD = U.pow(2).add(V.pow(2)).sqrt().rename("WSPD")
    # 강수량(m)을 mm로 변환
    TP = im.select("total_precipitation").multiply(1000).rename("TP_mm")
    return T.addBands([RH, WSPD, TP])

# 특정 날짜의 일평균/일합산 이미지 생성
def _daily_from(ic, date_str):
    d0 = ee.Date(date_str); d1 = d0.advance(1, "day")
    hourly = ic.filterDate(d0, d1).map(_per_hour_to_vars)
    Tmean = hourly.select("Tmean").mean()
    RH    = hourly.select("RH").mean()
    WSPD  = hourly.select("WSPD").mean()
    TP    = hourly.select("TP_mm").sum()
    return Tmean.addBands([RH, WSPD, TP])

# ERA5-Land와 ERA5-Global를 결합한 일 단위 이미지 생성
def daily_era5(date_str):
    land  = _daily_from(ERA5_LAND,   date_str)
    globe = _daily_from(ERA5_GLOBAL, date_str)
    fused = land.unmask(globe)  # Land에서 누락된 픽셀은 Global로 대체
    return fused.set({"date": date_str})

# getInfo에 지수 백오프 재시도 적용
def safe_getInfo(ee_obj):
    for k in range(MAX_RETRY):
        try:
            return ee_obj.getInfo()
        except Exception as e:
            wait = BACKOFF_BASE**k
            print(f"[WARN] getInfo retry {k+1}/{MAX_RETRY} in {wait:.2f}s -> {e}")
            time.sleep(wait)
    return ee_obj.getInfo()

# 이어달리기를 위한 기존 결과 로드 (있다면 재사용)
try:
    df_out = pd.read_csv(OUT_CSV)
    done_keys = set(zip(df_out['date'].astype(str), df_out['pid'].astype(int)))
    rows_out = df_out.to_dict('records')
    print(f"[RESUME] existing rows: {len(rows_out)} (이어달리기)")
except Exception:
    rows_out = []
    done_keys = set()

# 날짜 × 좌표 조합에 대해 샘플링 실행
total_dates = len(unique_dates_str)
print(f"[INFO] sampling by date × {len(df_sites)} sites  (dates={total_dates})")

processed_since_cp = 0
for idx, d in enumerate(tqdm(unique_dates_str, desc="[sampling by date]"), 1):
    # 이 날짜와 pid 조합이 모두 완료되어 있으면 건너뜀
    if all((d, int(pid)) in done_keys for pid in df_sites['pid'].tolist()):
        continue

    # 날짜별 좌표를 FeatureCollection으로 구성, 각 포인트는 버퍼 영역으로 확장
    fc = ee.FeatureCollection([
        ee.Feature(
            ee.Geometry.Point([float(r["lon"]), float(r["lat"])]).buffer(SAMPLE_RADIUS_M),
            {"pid": int(r["pid"]), "lon": float(r["lon"]), "lat": float(r["lat"])}
        )
        for _, r in df_sites.iterrows()
    ])

    # 해당 날짜의 ERA5 일별 이미지 구성
    img = daily_era5(d)

    # reduceRegions로 모든 포인트에 대해 평균값 추출
    reduced = img.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=SAMPLE_SCALE_M,
        tileScale=TILESCALE
    ).map(lambda f: f.set({"date": d}))

    data  = safe_getInfo(reduced)
    feats = data.get("features", []) if data else []
    for f in feats:
        p = f.get("properties", {})
        key = (str(p.get("date")), int(p.get("pid")))
        if key in done_keys:
            continue
        rows_out.append({
            "date":  p.get("date"),
            "pid":   p.get("pid"),
            "lon":   p.get("lon"),
            "lat":   p.get("lat"),
            "Tmean": p.get("Tmean_mean", p.get("Tmean")),
            "RH":    p.get("RH_mean",    p.get("RH")),
            "WSPD":  p.get("WSPD_mean",  p.get("WSPD")),
            "TP_mm": p.get("TP_mm_mean", p.get("TP_mm")),
        })
        done_keys.add(key)

    processed_since_cp += 1
    if processed_since_cp >= CHECKPOINT_EVERY or idx == total_dates:
        tmp = pd.DataFrame(rows_out).sort_values(["date","pid"]).reset_index(drop=True)
        tmp.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        print(f"[CP] saved {len(tmp)} rows at {d}  ({idx}/{total_dates})")
        processed_since_cp = 0

# 최종 결과 저장 및 품질 점검
df_out = pd.DataFrame(rows_out).sort_values(["date","pid"]).reset_index(drop=True)
df_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

num_total   = len(df_out)
num_dates   = df_out["date"].nunique()
num_sites   = df_out["pid"].nunique()
num_all_nan = df_out[["Tmean","RH","WSPD","TP_mm"]].isna().all(axis=1).sum()
num_any_nan = df_out[["Tmean","RH","WSPD","TP_mm"]].isna().any(axis=1).sum()

print(f"[SAVED] {OUT_CSV}  rows={num_total}  (sites={num_sites}, dates={num_dates})")
print(f"[QC] all-NaN rows = {num_all_nan} / any-NaN rows = {num_any_nan}")

# 좌표가 한국 대략 범위를 벗어나는 경우 경고
bad = df_sites[(df_sites["lat"]<33)|(df_sites["lat"]>39)|(df_sites["lon"]<124)|(df_sites["lon"]>132)]
if len(bad):
    print("[WARN] KR bounds outliers (first 5):")
    print(bad[["pid","lat","lon"]].head())

# CSV 다운로드 (Colab 환경)
files.download(OUT_CSV)



### 하루 일조량 추출

# MODIS/061/MCD18A1에서 3시간 단위 DSR 밴드를 합산해
# 하루 누적 일사량(DSR_total_MJm^2)을 구하는 스크립트
# 입력:
#   - TOTAL_DWI_with_NDVI11111_filled.xlsx  (date, lon, lat 포함)
# 출력:
#   - TOTAL_DWI_with_NDVI11111_filled_with_DSR.xlsx
# 처리:
#   - 각 행의 날짜/좌표에 대해 DSR_total_MJm^2 계산
#   - 에러 시 NaN 처리, 전체를 엑셀로 저장

import ee, pandas as pd, numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

# GEE 초기화 (project 지정, 실패 시 인증)
try:
    ee.Initialize(project='YOUR_GEE_PROJECT_ID')
    print("GEE initialized (project=YOUR_GEE_PROJECT_ID).")
except Exception:
    ee.Authenticate()
    ee.Initialize()
    print("GEE authenticated & initialized.")

# 파일 경로 및 컬럼 이름
INPUT_XLSX = "/content/TOTAL_DWI_with_NDVI11111_filled.xlsx"
OUTPUT_XLSX = "/content/TOTAL_DWI_with_NDVI11111_filled_with_DSR.xlsx"

DATE_COL = "date"
LON_COL  = "lon"
LAT_COL  = "lat"

# 사용할 MODIS DSR 밴드 목록
MCD18A1_DSR_BANDS = [
    'GMT_0000_DSR','GMT_0300_DSR','GMT_0600_DSR','GMT_0900_DSR',
    'GMT_1200_DSR','GMT_1500_DSR','GMT_1800_DSR','GMT_2100_DSR'
]

def daily_dsr_total_MJm2(lon: float, lat: float, date_str: str, scale: int = 1000):
    """
    특정 날짜(UTC)와 좌표에서 MODIS/061/MCD18A1의 3시간 DSR 밴드를 합산해
    하루 누적 복사량(MJ/m^2)을 반환.
    이미지 부재 또는 오류 시 np.nan 반환.
    """
    try:
        d0 = ee.Date(date_str)
        d1 = d0.advance(1, 'day')
        pt = ee.Geometry.Point([float(lon), float(lat)])

        ic = ee.ImageCollection('MODIS/061/MCD18A1').filterDate(d0, d1)
        img = ic.first()
        if img.getInfo() is None:
            return np.nan

        band_names = img.bandNames()
        actual_bands = ee.List(MCD18A1_DSR_BANDS).filter(ee.Filter.inList('item', band_names))

        # 3시간 단위 W/m^2 값을 합산 후 J/m^2, MJ/m^2로 변환
        dsr_sum_wm2 = img.select(actual_bands).reduce(ee.Reducer.sum())
        dsr_total_Jm2 = dsr_sum_wm2.multiply(10800)
        dsr_total_MJm2 = dsr_total_Jm2.divide(1e6)

        val = dsr_total_MJm2.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=pt,
            scale=scale,
            bestEffort=True
        ).get('sum')

        # 밴드 키 이름이 달라진 경우를 대비한 보완 로직
        if val is None:
            keys = dsr_total_MJm2.bandNames().getInfo()
            if keys:
                val = dsr_total_MJm2.reduceRegion(
                    reducer=ee.Reducer.first(),
                    geometry=pt,
                    scale=scale,
                    bestEffort=True
                ).get(keys[0])

        result = ee.Number(val).getInfo() if val is not None else np.nan
        return float(result) if result is not None else np.nan

    except Exception:
        return np.nan

# 입력 엑셀 읽기
df = pd.read_excel(INPUT_XLSX)

# 날짜를 문자열(YYYY-MM-DD)로 통일
def to_date_str(x):
    if pd.isna(x):
        return None
    if isinstance(x, (datetime, pd.Timestamp)):
        return x.strftime("%Y-%m-%d")
    s = str(x).strip()
    return s[:10]

if DATE_COL not in df.columns or LON_COL not in df.columns or LAT_COL not in df.columns:
    raise ValueError(f"필수 컬럼이 누락됨: '{DATE_COL}', '{LON_COL}', '{LAT_COL}' 필요")

df["_date_str"] = df[DATE_COL].apply(to_date_str)

# 각 행에 대해 DSR_total_MJm^2 계산
vals = []
for i, row in tqdm(df.iterrows(), total=len(df), desc="Computing DSR_total_MJm^2"):
    date_str = row["_date_str"]
    lon = row[LON_COL]
    lat = row[LAT_COL]

    if pd.isna(date_str) or pd.isna(lon) or pd.isna(lat):
        vals.append(np.nan)
        continue

    v = daily_dsr_total_MJm2(float(lon), float(lat), date_str)
    vals.append(v)

df["DSR_total_MJm^2"] = vals

df.drop(columns=["_date_str"], inplace=True)
df.to_excel(OUTPUT_XLSX, index=False)
print("Saved:", OUTPUT_XLSX)



### NDVI 추출 1차 (DWI 기반 전체 데이터셋)

# LANDSAT NDVI와 MODIS NDVI를 우선순위로 사용해
# 각 날짜·좌표에 대해 NDVI를 채우는 스크립트
# 입력:
#   - TOTAL_DWI_CatForest_cleaned.xlsx
# 출력:
#   - TOTAL_DWI_with_NDVI11111.xlsx
# 특징:
#   - LANDSAT 우선, 실패 시 MODIS
#   - 시간 창 및 공간 버퍼 크기를 점진적으로 확장
#   - NDVI, 사용 날짜, 사용 센서, 윈도우 정보, 버퍼 정보까지 기록
#   - 중간 체크포인트 엑셀 저장 기능 포함

import ee, pandas as pd, numpy as np, os, math, traceback, time
from datetime import datetime, timedelta
from tqdm import tqdm

# GEE 초기화
try:
    ee.Initialize(project='YOUR_GEE_PROJECT_ID')
    print("GEE initialized")
except Exception as e:
    print("Init failed, authenticating...", repr(e))
    ee.Authenticate()
    ee.Initialize()
    print("GEE authenticated & initialized")

# 경로 및 체크포인트 설정
INPUT_PATH   = "/content/TOTAL_DWI_CatForest_cleaned.xlsx"
OUTPUT_PATH  = "/content/TOTAL_DWI_with_NDVI11111.xlsx"
CHECKPOINT   = "/content/_ndvi_checkpoint.xlsx"
LOG_PATH     = "/content/_ndvi_log.txt"

DATE_COL, LAT_COL, LON_COL = "date", "lat", "lon"

# 사용할 GEE NDVI 컬렉션
LANDSAT_ID = "LANDSAT/COMPOSITES/C02/T1_L2_8DAY_NDVI"
MODIS_ID   = "MODIS/061/MOD13Q1"
BAND       = "NDVI"

# 시간 창(일 단위)과 공간 버퍼(미터) 단계
WINDOW_STEPS = [16, 24, 32, 45, 60]
BUFFER_STEPS = [120, 300, 600, 1000]

SAVE_EVERY   = 100
PRINT_EVERY  = 50

# 타깃 날짜와 가장 가까운 영상 하나를 선택
def _closest_image(col, target_date_str: str) -> ee.Image:
    d0 = ee.Date(target_date_str)
    def add_diff(img):
        t = ee.Date(img.get('system:time_start'))
        return img.set({
            'date_diff': t.difference(d0, 'day').abs(),
            'img_date': t.format('YYYY-MM-dd')
        })
    return ee.Image(col.map(add_diff).sort('date_diff').first())

# 하나의 GEE NDVI 데이터셋(LANDSAT 또는 MODIS)에 대해
# 시간 창 및 버퍼를 점진적으로 늘려가며 NDVI를 추출
def _try_dataset(lat: float, lon: float, date_str: str,
                 dataset_id: str, band: str, is_modis: bool):

    point = ee.Geometry.Point([float(lon), float(lat)])
    scale = 250 if dataset_id == MODIS_ID else 30
    sfac  = 0.0001 if is_modis else 1.0

    for win in WINDOW_STEPS:
        start = ee.Date((datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=win)).strftime("%Y-%m-%d"))
        end   = ee.Date((datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=win+1)).strftime("%Y-%m-%d"))
        col = ee.ImageCollection(dataset_id).filterDate(start, end).select(band)
        if col.size().getInfo() == 0:
            continue

        img = _closest_image(col, date_str)
        used_date = img.get('img_date').getInfo()

        # 1차: 버퍼 영역에서 median NDVI 계산
        for buf in BUFFER_STEPS:
            geom = point.buffer(buf)
            try:
                d = img.reduceRegion(
                    reducer=ee.Reducer.median(),
                    geometry=geom, scale=scale,
                    bestEffort=True, maxPixels=1e8
                ).get(band).getInfo()
            except Exception:
                d = None
            if d is not None:
                v = max(-1.0, min(1.0, float(d) * sfac))
                return v, used_date, win, buf, dataset_id

        # 2차: sample을 이용한 포인트 샘플링
        try:
            fc = img.sample({
                'region': point, 'scale': scale,
                'numPixels': 1, 'geometries': False, 'tileScale': 2
            })
            d2 = None if fc.size().getInfo() == 0 else ee.Feature(fc.first()).get(band).getInfo()
        except Exception:
            d2 = None
        if d2 is not None:
            v2 = max(-1.0, min(1.0, float(d2) * sfac))
            return v2, used_date, win, 0, dataset_id

        # 3차: unmask + focal_mean으로 빈 픽셀 보완 후 다시 버퍼 median
        filled = img.select(band).unmask().focal_mean(radius=150, units='meters', iterations=1)
        for buf2 in BUFFER_STEPS:
            geom2 = point.buffer(buf2)
            try:
                d3 = filled.reduceRegion(
                    reducer=ee.Reducer.median(),
                    geometry=geom2, scale=scale,
                    bestEffort=True, maxPixels=1e8
                ).get(band).getInfo()
            except Exception:
                d3 = None
            if d3 is not None:
                v3 = max(-1.0, min(1.0, float(d3) * sfac))
                return v3, used_date, win, buf2, dataset_id

    return None, None, None, None, None

# LANDSAT → MODIS 순으로 NDVI를 시도 후 값 반환
def get_ndvi(lat: float, lon: float, date_str: str):
    v, used, win, buf, src = _try_dataset(lat, lon, date_str, LANDSAT_ID, BAND, False)
    if v is not None:
        return v, used, os.path.basename(src), win, buf

    v, used, win, buf, src = _try_dataset(lat, lon, date_str, MODIS_ID, BAND, True)
    if v is not None:
        return v, used, os.path.basename(src), win, buf
    return np.nan, None, None, None, None

# 입력 엑셀 로드 및 날짜 정규화
df = pd.read_excel(INPUT_PATH)
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.strftime("%Y-%m-%d")

# NDVI 관련 컬럼이 없으면 생성
for c in ["NDVI", "NDVI_used_date", "NDVI_source", "NDVI_window", "NDVI_buffer"]:
    if c not in df.columns:
        df[c] = np.nan if c == "NDVI" else None

errors = []
filled = 0

# 각 행에 대해 GEE에서 NDVI 추출
for i, row in tqdm(df.iterrows(), total=len(df)):
    lat, lon, d = row[LAT_COL], row[LON_COL], row[DATE_COL]

    if pd.isna(lat) or pd.isna(lon) or pd.isna(d):
        df.at[i, "NDVI"]            = np.nan
        df.at[i, "NDVI_used_date"]  = None
        df.at[i, "NDVI_source"]     = None
        df.at[i, "NDVI_window"]     = None
        df.at[i, "NDVI_buffer"]     = None
        continue

    try:
        ndvi, used_date, src, win, buf = get_ndvi(float(lat), float(lon), str(d))
        df.at[i, "NDVI"]            = ndvi
        df.at[i, "NDVI_used_date"]  = used_date
        df.at[i, "NDVI_source"]     = src
        df.at[i, "NDVI_window"]     = win
        df.at[i, "NDVI_buffer"]     = buf
        if not (pd.isna(ndvi) or (isinstance(ndvi, float) and math.isnan(ndvi))):
            filled += 1
    except Exception as e:
        errors.append(f"[row {i}] {type(e).__name__}: {repr(e)}")
        df.at[i, "NDVI"]            = np.nan
        df.at[i, "NDVI_used_date"]  = None
        df.at[i, "NDVI_source"]     = None
        df.at[i, "NDVI_window"]     = None
        df.at[i, "NDVI_buffer"]     = None

    # 진행 상황 출력 및 체크포인트 저장
    if (i + 1) % PRINT_EVERY == 0:
        print(f"[{i+1}/{len(df)}] filled so far: {filled}")
    if (i + 1) % SAVE_EVERY == 0:
        df.to_excel(CHECKPOINT, index=False)
        print("Checkpoint saved ->", CHECKPOINT)

# 좌표별 그룹 내에서 NDVI 선형 보간으로 빈 구간 채우기
df = df.sort_values([LAT_COL, LON_COL, DATE_COL])
def _fill_group(g):
    g["NDVI"] = g["NDVI"].astype(float)
    g["NDVI"] = g["NDVI"].interpolate(method="linear", limit_direction="both")
    return g
df = df.groupby([LAT_COL, LON_COL], group_keys=False).apply(_fill_group)

# 최종 저장 및 에러 로그 기록
df.to_excel(OUTPUT_PATH, index=False)
print("Saved ->", OUTPUT_PATH, "| rows:", len(df), "| filled:", int((~pd.isna(df['NDVI'])).sum()))

if errors:
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(errors))
    print(f"Logged {len(errors)} exceptions -> {LOG_PATH}")
else:
    print("No exceptions.")



### NDVI 추출 2차 (0값 보완용)

# NDVI가 0으로 채워진 행에 대해
# Landsat8, Sentinel-2, MODIS, 월별 기후값 순으로 시도해 NDVI를 다시 채우는 스크립트
# 입력:
#   - TOTAL_DWI_with_NDVI11111.xlsx (기존 NDVI 포함)
# 출력:
#   - TOTAL_DWI_with_NDVI11111_filled.xlsx

try:
    ee.Initialize(project='YOUR_GEE_PROJECT_ID')
    print("GEE initialized")
except Exception:
    ee.Authenticate()
    ee.Initialize()
    print("GEE authenticated & initialized")

INPUT_XLSX  = "/content/TOTAL_DWI_with_NDVI11111.xlsx"
OUTPUT_XLSX = "/content/TOTAL_DWI_with_NDVI11111_filled.xlsx"

DATE_COL = None
LON_COL  = None
LAT_COL  = None
NDVI_COL = None

BUF_METERS   = 120
LS_WIN_DAYS  = 32
S2_WIN_DAYS  = 20
CLIM_YEARS   = 5
S2_CLOUD_TH  = 40
REDUCER      = ee.Reducer.mean()

def _safe_date(s):
    if pd.isna(s): return None
    if isinstance(s, (pd.Timestamp, datetime)):
        return pd.to_datetime(s).strftime("%Y-%m-%d")
    try:
        return pd.to_datetime(str(s)).strftime("%Y-%m-%d")
    except Exception:
        return None

def _value_at(img, band, geom, scale, reducer=REDUCER):
    d = img.select(band).reduceRegion(
        reducer=reducer, geometry=geom, scale=scale,
        maxPixels=1e8, bestEffort=True
    )
    return d.get(band)

def _closest(col, target_date):
    t = ee.Date(target_date)
    col2 = col.map(lambda im: im.set('d', ee.Date(im.get('system:time_start')).difference(t,'day').abs()))
    return ee.Image(col2.sort('d').first())

# Landsat L2 기반 NDVI
def _landsat_ndvi(point, date_str):
    t  = ee.Date(date_str)
    st = t.advance(-LS_WIN_DAYS,'day')
    en = t.advance( LS_WIN_DAYS,'day').advance(1,'day')

    col = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
           .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
           .filterDate(st, en)
           .filterBounds(point))

    def _mask_map(img):
        qa = img.select('QA_PIXEL')
        cloud  = qa.bitwiseAnd(1<<3).neq(0)
        shadow = qa.bitwiseAnd(1<<4).neq(0)
        snow   = qa.bitwiseAnd(1<<5).neq(0)
        mask = cloud.Or(shadow).Or(snow).Not()
        sr = img.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7']).multiply(0.0000275).add(-0.2)
        nir = sr.select('SR_B5')
        red = sr.select('SR_B4')
        ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI').updateMask(mask)
        return ndvi.copyProperties(img, img.propertyNames())

    ndvi_col = col.map(_mask_map)
    if ndvi_col.size().getInfo() == 0:
        return None, None
    best = _closest(ndvi_col, t)
    geom = point if BUF_METERS==0 else point.buffer(BUF_METERS)
    val  = _value_at(best, 'NDVI', geom, 30)
    return val, ee.Date(best.get('system:time_start')).format('YYYY-MM-dd')

# Sentinel-2 기반 NDVI
def _s2_ndvi(point, date_str):
    t  = ee.Date(date_str)
    st = t.advance(-S2_WIN_DAYS,'day')
    en = t.advance( S2_WIN_DAYS,'day').advance(1,'day')

    s2  = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterDate(st, en).filterBounds(point))

    prob = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY').filterDate(st, en).filterBounds(point)

    joined = ee.Join.saveFirst('cloud_prob').apply(
        primary=s2,
        secondary=prob,
        condition=ee.Filter.equals(leftField='system:index', rightField='system:index')
    )

    def _map(img):
        img = ee.Image(img)
        cp  = ee.Image(img.get('cloud_prob'))
        cp  = ee.Algorithms.If(cp, ee.Image(cp).select('probability'), ee.Image(0).rename('probability'))
        cp  = ee.Image(cp)

        scl = img.select('SCL')
        cloudMask = cp.gt(S2_CLOUD_TH) \
            .Or(scl.eq(3)).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11)).Not()

        b4 = img.select('B4').multiply(0.0001)
        b8 = img.select('B8').multiply(0.0001)
        ndvi = b8.subtract(b4).divide(b8.add(b4)).rename('NDVI').updateMask(cloudMask)
        return ndvi.copyProperties(img, img.propertyNames())

    ndvi_col = ee.ImageCollection(joined).map(_map)
    if ndvi_col.size().getInfo() == 0:
        return None, None
    best = _closest(ndvi_col, t)
    geom = point if BUF_METERS==0 else point.buffer(BUF_METERS)
    val  = _value_at(best, 'NDVI', geom, 10)
    return val, ee.Date(best.get('system:time_start')).format('YYYY-MM-dd')

# MODIS 기반 NDVI
def _modis_ndvi(point, date_str):
    t = ee.Date(date_str)
    col = (ee.ImageCollection('MODIS/061/MOD13Q1')
           .filterDate(t.advance(-40,'day'), t.advance(40,'day'))
           .filterBounds(point)
           .select('NDVI'))

    if col.size().getInfo() == 0:
        return None, None
    best = _closest(col, t)
    ndvi = ee.Image(best).multiply(0.0001).rename('NDVI')
    geom = point if BUF_METERS==0 else point.buffer(BUF_METERS)
    val  = _value_at(ndvi, 'NDVI', geom, 250)
    return val, ee.Date(best.get('system:time_start')).format('YYYY-MM-dd')

# 최근 5년 월별 MODIS 클라이마토로 NDVI 근사
def _modis_monthly_clim(point, date_str):
    t = ee.Date(date_str)
    month = t.get('month')
    start = t.advance(-CLIM_YEARS, 'year')
    col = (ee.ImageCollection('MODIS/061/MOD13Q1')
           .filterDate(start, t)
           .filterBounds(point)
           .filter(ee.Filter.calendarRange(month, month, 'month'))
           .select('NDVI')
           .map(lambda im: ee.Image(im).multiply(0.0001).rename('NDVI')))
    if col.size().getInfo() == 0:
        return None, None
    clim = col.median()
    geom = point if BUF_METERS==0 else point.buffer(BUF_METERS)
    val  = _value_at(clim, 'NDVI', geom, 250)
    return val, ee.String('climatology(m=').cat(ee.Number(month).format()).cat(')')

# 여러 소스를 순차적으로 시도하며 NDVI를 얻는 함수
def get_ndvi_with_fallback(lat, lon, date_str):
    pt = ee.Geometry.Point([float(lon), float(lat)])
    funcs = [_landsat_ndvi, _s2_ndvi, _modis_ndvi, _modis_monthly_clim]
    for f in funcs:
        try:
            val, when = f(pt, date_str)
            if val is None:
                continue
            is_null = ee.Algorithms.IsEqual(val, None).getInfo()
            if not is_null:
                return float(ee.Number(val).getInfo())
        except ee.EEException:
            time.sleep(0.5)
            continue
        except Exception:
            continue
    return None

# 입력 엑셀 로드 및 칼럼 자동 탐지
df = pd.read_excel(INPUT_XLSX)
cols_lower = {c: c.lower() for c in df.columns}

def _guess(colnames, keys):
    for c in colnames:
        cl = c.lower()
        if any(k in cl for k in keys):
            return c
    return None

ndvi_col = NDVI_COL or _guess(df.columns, ['ndvi'])
date_col = DATE_COL or _guess(df.columns, ['date','날짜','datetime'])
lon_col  = LON_COL  or _guess(df.columns, ['lon','경도','x'])
lat_col  = LAT_COL  or _guess(df.columns, ['lat','위도','y'])

assert ndvi_col is not None, "NDVI 컬럼을 찾지 못했습니다. NDVI_COL에 명시하세요."
assert date_col is not None, "date 컬럼을 찾지 못했습니다. DATE_COL에 명시하세요."
assert lon_col  is not None and lat_col is not None, "lon/lat 컬럼을 찾지 못했습니다."

print("Detected columns ->",
      f"NDVI:{ndvi_col}, date:{date_col}, lon:{lon_col}, lat:{lat_col}")

# NDVI가 0인 행만 대상으로 GEE에서 다시 추출
mask = (df[ndvi_col].fillna(0) == 0) & df[lon_col].notna() & df[lat_col].notna() & df[date_col].notna()
rows_to_fill = df[mask].copy()
print(f"Rows to fill (NDVI==0): {len(rows_to_fill)} / {len(df)}")

filled_values = []
idx_list = rows_to_fill.index.tolist()

for i in tqdm(idx_list, desc="Filling NDVI"):
    lat = df.at[i, lat_col]
    lon = df.at[i, lon_col]
    date_str = _safe_date(df.at[i, date_col])
    if (lat is None) or (lon is None) or (date_str is None):
        filled_values.append((i, None))
        continue
    try:
        val = get_ndvi_with_fallback(lat, lon, date_str)
    except Exception:
        val = None
    filled_values.append((i, val))

# NDVI 값을 덮어쓰기 (None은 NaN 처리)
for i, val in filled_values:
    if val is not None:
        df.at[i, ndvi_col] = val
    else:
        df.at[i, ndvi_col] = np.nan

df.to_excel(OUTPUT_XLSX, index=False)
print("Saved:", OUTPUT_XLSX)



### 고정 좌표(15개) 기준 NDVI 추출

# 15개 고정 좌표와 날짜에 대해
# LANDSAT8 8일 합성 NDVI → MODIS MOD13Q1 순으로 NDVI를 추출하는 스크립트
# 입력:
#   - wildfire_dataset.csv (lat, lon, date 포함)
# 출력:
#   - wildfire_dataset_with_NDVI.csv

# -*- coding: utf-8 -*-
import ee
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# GEE 초기화
try:
    ee.Initialize(project='YOUR_GEE_PROJECT_ID')
except:
    ee.Authenticate()
    ee.Initialize()

# 경로 설정
INPUT_CSV  = "/content/wildfire_dataset.csv"
OUTPUT_CSV = "/content/wildfire_dataset_with_NDVI.csv"

# 데이터 로드 및 기본 전처리
df = pd.read_csv(INPUT_CSV)
df = df.rename(columns=str.lower)
df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

if "lat" not in df.columns or "lon" not in df.columns:
    raise ValueError("lat, lon, date 컬럼 필요")

# GEE NDVI 컬렉션 참조
LAND8_ID = "LANDSAT/COMPOSITES/C02/T1_L2_8DAY_NDVI"
MODIS_ID = "MODIS/061/MOD13Q1"

LAND8_COL = ee.ImageCollection(LAND8_ID)
MODIS_COL = ee.ImageCollection(MODIS_ID)

LAND8_SCALE = 30
MODIS_SCALE = 250
SPATIAL_LAND8 = [120, 250]
SPATIAL_MODIS = [250, 500]

# 지정한 날짜와 가장 가까운 이미지를 선택
def closest_image(col, date_str):
    d0 = ee.Date(date_str)
    def add_diff(im):
        t = ee.Date(im.get("system:time_start"))
        return im.set("d", t.difference(d0, "day").abs())
    return ee.Image(col.map(add_diff).sort("d").first())

# 단일 좌표에 대해 NDVI를 추출 (Landsat8 → MODIS 순으로 시도)
def extract_ndvi_for_one(lat, lon, date_str):
    point = ee.Geometry.Point([lon, lat])

    # Landsat8 8일 합성 NDVI
    try:
        img_l8 = closest_image(LAND8_COL, date_str).select("NDVI")
        for buf in SPATIAL_LAND8:
            v = img_l8.reduceRegion(
                reducer=ee.Reducer.median(),
                geometry=point.buffer(buf),
                scale=LAND8_SCALE,
                bestEffort=True
            ).get("NDVI").getInfo()
            if v is not None:
                return float(v)
    except:
        pass

    # MODIS NDVI
    try:
        img_m = closest_image(MODIS_COL, date_str).select("NDVI").multiply(0.0001)
        for buf in SPATIAL_MODIS:
            v = img_m.reduceRegion(
                reducer=ee.Reducer.median(),
                geometry=point.buffer(buf),
                scale=MODIS_SCALE,
                bestEffort=True
            ).get("NDVI").getInfo()
            if v is not None:
                return float(v)
    except:
        pass

    return None

# 병렬 처리를 위한 워커 함수
def worker(row):
    try:
        return extract_ndvi_for_one(row["lat"], row["lon"], row["date"])
    except:
        return None

rows = df.to_dict("records")
results = []

print("Extracting NDVI ...")

# ThreadPoolExecutor로 NDVI 병렬 계산
with ThreadPoolExecutor(max_workers=6) as ex:
    futures = {ex.submit(worker, r): i for i, r in enumerate(rows)}
    for f in tqdm(as_completed(futures), total=len(futures)):
        results.append(f.result())

# 결과 저장
df["NDVI"] = results
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("Saved:", OUTPUT_CSV)
print("Shape:", df.shape)
print("NDVI non-null:", df["NDVI"].notnull().sum())



### 최종 선별 좌표용 DEM/TMI 등 지형 지표 추출

# 입력 좌표(lat, lon)에 대해
# SRTM DEM으로부터 고도, 사면 방향, 상대고, slope_pos, TMI 등을 계산하는 스크립트
# 입력:
#   - CSV (lat, lon 필수, 추가 컬럼은 모두 유지)
# 출력:
#   - input_with_TMI.csv (기존 컬럼 + elev_gee, aspect_deg, rel_h, slope_pos, TMI 등)

# -*- coding: utf-8 -*-
import ee, pandas as pd, np

# Colab 환경에서 파일 업로드를 위한 준비
try:
    from google.colab import files
except ImportError:
    files = None

# GEE 초기화
PROJECT_ID = "YOUR_GEE_PROJECT_ID"

try:
    ee.Initialize(project=PROJECT_ID)
except Exception:
    ee.Authenticate(project=PROJECT_ID)
    ee.Initialize(project=PROJECT_ID)

print("[GEE] initialized")

# 입력 CSV 로드 (colab이면 업로드, 로컬이면 경로 이용)
if files is not None:
    uploaded = files.upload()
    fname = list(uploaded.keys())[0]
    print(f"[INFO] 업로드된 CSV: {fname}")
    df = pd.read_csv(fname)
else:
    df = pd.read_csv(r"C:\path\to\your\input.csv")

# 기본 필수 컬럼 확인
if "lat" not in df.columns or "lon" not in df.columns:
    raise ValueError("입력 CSV에 'lat', 'lon' 컬럼이 있어야 합니다.")

# 병합용 id(pid) 생성
df = df.reset_index(drop=True)
df["pid"] = df.index

print("[INPUT COLUMNS]", df.columns.tolist())
print(df.head())

# pandas DataFrame → GEE FeatureCollection(pid + geometry만 사용)
def df_to_fc_with_pid(df_in, lat_col="lat", lon_col="lon"):
    feats = []
    for _, row in df_in.iterrows():
        lat = float(row[lat_col])
        lon = float(row[lon_col])
        geom = ee.Geometry.Point([lon, lat])
        props = {"pid": int(row["pid"])}
        feats.append(ee.Feature(geom, props))
    return ee.FeatureCollection(feats)

fc_points = df_to_fc_with_pid(df, "lat", "lon")

# DEM, Aspect, 주변 min/max 고도, 상대고(rel_h) 계산용 이미지 준비
dem = ee.Image("USGS/SRTMGL1_003").select("elevation")
aspect = ee.Terrain.aspect(dem).rename("aspect_deg")

kernel = ee.Kernel.circle(radius=500, units="meters")
dem_min = dem.reduceNeighborhood(reducer=ee.Reducer.min(), kernel=kernel).rename("dem_min")
dem_max = dem.reduceNeighborhood(reducer=ee.Reducer.max(), kernel=kernel).rename("dem_max")

# 상대고: (현재 고도 - 주변 최소) / (주변 최대 - 주변 최소)
rel_h = dem.subtract(dem_min).divide(dem_max.subtract(dem_min)).rename("rel_h")

terrain_img = dem.addBands([aspect, dem_min, dem_max, rel_h])

# 좌표별로 DEM 및 지형 지표를 샘플링
sampled = terrain_img.sampleRegions(
    collection=fc_points,
    scale=30,
    geometries=False
)

sampled_dict = sampled.getInfo()

rows = []
for f in sampled_dict["features"]:
    props = f["properties"]
    rows.append(props)

terrain_df = pd.DataFrame(rows)
print("[TERRAIN DF COLUMNS]", terrain_df.columns.tolist())
print(terrain_df.head())

terrain_df = terrain_df.rename(columns={"elevation": "elev_gee"})

# 상대고(rel_h)를 바탕으로 slope_pos 범주 분류
def classify_slope_pos(rel_h_val):
    if pd.isna(rel_h_val):
        return np.nan
    if rel_h_val < 0.2:
        return "산록하부"
    elif rel_h_val < 0.4:
        return "산록상부"
    elif rel_h_val < 0.6:
        return "산복하부"
    elif rel_h_val < 0.8:
        return "산복상부"
    else:
        return "산정하부"

terrain_df["slope_pos"] = terrain_df["rel_h"].apply(classify_slope_pos)

print("[AFTER slope_pos]")
print(terrain_df[["pid", "elev_gee", "aspect_deg", "rel_h", "slope_pos"]].head())

# 원본 df와 pid 기준으로 병합
merged = df.merge(
    terrain_df[["pid", "elev_gee", "aspect_deg", "rel_h", "slope_pos"]],
    on="pid",
    how="left"
)

print("[MERGED COLUMNS]", merged.columns.tolist())
print(merged.head())

# TMI 계수 함수들 (사면 방향, 상대고 위치, 고도 구간)
def tmi_aspect(aspect_deg):
    a = aspect_deg % 360
    if 67.5 <= a < 112.5:
        return 1.5
    elif (292.5 <= a < 360) or (0 <= a < 22.5) or (247.5 <= a < 292.5):
        return 2.5
    elif 112.5 <= a < 202.5:
        return 4
    elif (22.5 <= a < 67.5) or (292.5 <= a < 337.5):
        return 4.5
    elif 202.5 <= a < 247.5:
        return 5
    return np.nan

def tmi_position(pos):
    mapping = {
        "산정하부": 0.5,
        "산복상부": 0.5,
        "산복하부": 1,
        "산록상부": 1.5,
        "산록하부": 5
    }
    return mapping.get(pos, np.nan)

def tmi_elevation(elev):
    if elev >= 876:
        return 1
    elif 628 <= elev < 876:
        return 2
    elif 380 <= elev < 628:
        return 3
    elif 132 <= elev < 380:
        return 4
    elif elev < 132:
        return 5
    return np.nan

def calc_tmi(aspect_deg, pos, elev):
    return tmi_aspect(aspect_deg) + tmi_position(pos) + tmi_elevation(elev)

# TMI 구성 요소 및 최종 TMI 계산
merged["TMI_aspect"] = merged["aspect_deg"].apply(tmi_aspect)
merged["TMI_pos"]    = merged["slope_pos"].apply(tmi_position)
merged["TMI_elev"]   = merged["elev_gee"].apply(tmi_elevation)
merged["TMI"] = merged["TMI_aspect"] + merged["TMI_pos"] + merged["TMI_elev"]

print(merged[["lat","lon","elev_gee","aspect_deg","slope_pos","TMI"]].head())

# 최종 CSV 저장
OUT_PATH = "input_with_TMI.csv"
merged.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print(f"[SAVED] {OUT_PATH}")

if files is not None:
    files.download(OUT_PATH)



### 최종 데이터셋 계절별 FFDRI 분포 확인 및 LSTM 1차 모델

# 최종 LSTM 학습용 데이터셋에서
# FFDRI의 월별 통계와 계절대별 분포를 확인하고
# 봄/가을겨울 계절에 대해 LSTM 1차 모델을 학습하는 코드

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras import layers, models

# LSTM용 시계열 데이터 로드 및 기본 정보 출력
df = pd.read_csv("/content/lstm_dataset.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["pid", "date"]).reset_index(drop=True)

print(df.head())
print(df.columns)
print(df["pid"].nunique(), df["date"].min(), df["date"].max())

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/content/lstm_dataset.csv")
df["date"] = pd.to_datetime(df["date"])
df["Month"] = df["date"].dt.month

# 월별 FFDRI 기초 통계량 계산
monthly_stats = df.groupby("Month")["FFDRI"].agg([
    ("mean_FFDRI", "mean"),
    ("std_FFDRI", "std"),
    ("min_FFDRI", "min"),
    ("max_FFDRI", "max"),
    ("iqr_FFDRI", lambda x: x.quantile(0.75) - x.quantile(0.25)),
])

# Range(최대-최소) 추가
monthly_stats["range_FFDRI"] = monthly_stats["max_FFDRI"] - monthly_stats["min_FFDRI"]

monthly_stats

# 계절대별 FFDRI KDE 비교 플롯
plt.figure(figsize=(10,6))
sns.kdeplot(df[df["Month"].isin([2,3,4])]["FFDRI"], label="봄(2–4월)", linewidth=2)
sns.kdeplot(df[df["Month"].isin([10,11,12])]["FFDRI"], label="10–12월", linewidth=2)
sns.kdeplot(df[df["Month"].isin([5,6,7,8,9])]["FFDRI"], label="5–9월", linewidth=2)
plt.title("계절대별 FFDRI KDE 비교")
plt.legend()
plt.show()

# LSTM 학습을 위한 계절 구분 값 생성
df["Month"] = df["date"].dt.month

def season_category(m):
    if m in [2, 3, 4]:
        return 1  # 봄
    elif m in [10, 11, 12]:
        return 2  # 가을·겨울
    else:
        return 0  # 기타 계절

df["season_cat"] = df["Month"].apply(season_category)
df["season_cat"].value_counts()

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# pid별로 연속된 날짜(seq_len일)를 만족하는 시계열 윈도우 생성
def make_sequences_by_pid(df_sub, feature_cols, target_col, seq_len=14):
    X_list, y_list = [], []

    for pid, g in df_sub.groupby("pid"):
        g = g.sort_values("date")
        values = g[feature_cols + [target_col]].values
        dates = g["date"].values

        if len(g) <= seq_len:
            continue

        for i in range(len(g) - seq_len):
            window_dates = dates[i:i+seq_len]
            diffs = (window_dates[1:] - window_dates[:-1]).astype("timedelta64[D]").astype(int)
            if not np.all(diffs == 1):
                continue

            X_list.append(values[i:i+seq_len, :-1])
            y_list.append(values[i+seq_len, -1])

    if len(X_list) == 0:
        return None, None
    return np.array(X_list), np.array(y_list)

# 가벼운 LSTM 모델 정의
def build_light_lstm(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(32),
        layers.Dense(8, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Adam(1e-3),
        metrics=["mae"]
    )
    return model

# 사용할 피처/타깃/시퀀스 길이 및 학습/테스트 분리 기준일
feature_cols = ["Tmean", "RH", "WSPD", "TP_mm", "sunlight_era5", "NDVI"]
target_col = "FFDRI"
seq_len = 14
split_date = pd.Timestamp("2024-01-01")
season_list = [1, 2]

quick_results = []

# 봄(1), 가을·겨울(2) 계절별로 LSTM 학습 및 평가
for target_season in season_list:
    print("=" * 60)
    print("SEASON", target_season)
    print("=" * 60)

    df_season = df[df["season_cat"] == target_season].copy()
    df_season["date"] = pd.to_datetime(df_season["date"])
    df_season = df_season.sort_values(["pid", "date"])

    train_df = df_season[df_season["date"] < split_date].copy()
    test_df = df_season[df_season["date"] >= split_date].copy()

    if len(train_df) < seq_len + 30 or len(test_df) < seq_len + 10:
        print("skip season", target_season)
        continue

    # 피처 스케일링
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # LSTM 입력 시퀀스 생성
    X_train, y_train = make_sequences_by_pid(train_df, feature_cols, target_col, seq_len)
    X_test, y_test = make_sequences_by_pid(test_df, feature_cols, target_col, seq_len)

    if X_train is None or X_test is None:
        print("sequence insufficient")
        continue

    print("train:", X_train.shape, "test:", X_test.shape)

    model = build_light_lstm((X_train.shape[1], X_train.shape[2]))
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=12,
        batch_size=64,
        callbacks=[es],
        verbose=1
    )

    y_pred = model.predict(X_test).ravel()
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MAE:", round(mae, 3), "R2:", round(r2, 3))

    quick_results.append({
        "season": target_season,
        "train_seq": len(X_train),
        "test_seq": len(X_test),
        "MAE": mae,
        "R2": r2
    })

quick_results_df = pd.DataFrame(quick_results)
quick_results_df


### LSTM 2차모델(DWI 기반 예측)

import pandas as pd
import numpy as np

# 1) 데이터 로드 및 기본 전처리
df = pd.read_csv("/content/lstm_dataset_with_type.csv")
df["date"] = pd.to_datetime(df["date"])

print(df.columns)
print(df.head())

# 2) 월, 계절 범주 생성 (봄=1, 가을·겨울=2, 나머지=0)
df["Month"] = df["date"].dt.month

def season_category(m):
    if m in [2, 3, 4]:
        return 1
    elif m in [10, 11, 12]:
        return 2
    else:
        return 0

df["season_cat"] = df["Month"].apply(season_category)

# 3) LSTM 입력 피처 및 타깃, 보조 변수 정의
#    - feature_cols: 기상 + DWI + 계절
#    - target_col: 다음날 DWI
#    - aux_cols: 이후 FFDRI 계산용 FMI, TMI, day_weight
feature_cols = [
    "Tmean", "RH", "WSPD", "TP_mm", "sunlight_era5",
    "DWI",
    "season_cat"
]

target_col = "DWI"
aux_cols   = ["FMI", "TMI", "day_weight"]

def make_sequences(df_sub, feature_cols, target_col, aux_cols, seq_len=60):
    """
    여러 pid를 포함한 DataFrame을 받아
      X: (N, seq_len, num_features)
      y: (N,)
      aux: (N, len(aux_cols))
    형태로 시퀀스 데이터 생성.
    날짜가 하루씩 연속인 구간만 사용.
    """
    X_list, y_list, aux_list = [], [], []

    for pid, g in df_sub.groupby("pid"):
        g = g.sort_values("date")

        feat_vals   = g[feature_cols].values
        target_vals = g[target_col].values
        aux_vals    = g[aux_cols].values
        dates       = g["date"].values

        if len(g) <= seq_len:
            continue

        for i in range(len(g) - seq_len):
            window_dates = dates[i:i+seq_len+1]
            diffs = (window_dates[1:] - window_dates[:-1]).astype("timedelta64[D]").astype(int)

            # 날짜가 1일 단위로 연속인 구간만 사용
            if not np.all(diffs == 1):
                continue

            X_list.append(feat_vals[i:i+seq_len])
            y_list.append(target_vals[i+seq_len])
            aux_list.append(aux_vals[i+seq_len])

    if len(X_list) == 0:
        return None, None, None

    return np.array(X_list), np.array(y_list), np.array(aux_list)

from sklearn.preprocessing import StandardScaler

# 4) 학습/테스트 기간 분할 및 스케일링
seq_len    = 60
split_date = pd.Timestamp("2023-01-01")

train_df = df[df["date"] < split_date].copy()
test_df  = df[df["date"] >= split_date].copy()

x_scaler = StandardScaler()
train_df[feature_cols] = x_scaler.fit_transform(train_df[feature_cols])
test_df[feature_cols]  = x_scaler.transform(test_df[feature_cols])

# 5) 시퀀스 생성
X_train, y_train, aux_train = make_sequences(train_df, feature_cols, target_col, aux_cols, seq_len)
X_test,  y_test,  aux_test  = make_sequences(test_df,  feature_cols, target_col, aux_cols, seq_len)

print("X_train:", X_train.shape, "X_test:", X_test.shape)

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import mean_absolute_error, r2_score

def build_lstm_model(input_shape):
    """
    2층 LSTM 기반 DWI 예측 모델
      입력: (seq_len, num_features)
      출력: 다음날 DWI 스칼라
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dense(16, activation="relu"),
        layers.Dense(1)
    ])

    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Adam(1e-3),
        metrics=["mae"]
    )
    return model

# 6) LSTM 학습
input_shape = (X_train.shape[1], X_train.shape[2])
model = build_lstm_model(input_shape)

es = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=40,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# 7) DWI 예측 및 성능 평가
y_pred_dwi = model.predict(X_test).ravel()

y_true_dwi      = y_test.copy()
y_pred_dwi_true = y_pred_dwi.copy()

assert len(y_true_dwi) == len(y_pred_dwi_true) == aux_test.shape[0], \
    f"length mismatch: y_true={len(y_true_dwi)}, y_pred={len(y_pred_dwi_true)}, aux={aux_test.shape[0]}"

print("=== DWI 예측 성능 ===")
print("MAE (DWI):", mean_absolute_error(y_true_dwi, y_pred_dwi_true))
print("R2  (DWI):", r2_score(y_true_dwi, y_pred_dwi_true))

# 8) 예측 DWI 기반 FFDRI 예측 성능 평가
FMI_test        = aux_test[:, 0]
TMI_test        = aux_test[:, 1]
day_weight_test = aux_test[:, 2]

FFDRI_true = (y_true_dwi      + FMI_test + TMI_test) * day_weight_test
FFDRI_pred = (y_pred_dwi_true + FMI_test + TMI_test) * day_weight_test

print("\n=== FFDRI 예측 성능 (DWI 예측 기반) ===")
print("MAE (FFDRI):", mean_absolute_error(FFDRI_true, FFDRI_pred))
print("R2  (FFDRI):", r2_score(FFDRI_true, FFDRI_pred))


### 딥러닝 데이터셋 확보 (ERA5 기상변수 추출)

!pip -q install earthengine-api pandas tqdm

import ee, pandas as pd, datetime as dt, time
from google.colab import files
from tqdm import tqdm

# 1) GEE 프로젝트 및 입출력 경로 설정
PROJECT_ID = "YOUR_GEE_PROJECT_ID"
CSV_PATH   = "/content/sites_by_type.csv"
OUT_CSV    = "/content/fri_inputs_by_row.csv"

# ERA5 해상도 기준 추출 설정
SAMPLE_RADIUS_M = 9000
SAMPLE_SCALE_M  = 9000
TILESCALE       = 4

# 날짜 구간 설정
DATE_START = dt.date(2019, 1, 1)
DATE_END   = dt.date(2024, 12, 31)

# 타임존 경계 보정(일 단위)
DATE_SHIFT_DAYS = 0

# 체크포인트 저장 주기(일 단위)
CHECKPOINT_EVERY = 50

# getInfo 재시도 설정
MAX_RETRY   = 6
BACKOFF_BASE= 1.5

# 2) GEE 초기화
try:
    ee.Initialize(project=PROJECT_ID)
except Exception:
    ee.Authenticate(project=PROJECT_ID)
    ee.Initialize(project=PROJECT_ID)
print("[GEE] initialized")
ee.data.setDeadline(120000)

# 3) 좌표 CSV 로드 또는 업로드
try:
    _ = open(CSV_PATH, "r")
except FileNotFoundError:
    print("[UPLOAD] 선택창에서 sites_by_type.csv 업로드")
    uploaded = files.upload()
    if "sites_by_type.csv" not in uploaded:
        name = next(iter(uploaded))
        with open(CSV_PATH, "wb") as f:
            f.write(uploaded[name])
        print(f"[INFO] 업로드 파일을 sites_by_type.csv로 저장: {name} → sites_by_type.csv")
    else:
        print("[INFO] sites_by_type.csv 업로드 완료")

df_sites = pd.read_csv(CSV_PATH, encoding="utf-8-sig", engine="python")
df_sites.columns = [c.strip().lower().replace("\ufeff", "") for c in df_sites.columns]

need = {"lat", "lon"}
if not (need <= set(df_sites.columns)):
    raise ValueError(f"필수 컬럼 누락. 현재 헤더: {list(df_sites.columns)} (필요: {sorted(need)})")

df_sites["lon"] = pd.to_numeric(df_sites["lon"], errors="coerce")
df_sites["lat"] = pd.to_numeric(df_sites["lat"], errors="coerce")
df_sites = df_sites.dropna(subset=["lon", "lat"]).drop_duplicates(subset=["lon", "lat"]).reset_index(drop=True)
df_sites["pid"] = df_sites.index.astype(int)

print(f"[INFO] sites: {len(df_sites)} rows")
print(df_sites.head())

# 4) 날짜 리스트 생성
def daterange(d0, d1):
    cur = d0
    while cur <= d1:
        yield cur
        cur = cur + dt.timedelta(days=1)

unique_dates = [d for d in daterange(DATE_START, DATE_END)]
if DATE_SHIFT_DAYS != 0:
    unique_dates = [d + dt.timedelta(days=DATE_SHIFT_DAYS) for d in unique_dates]
unique_dates_str = [d.isoformat() for d in unique_dates]
print(f"[INFO] date range: {unique_dates_str[0]} ~ {unique_dates_str[-1]}  (days={len(unique_dates_str)})")

# 5) ERA5 ImageCollection 및 일단위 변수 변환 함수
ERA5_LAND   = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
ERA5_GLOBAL = ee.ImageCollection("ECMWF/ERA5/HOURLY")

def _per_hour_to_vars(im):
    # 시간별 온도, 이슬점, 풍속, 강수량을 Tmean, RH, WSPD, TP_mm로 변환
    T  = im.select("temperature_2m").subtract(273.15).rename("Tmean")
    Td = im.select("dewpoint_temperature_2m").subtract(273.15)
    a, b = 17.625, 243.04
    RH = Td.expression(
        "100*exp(a*Td/(b+Td) - a*T/(b+T))",
        {"a": a, "b": b, "Td": Td, "T": T}
    ).rename("RH")
    U = im.select("u_component_of_wind_10m")
    V = im.select("v_component_of_wind_10m")
    WSPD = U.pow(2).add(V.pow(2)).sqrt().rename("WSPD")
    TP = im.select("total_precipitation").multiply(1000).rename("TP_mm")
    return T.addBands([RH, WSPD, TP])

def _daily_from(ic, date_str):
    d0 = ee.Date(date_str)
    d1 = d0.advance(1, "day")
    hourly = ic.filterDate(d0, d1).map(_per_hour_to_vars)
    Tmean = hourly.select("Tmean").mean()
    RH    = hourly.select("RH").mean()
    WSPD  = hourly.select("WSPD").mean()
    TP    = hourly.select("TP_mm").sum()
    return Tmean.addBands([RH, WSPD, TP])

def daily_era5(date_str):
    # land 우선, 누락 시 global로 보완
    land  = _daily_from(ERA5_LAND,   date_str)
    globe = _daily_from(ERA5_GLOBAL, date_str)
    fused = land.unmask(globe)
    return fused.set({"date": date_str})

def safe_getInfo(ee_obj):
    # getInfo 호출 시 지수 백오프 재시도
    for k in range(MAX_RETRY):
        try:
            return ee_obj.getInfo()
        except Exception as e:
            wait = BACKOFF_BASE**k
            print(f"[WARN] getInfo retry {k+1}/{MAX_RETRY} in {wait:.2f}s -> {e}")
            time.sleep(wait)
    return ee_obj.getInfo()

# 6) 이어달리기용 체크포인트 복원
try:
    df_out = pd.read_csv(OUT_CSV)
    done_keys = set(zip(df_out["date"].astype(str), df_out["pid"].astype(int)))
    rows_out = df_out.to_dict("records")
    print(f"[RESUME] existing rows: {len(rows_out)}")
except Exception:
    rows_out = []
    done_keys = set()

# 7) 날짜 × 좌표 루프를 돌며 일 평균 변수 추출
total_dates = len(unique_dates_str)
print(f"[INFO] sampling by date × {len(df_sites)} sites  (dates={total_dates})")

processed_since_cp = 0

for idx, d in enumerate(tqdm(unique_dates_str, desc="[sampling by date]"), 1):
    if all((d, int(pid)) in done_keys for pid in df_sites["pid"].tolist()):
        continue

    fc = ee.FeatureCollection([
        ee.Feature(
            ee.Geometry.Point([float(r["lon"]), float(r["lat"])]).buffer(SAMPLE_RADIUS_M),
            {"pid": int(r["pid"]), "lon": float(r["lon"]), "lat": float(r["lat"])}
        )
        for _, r in df_sites.iterrows()
    ])

    img = daily_era5(d)

    reduced = img.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=SAMPLE_SCALE_M,
        tileScale=TILESCALE
    ).map(lambda f: f.set({"date": d}))

    data  = safe_getInfo(reduced)
    feats = data.get("features", []) if data else []

    for f in feats:
        p = f.get("properties", {})
        key = (str(p.get("date")), int(p.get("pid")))
        if key in done_keys:
            continue
        rows_out.append({
            "date":  p.get("date"),
            "pid":   p.get("pid"),
            "lon":   p.get("lon"),
            "lat":   p.get("lat"),
            "Tmean": p.get("Tmean_mean", p.get("Tmean")),
            "RH":    p.get("RH_mean",    p.get("RH")),
            "WSPD":  p.get("WSPD_mean",  p.get("WSPD")),
            "TP_mm": p.get("TP_mm_mean", p.get("TP_mm")),
        })
        done_keys.add(key)

    processed_since_cp += 1
    if processed_since_cp >= CHECKPOINT_EVERY or idx == total_dates:
        tmp = pd.DataFrame(rows_out).sort_values(["date", "pid"]).reset_index(drop=True)
        tmp.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        print(f"[CP] saved {len(tmp)} rows at {d}  ({idx}/{total_dates})")
        processed_since_cp = 0

# 8) 최종 저장 및 품질 점검
df_out = pd.DataFrame(rows_out).sort_values(["date", "pid"]).reset_index(drop=True)
df_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

num_total   = len(df_out)
num_dates   = df_out["date"].nunique()
num_sites   = df_out["pid"].nunique()
num_all_nan = df_out[["Tmean", "RH", "WSPD", "TP_mm"]].isna().all(axis=1).sum()
num_any_nan = df_out[["Tmean", "RH", "WSPD", "TP_mm"]].isna().any(axis=1).sum()

print(f"[SAVED] {OUT_CSV}  rows={num_total}  (sites={num_sites}, dates={num_dates})")
print(f"[QC] all-NaN rows = {num_all_nan} / any-NaN rows = {num_any_nan}")

bad = df_sites[
    (df_sites["lat"] < 33) | (df_sites["lat"] > 39) |
    (df_sites["lon"] < 124) | (df_sites["lon"] > 132)
]
if len(bad):
    print("[WARN] KR bounds outliers (first 5):")
    print(bad[["pid", "lat", "lon"]].head())

files.download(OUT_CSV)


### 산불 피해규모 기반 FFDRI_new 예측 시도 (RandomForest, ElasticNet)

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

import matplotlib.pyplot as plt

# 1) 산불 피해규모 데이터 로드
DATA_PATH = "/content/wildfire_dataset_GBDG_FFDRI.csv"
df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
df.head()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# 2) 로그 스케일 타깃 생성
target_col = "피해면적_합계"
df["target_log"] = np.log(df[target_col] + 0.01)

# 3) 입력 피처 설정
feature_cols = [
    "tmean", "rh", "eh", "wspd", "tp_mm", "rne",
    "sunlight_era5", "NDVI", "pdwi", "dwi",
    "fmi", "tmi", "ffdri"
]

X = df[feature_cols]
y = df["target_log"]

print("입력변수:", feature_cols)
print("X shape:", X.shape, "y shape:", y.shape)

from sklearn.model_selection import train_test_split

# 4) 학습/검증/테스트 분할
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, shuffle=True
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42, shuffle=True
)

print("Train:", X_train.shape)
print("Valid:", X_valid.shape)
print("Test:", X_test.shape)

# 5) RandomForest 회귀 모델 학습
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

def evaluate(model, X, y_true, name=""):
    """
    피팅된 모델과 X, y_true를 받아
    로그 스케일과 원 스케일에서 성능 평가.
    """
    y_pred_log = model.predict(X)

    mae_log = mean_absolute_error(y_true, y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_true, y_pred_log))
    r2_log   = r2_score(y_true, y_pred_log)
    spear_log, _ = spearmanr(y_true, y_pred_log)

    y_true_raw = np.exp(y_true) - 0.01
    y_pred_raw = np.exp(y_pred_log) - 0.01

    mae = mean_absolute_error(y_true_raw, y_pred_raw)
    rmse = np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))
    r2   = r2_score(y_true_raw, y_pred_raw)
    spear, _ = spearmanr(y_true_raw, y_pred_raw)

    print(f"\n=== {name} 평가 ===")
    print(f"[Log]  MAE={mae_log:.4f}, RMSE={rmse_log:.4f}, R2={r2_log:.4f}, Spearman={spear_log:.4f}")
    print(f"[Raw]  MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, Spearman={spear:.4f}")

    return y_pred_raw, y_pred_log

# 6) RandomForest 성능 평가
train_pred, _ = evaluate(rf, X_train, y_train, "Train")
valid_pred, _ = evaluate(rf, X_valid, y_valid, "Valid")
test_pred,  _ = evaluate(rf, X_test,  y_test,  "Test")

# 7) 전체 데이터에 대한 위험도 인덱스 생성
all_pred_log = rf.predict(X)
all_pred_raw = np.exp(all_pred_log) - 0.01

df["pred_area"] = all_pred_raw

min_val, max_val = df["pred_area"].min(), df["pred_area"].max()
df["danger_index"] = 100 * (df["pred_area"] - min_val) / (max_val - min_val + 1e-9)

print(df[["date", target_col, "pred_area", "danger_index"]].head())

plt.hist(df["danger_index"], bins=30)
plt.title("Danger Index Distribution (0~100)")
plt.xlabel("Index")
plt.ylabel("Count")
plt.show()

# 8) ElasticNet 기반 대안 모델 학습
DATA_PATH = "/content/wildfire_dataset_GBDG_FFDRI.csv"
df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
df.head()

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, shuffle=True
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42, shuffle=True
)

print("Train:", X_train.shape)
print("Valid:", X_valid.shape)
print("Test:", X_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled  = scaler.transform(X_test)
X_all_scaled   = scaler.transform(X)

from sklearn.linear_model import ElasticNetCV

enet = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
    alphas=np.logspace(-4, 2, 50),
    cv=5,
    random_state=42
)

enet.fit(X_train_scaled, y_train)

print("Best alpha:", enet.alpha_)
print("Best l1_ratio:", enet.l1_ratio_)

def eval_split(model, Xs, y_true, name=""):
    """
    ElasticNet 모델의 로그/원 스케일 평가.
    """
    y_pred_log = model.predict(Xs)

    mae_log = mean_absolute_error(y_true, y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_true, y_pred_log))
    r2_log   = r2_score(y_true, y_pred_log)
    spear_log, _ = spearmanr(y_true, y_pred_log)

    y_true_raw = np.exp(y_true) - 0.01
    y_pred_raw = np.exp(y_pred_log) - 0.01

    mae = mean_absolute_error(y_true_raw, y_pred_raw)
    rmse = np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))
    r2   = r2_score(y_true_raw, y_pred_raw)
    spear, _ = spearmanr(y_true_raw, y_pred_raw)

    print(f"\n=== {name} 평가 ===")
    print(f"[Log] MAE={mae_log:.4f}, RMSE={rmse_log:.4f}, R2={r2_log:.4f}, Spearman={spear_log:.4f}")
    print(f"[Raw] MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, Spearman={spear:.4f}")

eval_split(enet, X_train_scaled, y_train, "Train")
eval_split(enet, X_valid_scaled, y_valid, "Valid")
eval_split(enet, X_test_scaled,  y_test,  "Test")

df["pred_log"]  = enet.predict(X_all_scaled)
df["pred_area"] = np.exp(df["pred_log"]) - 0.01

min_v = df["pred_area"].min()
max_v = df["pred_area"].max()
df["danger_index"] = 100 * (df["pred_area"] - min_v) / (max_v - min_v + 1e-9)

df[["date", target_col, "pred_area", "danger_index"]].head()


### FFDRI_new 식 추출 (회귀계수 기반 새 지수 정의)

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

# 1) CSV 업로드 및 로드
CSV_PATH = None

try:
    from google.colab import files
    uploaded = files.upload()
    CSV_PATH = list(uploaded.keys())[0]
    print("업로드한 파일명:", CSV_PATH)
except Exception:
    print("Colab이 아니면 CSV_PATH를 직접 지정하세요.")
    CSV_PATH = "lstm_dataset.csv"

df = pd.read_csv(CSV_PATH)

print("\nCSV 컬럼 목록:")
print(df.columns.tolist())

# 2) FFDRI_new 구성에 사용할 피처와 타깃 지정
feature_cols = [
    "DWI",
    "FMI",
    "TMI",
    "sunlight_era5",
    "NDVI"
]

target_col = "FFDRI"

missing = [c for c in feature_cols + [target_col] if c not in df.columns]
if len(missing) > 0:
    raise ValueError(f"CSV에 없는 컬럼이 있습니다. 다음 이름을 CSV에 맞게 수정해야 합니다: {missing}")

# 3) 결측 제거 및 학습 데이터 구성
df_clean = df.dropna(subset=feature_cols + [target_col]).copy()

X = df_clean[feature_cols].copy()
y = df_clean[target_col].values

print(f"\n사용 데이터 개수: {len(df_clean)} 행")

# 4) 표준화 후 Ridge 회귀 모델 학습
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

reg = Ridge(alpha=1.0, random_state=42)
reg.fit(X_std, y)

y_pred = reg.predict(X_std)
r2   = r2_score(y, y_pred)
mse  = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

print("\n모델 성능 (전체 데이터 기준)")
print(f"R²  : {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# 5) 표준화 공간 계수를 원 단위 계수로 변환
beta0      = reg.intercept_
betas_std  = reg.coef_
means      = scaler.mean_
scales     = scaler.scale_

coeff_orig     = betas_std / scales
intercept_orig = beta0 - np.sum(betas_std * means / scales)

# 6) 최종 FFDRI_new 선형식 출력
print("\n지역특화 FFDRI_new 공식 (원 단위)")
formula = f"FFDRI_new = {intercept_orig:.4f}"
for name, c in zip(feature_cols, coeff_orig):
    sign = " + " if c >= 0 else " - "
    formula += f"{sign}{abs(c):.4f} * {name}"
print(formula)

coef_table = pd.DataFrame({
    "feature": feature_cols,
    "coef_original_space": coeff_orig,
    "coef_standard_space": betas_std
})
print("\n계수 상세표")
print(coef_table)

print("\n간단 해석")
for name, c in zip(feature_cols, coeff_orig):
    print(f"{name}: 계수 {c:.4f}")

# 7) 새 FFDRI_new 값 계산 후 원본 df에 병합
ffdri_new = intercept_orig
for name, c in zip(feature_cols, coeff_orig):
    ffdri_new += c * df_clean[name].values

df_clean["FFDRI_new"] = ffdri_new

df_out = df.copy()
df_out = df_out.merge(
    df_clean[["FFDRI_new"]],
    left_index=True,
    right_index=True,
    how="left"
)

output_name = "ffdri_with_new_index.csv"
df_out.to_csv(output_name, index=False)
print("\n새 지수(FFDRI_new)를 포함한 CSV 저장 완료:", output_name)

try:
    from google.colab import files
    files.download(output_name)
except Exception:
    pass


### LSTM vs GRU 모델 학습 및 계절별 비교 (FFDRI_new 예측)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

print("TensorFlow version:", tf.__version__)

# 1) FFDRI_new가 포함된 CSV 로드
if IN_COLAB:
    print("FFDRI_new가 포함된 lstm_dataset.csv를 업로드하세요.")
    uploaded = files.upload()
    CSV_PATH = list(uploaded.keys())[0]
else:
    CSV_PATH = "lstm_dataset.csv"

df = pd.read_csv(CSV_PATH)
print("원본 데이터 shape:", df.shape)

if "date" not in df.columns:
    raise ValueError("CSV에 'date' 컬럼이 필요합니다.")

df["date"] = pd.to_datetime(df["date"])

# 좌표 ID 컬럼 결정
if "pid" in df.columns:
    id_col = "pid"
elif "site_id" in df.columns:
    id_col = "site_id"
else:
    id_col = "pid"
    df[id_col] = 0

# season 컬럼 없으면 month에서 생성
if "season" not in df.columns:
    df["month"] = df["date"].dt.month

    def season_from_month(m):
        if m in [2, 3, 4]:
            return "spring"
        elif m in [10, 11, 12]:
            return "fall"
        else:
            return "other"

    df["season"] = df["month"].apply(season_from_month)

print("season 분포:")
print(df["season"].value_counts())

# FFDRI_new 타깃 확인
TARGET_COL = "FFDRI_new"
if TARGET_COL not in df.columns:
    raise ValueError("CSV에 'FFDRI_new' 컬럼이 필요합니다.")

# 2) 동적 입력 feature 후보 설정 (타깃은 제외)
candidate_cols = [
    "Tmean", "RH", "WSPD", "TP_mm",
    "sunlight_era5", "DWI", "FMI", "TMI",
    "NDVI", "day_weight", "FFDRI"
]

dynamic_features = []
for c in candidate_cols:
    if c in df.columns and c != TARGET_COL:
        dynamic_features.append(c)

print("사용할 입력 feature:", dynamic_features)
n_features = len(dynamic_features)
if n_features == 0:
    raise ValueError("dynamic_features가 비어 있습니다.")

# 3) 슬라이딩 윈도우 생성 함수
LOOKBACK = 14
HORIZON  = 7

def create_sequences(sub_df, lookback, horizon,
                     feature_cols, id_col, target_col=TARGET_COL):
    """
    id별로 정렬된 시계열에서
      X: (N, lookback, n_features)
      y: (N, horizon)
      meta: pid, base_date, lat, lon
    생성.
    """
    X_list, y_list, meta_list = [], [], []

    for pid, g in sub_df.groupby(id_col):
        g = g.sort_values("date").reset_index(drop=True)

        lat_val = g["lat"].iloc[0] if "lat" in g.columns else np.nan
        lon_val = g["lon"].iloc[0] if "lon" in g.columns else np.nan

        feats  = g[feature_cols].values
        target = g[target_col].values

        if len(g) < lookback + horizon:
            continue

        for i in range(len(g) - lookback - horizon + 1):
            X_list.append(feats[i:i+lookback])
            y_list.append(target[i+lookback:i+lookback+horizon])

            base_idx  = i + lookback - 1
            base_date = g.loc[base_idx, "date"]

            meta_list.append({
                "pid": pid,
                "base_date": base_date,
                "lat": lat_val,
                "lon": lon_val
            })

    if not X_list:
        X = np.empty((0, lookback, len(feature_cols)))
        y = np.empty((0, horizon))
        meta_df = pd.DataFrame(columns=["pid", "base_date", "lat", "lon"])
    else:
        X = np.array(X_list, dtype="float32")
        y = np.array(y_list, dtype="float32")
        meta_df = pd.DataFrame(meta_list)

    return X, y, meta_df

# 4) 계절별 데이터 분할 및 스케일링
def build_splits_for_season(df_all, season_name,
                            feature_cols, id_col):
    """
    특정 season에 대해 train/val/test 분할,
    시퀀스 생성 및 스케일링까지 수행.
    """
    sub = df_all[df_all["season"] == season_name].copy()
    if sub.empty:
        print(f"[{season_name}] season 데이터 없음")
        return None

    sub = sub.sort_values(["date", id_col])

    train_df = sub[sub["date"] < pd.Timestamp("2023-01-01")]
    val_df   = sub[
        (sub["date"] >= pd.Timestamp("2023-01-01")) &
        (sub["date"] <  pd.Timestamp("2024-01-01"))
    ]
    test_df  = sub[sub["date"] >= pd.Timestamp("2024-01-01")]

    X_train, y_train, meta_train = create_sequences(
        train_df, LOOKBACK, HORIZON, feature_cols, id_col
    )
    X_val,   y_val,   meta_val   = create_sequences(
        val_df,   LOOKBACK, HORIZON, feature_cols, id_col
    )
    X_test,  y_test,  meta_test  = create_sequences(
        test_df,  LOOKBACK, HORIZON, feature_cols, id_col
    )

    print(f"[{season_name}] samples: train {X_train.shape[0]} / val {X_val.shape[0]} / test {X_test.shape[0]}")

    scaler = StandardScaler()
    n_feat = len(feature_cols)

    if X_train.shape[0] > 0:
        Xtr_2d = X_train.reshape(-1, n_feat)
        X_train_scaled = scaler.fit_transform(Xtr_2d).reshape(X_train.shape)
    else:
        X_train_scaled = X_train

    def transform_X(X):
        if X.shape[0] == 0:
            return X
        X2d = X.reshape(-1, n_feat)
        return scaler.transform(X2d).reshape(X.shape)

    X_val_scaled  = transform_X(X_val)
    X_test_scaled = transform_X(X_test)

    return {
        "X_train": X_train_scaled, "y_train": y_train, "meta_train": meta_train,
        "X_val":   X_val_scaled,   "y_val":   y_val,   "meta_val":   meta_val,
        "X_test":  X_test_scaled,  "y_test":  y_test,  "meta_test":  meta_test,
        "scaler":  scaler
    }

# 5) LSTM/GRU 모델 정의
def build_lstm_model(timesteps, n_features, horizon):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(timesteps, n_features)),
        LSTM(32),
        Dropout(0.2),
        Dense(horizon)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
        metrics=["mae"]
    )
    return model

def build_gru_model(timesteps, n_features, horizon):
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(timesteps, n_features)),
        GRU(32),
        Dropout(0.2),
        Dense(horizon)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
        metrics=["mae"]
    )
    return model

# 6) 학습 이력 시각화
def plot_history(history, season, model_name):
    h = history.history
    plt.figure(figsize=(7, 4))
    plt.plot(h["loss"], label="train_loss")
    if "val_loss" in h:
        plt.plot(h["val_loss"], label="val_loss")
    plt.title(f"{season} - {model_name} Training History")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 7) 테스트셋 성능 평가
def evaluate_on_test(model, X_test, y_test, season, model_name):
    """
    t+1, t+1~t+7 구간에서 RMSE, MAE, 상관계수 계산.
    """
    if X_test.shape[0] == 0:
        print(f"[{season}][{model_name}] 테스트셋 없음")
        return {}

    y_pred = model.predict(X_test, verbose=0)
    assert y_pred.shape == y_test.shape

    y_true_1 = y_test[:, 0]
    y_pred_1 = y_pred[:, 0]
    rmse_1 = np.sqrt(mean_squared_error(y_true_1, y_pred_1))
    mae_1  = mean_absolute_error(y_true_1, y_pred_1)
    corr_1 = np.corrcoef(y_true_1, y_pred_1)[0, 1]

    rmse_all = np.sqrt(mean_squared_error(y_test.reshape(-1), y_pred.reshape(-1)))
    mae_all  = mean_absolute_error(y_test.reshape(-1), y_pred.reshape(-1))

    print(f"\n[{season}][{model_name}] 테스트셋 성능")
    print("  (t+1)  RMSE:", rmse_1, " MAE:", mae_1, " Corr:", corr_1)
    print("  (1~7) RMSE:", rmse_all, " MAE:", mae_all)

    return {
        "rmse_1": rmse_1, "mae_1": mae_1, "corr_1": corr_1,
        "rmse_all": rmse_all, "mae_all": mae_all
    }

# 8) 계절별 LSTM vs GRU 비교 루프
EPOCHS = 100
BATCH  = 64

summary_rows = []

for season_name in ["spring", "fall"]:
    print(f"\nSeason: {season_name}")

    splits = build_splits_for_season(df, season_name, dynamic_features, id_col)
    if splits is None:
        continue

    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_val   = splits["X_val"]
    y_val   = splits["y_val"]
    X_test  = splits["X_test"]
    y_test  = splits["y_test"]

    if X_train.shape[0] == 0:
        print(f"[{season_name}] 학습 가능한 시퀀스 없음")
        continue

    for model_name, builder in [("LSTM", build_lstm_model), ("GRU", build_gru_model)]:
        print(f"\n{season_name} / {model_name} 학습")

        model = builder(LOOKBACK, n_features, HORIZON)
        model.summary()

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            )
        ]

        if X_val.shape[0] > 0:
            history = model.fit(
                X_train, y_train,
                epochs=EPOCHS,
                batch_size=BATCH,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        else:
            history = model.fit(
                X_train, y_train,
                epochs=EPOCHS,
                batch_size=BATCH,
                callbacks=callbacks,
                verbose=1
            )

        plot_history(history, season_name, model_name)

        h = history.history
        train_loss = h["loss"][-1]
        train_mae  = h["mae"][-1]
        if "val_loss" in h:
            val_loss = h["val_loss"][-1]
            val_mae  = h["val_mae"][-1]
        else:
            val_loss = np.nan
            val_mae  = np.nan

        metrics_test = evaluate_on_test(model, X_test, y_test, season_name, model_name)

        summary_rows.append({
            "season": season_name,
            "model": model_name,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mae": train_mae,
            "val_mae": val_mae,
            **metrics_test
        })

# 9) LSTM vs GRU 요약 테이블
if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    print("\nLSTM vs GRU 학습/검정 비교 요약")
    print(summary_df)

    if IN_COLAB:
        summary_df.to_csv("lstm_gru_ffdri_new_summary.csv",
                          index=False, encoding="utf-8-sig")
        files.download("lstm_gru_ffdri_new_summary.csv")
else:
    print("요약할 결과가 없습니다.")


### 2025년 4월 28일 대구 함지산 대형산불일자 모델 테스트 (GEE 기반 GRU 예측)

import io
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    files = None

import tensorflow as tf
from tensorflow.keras.models import load_model

import ee

# 0) GEE 초기화
PROJECT_ID = "YOUR_GEE_PROJECT_ID"

try:
    ee.Initialize(project=PROJECT_ID)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)

print("[GEE] initialized")

# 1) 사용자 입력 (예측 날짜, 좌표, 시즌 등)
FORECAST_DATE = "2025-04-28"
FORECAST_DATE = datetime.strptime(FORECAST_DATE, "%Y-%m-%d").date()

NEW_LAT = 35.9181780434288
NEW_LON = 128.566700685493
NEIGHBOR_KM = 5

SEASON   = "spring"
TIMEZONE = "Asia/Seoul"

LOOKBACK = 14
HORIZON  = 7

OUTPUT_CSV = f"ffdri_new_forecast_NEWPOINT_{SEASON}_{FORECAST_DATE}_7d.csv"

print("입력 좌표:", NEW_LAT, NEW_LON)
print("선택 시즌:", SEASON)
print("예측 기준일:", FORECAST_DATE)

# 2) lstm_dataset.csv 로드
if IN_COLAB:
    print("lstm_dataset.csv 업로드")
    uploaded = files.upload()
    csv_name = list(uploaded.keys())[0]
else:
    csv_name = "lstm_dataset.csv"

df = pd.read_csv(csv_name, parse_dates=["date"])
df = df.sort_values(["pid", "date"]).reset_index(drop=True)

# season 컬럼이 없으면 월 기반으로 생성
if "season" not in df.columns:
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].apply(
        lambda m: "spring" if m in [2, 3, 4]
        else ("fall" if m in [10, 11, 12] else "other")
    )

df = df[df["season"] == SEASON].copy()
if df.empty:
    raise ValueError(f"{SEASON} 시즌 데이터가 없습니다.")

df["doy"] = df["date"].dt.dayofyear

# 3) Feature / Target 정의
TARGET = "FFDRI_new"
if TARGET not in df.columns:
    raise ValueError("FFDRI_new 컬럼이 필요합니다.")

candidate_features = [
    "Tmean", "RH", "WSPD", "TP_mm",
    "sunlight_era5", "DWI", "FMI", "TMI",
    "NDVI", "day_weight", "FFDRI"
]

FEATURES = [c for c in candidate_features if c in df.columns and c != TARGET]
n_features = len(FEATURES)

if n_features == 0:
    raise ValueError("사용 가능한 Feature가 없습니다. candidate_features를 확인하세요.")

print("사용 Features:", FEATURES)
print("n_features:", n_features)

# 4) GRU 학습과 동일한 방식으로 스케일러 구성
def make_sequences(sub_df):
    X_list, y_list = [], []

    for pid, g in sub_df.groupby("pid"):
        g = g.sort_values("date").reset_index(drop=True)

        X_raw = g[FEATURES].values
        y_raw = g[TARGET].values

        if len(g) < LOOKBACK + HORIZON:
            continue

        for i in range(LOOKBACK, len(g) - HORIZON + 1):
            X_list.append(X_raw[i-LOOKBACK:i])
            y_list.append(y_raw[i:i+HORIZON])

    if len(X_list) == 0:
        return None, None

    return np.array(X_list, dtype="float32"), np.array(y_list, dtype="float32")

X_all, y_all = make_sequences(df)
if X_all is None:
    raise ValueError("데이터가 부족해 시퀀스를 만들 수 없습니다.")

N = X_all.shape[0]
idx = int(N * 0.8)
X_train = X_all[:idx]

scaler = StandardScaler()
X_train_2d = X_train.reshape(-1, n_features)
scaler.fit(X_train_2d)

def transform_X(z):
    z2 = z.reshape(-1, n_features)
    z2 = scaler.transform(z2)
    return z2.reshape(z.shape)

# 5) 주변 학습 좌표 탐색 (NEIGHBOR_KM 이내 pid 선택)
pid_locs = df.groupby("pid")[["lat", "lon"]].first().reset_index()

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

pid_locs["dist_km"] = pid_locs.apply(
    lambda r: haversine(NEW_LAT, NEW_LON, r["lat"], r["lon"]),
    axis=1
)

near = pid_locs[pid_locs["dist_km"] <= NEIGHBOR_KM]

if near.empty:
    print("주변 좌표 없음 → 전체 pid 사용")
    near_pids = pid_locs["pid"].tolist()
else:
    print(len(near), "개 좌표 사용")
    near_pids = near["pid"].tolist()

# 6) DOY 기반 기후 클라이모(평년값) 계산 (pid, doy 기준)
clim_features = df.groupby(["pid", "doy"])[FEATURES].mean()

def get_climo_vector(date_obj):
    doy = date_obj.timetuple().tm_yday
    rows = []

    for pid in near_pids:
        key = (pid, doy)
        if key in clim_features.index:
            rows.append(clim_features.loc[key].values)

    if rows:
        return np.mean(rows, axis=0).astype("float32")
    return clim_features.mean().values.astype("float32")

# 7) GEE ERA5-Land에서 하루 평균 기상요소 가져오기
def fetch_weather_by_date_gee(lat, lon, date_obj):
    start = ee.Date(date_obj.strftime("%Y-%m-%d"))
    end   = start.advance(1, "day")

    col = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").filterDate(start, end)

    img_mean = col.select([
        "temperature_2m",
        "dewpoint_temperature_2m",
        "u_component_of_wind_10m",
        "v_component_of_wind_10m"
    ]).mean()

    img_sum_tp = col.select("total_precipitation").sum().rename("total_precipitation_sum")

    img_agg = img_mean.addBands(img_sum_tp)

    pt = ee.Geometry.Point(lon, lat)
    vals = img_agg.sample(pt, scale=9000).first().getInfo()["properties"]

    T_K  = vals["temperature_2m"]
    Td_K = vals["dewpoint_temperature_2m"]
    u10  = vals["u_component_of_wind_10m"]
    v10  = vals["v_component_of_wind_10m"]
    tp_m = vals["total_precipitation_sum"]

    T_C  = T_K  - 273.15
    Td_C = Td_K - 273.15

    wspd = float((u10**2 + v10**2) ** 0.5)
    tp_mm = float(tp_m * 1000.0)

    import math
    def esat(T):
        return 6.112 * math.exp((17.67*T)/(T+243.5))

    e  = esat(Td_C)
    es = esat(T_C)
    rh = float(100.0 * e / es) if es > 0 else np.nan

    result = {
        "T_mean":   float(T_C),
        "T_max":    float(T_C),
        "RH_mean":  rh,
        "WSPD_mean": wspd,
        "TP_sum":   tp_mm,
    }
    return result

# 8) DWI 계산 함수 (계절별 식)
def pre_spring(T, RH, W):
    return 1/(1+np.exp(2.706 + 0.088*T - 0.055*RH - 0.023*RH - 0.104*W))

def pre_fall(T, RH, W):
    return 1/(1+np.exp(1.099 + 0.117*T - 0.069*RH - 0.182*W))

def compute_DWI(weather):
    if SEASON == "spring":
        pre = pre_spring(weather["T_max"], weather["RH_mean"], weather["WSPD_mean"])
    else:
        pre = pre_fall(weather["T_max"], weather["RH_mean"], weather["WSPD_mean"])

    tp = weather["TP_sum"]
    if tp >= 10:
        rne = 0.3
    elif tp >= 1:
        rne = 0.7
    else:
        rne = 1.0
    return pre * rne

# 9) FFDRI_new 계산식 (회귀 기반 공식)
def compute_FFDRI_new(dwi, fmi, tmi, sunlight, ndvi):
    return (
        5.3012
        + 6.6232*dwi
        + 0.7657*fmi
        + 1.0994*tmi
        + 0.0906*sunlight
        - 10.6796*ndvi
    )

# 10) LOOKBACK 시퀀스 구성 (FORECAST_DATE 기준)
feat_index = {name: i for i, name in enumerate(FEATURES)}
seq_rows = []

for offset in range(LOOKBACK-1, 0, -1):
    d = FORECAST_DATE - timedelta(days=offset)
    vec = get_climo_vector(d)
    seq_rows.append(vec)

today_vec = get_climo_vector(FORECAST_DATE)

weather_input = fetch_weather_by_date_gee(NEW_LAT, NEW_LON, FORECAST_DATE)
dwi_today = compute_DWI(weather_input)

if "Tmean" in feat_index:
    today_vec[feat_index["Tmean"]] = weather_input["T_mean"]
if "RH" in feat_index:
    today_vec[feat_index["RH"]] = weather_input["RH_mean"]
if "WSPD" in feat_index:
    today_vec[feat_index["WSPD"]] = weather_input["WSPD_mean"]
if "TP_mm" in feat_index:
    today_vec[feat_index["TP_mm"]] = weather_input["TP_sum"]
if "DWI" in feat_index:
    today_vec[feat_index["DWI"]] = dwi_today

seq_rows.append(today_vec)

seq_feat = np.stack(seq_rows, axis=0)
seq_feat = seq_feat.reshape(1, LOOKBACK, n_features)
seq_feat_s = transform_X(seq_feat)

# 11) GRU 모델 로드 및 7일 예측 수행
if IN_COLAB:
    print("gru 모델 업로드 (예: gru_spring.h5)")
    uploaded_model = files.upload()
    model_name = list(uploaded_model.keys())[0]
else:
    model_name = f"gru_{SEASON}.h5"

model = load_model(model_name, compile=False)

print("model.input_shape:", model.input_shape)
print("입력 seq_feat_s.shape:", seq_feat_s.shape)

y_pred = model.predict(seq_feat_s)[0]

# 12) 오늘 FFDRI_new 계산 (보고용)
FMI_  = today_vec[feat_index["FMI"]]           if "FMI" in feat_index else np.nan
TMI_  = today_vec[feat_index["TMI"]]           if "TMI" in feat_index else np.nan
NDVI_ = today_vec[feat_index["NDVI"]]          if "NDVI" in feat_index else np.nan
SUN_  = today_vec[feat_index["sunlight_era5"]] if "sunlight_era5" in feat_index else np.nan

ff_today = compute_FFDRI_new(dwi_today, FMI_, TMI_, SUN_, NDVI_)

# 13) 위험등급 구간 정의 (분위수 기반)
q25 = df["FFDRI_new"].quantile(0.25)
q50 = df["FFDRI_new"].quantile(0.50)
q75 = df["FFDRI_new"].quantile(0.75)
q90 = df["FFDRI_new"].quantile(0.90)

print("FFDRI_new quantiles (season =", SEASON, ")")
print("Q25:", q25, " Q50:", q50, " Q75:", q75, " Q90:", q90)

def risk(v):
    if v < q25:
        return "low"
    elif v < q50:
        return "moderate"
    elif v < q75:
        return "high"
    elif v < q90:
        return "very_high"
    else:
        return "extreme"

# 14) 예측 결과 출력
print("\n예측 결과 (GEE 기반)")
print("예측 기준일:", FORECAST_DATE)
print("입력 좌표:", NEW_LAT, NEW_LON)
print("GEE(ERA5-Land) 입력:", weather_input)
print("DWI:", dwi_today)
print("오늘 FFDRI_new 계산값:", ff_today)

for i in range(HORIZON):
    print(f"D+{i+1}: {y_pred[i]:.3f} | 위험등급: {risk(y_pred[i]+5.5)}")

# 15) CSV 저장
out = pd.DataFrame({
    "base_date": [FORECAST_DATE],
    "lat": [NEW_LAT],
    "lon": [NEW_LON],
    "FFDRI_new_today": [ff_today],
})

for i in range(HORIZON):
    out[f"FFDRI_new_model_D{i+1}"] = [y_pred[i]]
    out[f"risk_D{i+1}"] = [risk(y_pred[i])]

out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

if IN_COLAB:
    files.download(OUTPUT_CSV)



### 계절 분리 적용 테스트 (Open-Meteo + GRU)

import io
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    files = None

import tensorflow as tf
from tensorflow.keras.models import load_model

# 0) 사용자 입력
NEW_LAT = 35.9181780434288
NEW_LON = 128.566700685493
NEIGHBOR_KM = 5

SEASON   = "spring"
TIMEZONE = "Asia/Seoul"

LOOKBACK = 14
HORIZON  = 7

today = datetime.now().date()
OUTPUT_CSV = f"ffdri_new_forecast_NEWPOINT_{SEASON}_{today}_7d.csv"

print("입력 좌표:", NEW_LAT, NEW_LON)
print("선택 시즌:", SEASON)

# 1) lstm_dataset.csv 로드
if IN_COLAB:
    print("lstm_dataset.csv 업로드")
    uploaded = files.upload()
    csv_name = list(uploaded.keys())[0]
else:
    csv_name = "lstm_dataset.csv"

df = pd.read_csv(csv_name, parse_dates=["date"])
df = df.sort_values(["pid", "date"]).reset_index(drop=True)

if "season" not in df.columns:
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].apply(
        lambda m: "spring" if m in [2, 3, 4]
        else ("fall" if m in [10, 11, 12] else "other")
    )

df = df[df["season"] == SEASON].copy()
if df.empty:
    raise ValueError(f"{SEASON} 시즌 데이터가 없습니다.")

df["doy"] = df["date"].dt.dayofyear

# 2) Feature / Target 정의
TARGET = "FFDRI_new"
if TARGET not in df.columns:
    raise ValueError("FFDRI_new 컬럼이 필요합니다. (학습 때와 동일 CSV)")

candidate_features = [
    "Tmean", "RH", "WSPD", "TP_mm",
    "sunlight_era5", "DWI", "FMI", "TMI",
    "NDVI", "day_weight", "FFDRI"
]

FEATURES = [c for c in candidate_features if c in df.columns and c != TARGET]
n_features = len(FEATURES)

if n_features == 0:
    raise ValueError("사용 가능한 Feature가 없습니다. candidate_features를 확인하세요.")

print("사용 Feature:", FEATURES)
print("Feature 개수:", n_features)

# 3) GRU 학습 시 사용된 시퀀스/스케일러 복원
def make_sequences(sub_df):
    X_list, y_list = [], []

    for pid, g in sub_df.groupby("pid"):
        g = g.sort_values("date").reset_index(drop=True)

        X_raw = g[FEATURES].values
        y_raw = g[TARGET].values

        if len(g) < LOOKBACK + HORIZON:
            continue

        for i in range(LOOKBACK, len(g) - HORIZON + 1):
            X_list.append(X_raw[i-LOOKBACK:i])
            y_list.append(y_raw[i:i+HORIZON])

    if len(X_list) == 0:
        return None, None

    return np.array(X_list, dtype="float32"), np.array(y_list, dtype="float32")

X_all, y_all = make_sequences(df)
if X_all is None:
    raise ValueError("시퀀스를 만들 수 없습니다. (데이터 길이 부족)")

N = X_all.shape[0]
idx = int(N * 0.8)
X_train = X_all[:idx]

scaler = StandardScaler()
X_train_2d = X_train.reshape(-1, n_features)
scaler.fit(X_train_2d)

def transform_X(z):
    z2 = z.reshape(-1, n_features)
    z2 = scaler.transform(z2)
    return z2.reshape(z.shape)

# 4) 주변 학습 좌표 탐색
pid_locs = df.groupby("pid")[["lat", "lon"]].first().reset_index()

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat/2)**2
        + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))
        * np.sin(dlon/2)**2
    )
    return 2 * R * np.arcsin(np.sqrt(a))

pid_locs["dist_km"] = pid_locs.apply(
    lambda row: haversine(NEW_LAT, NEW_LON, row["lat"], row["lon"]),
    axis=1
)

near = pid_locs[pid_locs["dist_km"] <= NEIGHBOR_KM]

if near.empty:
    print(f"주변 {NEIGHBOR_KM}km 내 학습좌표 없음 → 전체 pid 사용")
    near_pids = pid_locs["pid"].tolist()
else:
    print(f"{len(near)}개 학습좌표가 {NEIGHBOR_KM}km 내 존재")
    near_pids = near["pid"].tolist()

# 5) DOY별 Feature 클라이모
clim_features = df.groupby(["pid", "doy"])[FEATURES].mean()

def get_climo_row_for_date(d):
    doy = d.timetuple().tm_yday
    rows = []

    for pid in near_pids:
        key = (pid, doy)
        if key in clim_features.index:
            rows.append(clim_features.loc[key].values)

    if rows:
        return np.mean(rows, axis=0).astype("float32")
    return clim_features.mean().values.astype("float32")

# 6) Open-Meteo에서 오늘 기상 정보 가져오기
def fetch_today_weather(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "forecast_days": 1,
        "timezone": TIMEZONE,
    }
    res = requests.get(url, params=params)
    res.raise_for_status()
    data = res.json()

    h = pd.DataFrame(data["hourly"])
    h["time"] = pd.to_datetime(h["time"])
    h["date"] = h["time"].dt.date
    today_block = h[h["date"] == today]

    if today_block.empty:
        raise RuntimeError("Open-Meteo 결과에서 오늘 날짜 데이터가 없습니다.")

    return {
        "T_max":   float(today_block["temperature_2m"].max()),
        "RH_mean": float(today_block["relative_humidity_2m"].mean()),
        "WSPD_mean": float(today_block["wind_speed_10m"].mean()),
        "TP_sum":  float(today_block["precipitation"].sum())
    }

# 7) 오늘 DWI 계산 (계절별 식)
def pre_spring(T, RH, W):
    return 1/(1+np.exp(2.706 + 0.088*T - 0.055*RH - 0.023*RH - 0.104*W))

def pre_fall(T, RH, W):
    return 1/(1+np.exp(1.099 + 0.117*T - 0.069*RH - 0.182*W))

def compute_DWI(weather):
    if SEASON == "spring":
        pre = pre_spring(weather["T_max"], weather["RH_mean"], weather["WSPD_mean"])
    else:
        pre = pre_fall(weather["T_max"], weather["RH_mean"], weather["WSPD_mean"])

    tp = weather["TP_sum"]
    if tp >= 10:
        rne = 0.3
    elif tp >= 1:
        rne = 0.7
    else:
        rne = 1.0
    return pre * rne

weather_today = fetch_today_weather(NEW_LAT, NEW_LON)
dwi_today = compute_DWI(weather_today)

# 8) FFDRI_new 계산 함수
def compute_FFDRI_new(dwi, fmi, tmi, sunlight, ndvi):
    return (
        5.3012
        + 6.6232*dwi
        + 0.7657*fmi
        + 1.0994*tmi
        + 0.0906*sunlight
        - 10.6796*ndvi
    )

# 9) 새 좌표의 14일치 Feature 시퀀스 만들기
feat_index = {name: i for i, name in enumerate(FEATURES)}
seq_rows = []

for offset in range(LOOKBACK-1, 0, -1):
    d = today - timedelta(days=offset)
    row = get_climo_row_for_date(d)
    seq_rows.append(row)

today_feat = get_climo_row_for_date(today)

if "Tmean" in feat_index:
    today_feat[feat_index["Tmean"]] = weather_today["T_max"]
if "RH" in feat_index:
    today_feat[feat_index["RH"]] = weather_today["RH_mean"]
if "WSPD" in feat_index:
    today_feat[feat_index["WSPD"]] = weather_today["WSPD_mean"]
if "TP_mm" in feat_index:
    today_feat[feat_index["TP_mm"]] = weather_today["TP_sum"]
if "DWI" in feat_index:
    today_feat[feat_index["DWI"]] = dwi_today

seq_rows.append(today_feat)

seq_feat = np.stack(seq_rows, axis=0)
seq_feat = seq_feat.reshape(1, LOOKBACK, n_features)

# 10) 입력 Feature 표준화
seq_feat_s = transform_X(seq_feat)

# 11) GRU 모델 로드 및 예측
if IN_COLAB:
    print("GRU 모델(.h5) 업로드 (예: gru_fall.h5)")
    uploaded_model = files.upload()
    model_name = list(uploaded_model.keys())[0]
else:
    model_name = f"gru_{SEASON}.h5"

model = load_model(model_name, compile=False)
print("model.input_shape:", model.input_shape)
print("입력 seq_feat_s.shape:", seq_feat_s.shape)

y_pred = model.predict(seq_feat_s)[0]

# 12) 오늘 FFDRI_new 계산 (보고용)
FMI_   = today_feat[feat_index["FMI"]]   if "FMI" in feat_index else np.nan
TMI_   = today_feat[feat_index["TMI"]]   if "TMI" in feat_index else np.nan
NDVI_  = today_feat[feat_index["NDVI"]]  if "NDVI" in feat_index else np.nan
SUN_   = today_feat[feat_index["sunlight_era5"]] if "sunlight_era5" in feat_index else np.nan

ff_today = compute_FFDRI_new(dwi_today, FMI_, TMI_, SUN_, NDVI_)

# 13) 위험등급 구간 정의 (분위수 기반)
q25 = df["FFDRI_new"].quantile(0.25)
q50 = df["FFDRI_new"].quantile(0.50)
q75 = df["FFDRI_new"].quantile(0.75)
q90 = df["FFDRI_new"].quantile(0.90)

print("FFDRI_new quantiles (season =", SEASON, ")")
print("Q25:", q25, " Q50:", q50, " Q75:", q75, " Q90:", q90)

def risk(v):
    if v < q25:
        return "low"
    elif v < q50:
        return "moderate"
    elif v < q75:
        return "high"
    elif v < q90:
        return "very_high"
    else:
        return "extreme"

# 14) 결과 출력
print("\n예측 결과 (새 좌표)")
print("입력 좌표 :", NEW_LAT, NEW_LON)
print("오늘 날짜 :", today)
print("오늘 FFDRI_new (계산값) :", ff_today)
print("Open-Meteo 오늘 T/RH/WSPD/TP:", weather_today)
print("오늘 DWI :", dwi_today)

for h in range(HORIZON):
    print(f"D+{h+1} = {y_pred[h]:.3f} | 위험등급: {risk(y_pred[h])}")

# 15) CSV 저장
out = pd.DataFrame({
    "base_date": [today],
    "lat": [NEW_LAT],
    "lon": [NEW_LON],
    "FFDRI_new_today": [ff_today],
})

for h in range(HORIZON):
    out[f"FFDRI_new_model_{h+1}d"] = [y_pred[h]]
    out[f"risk_{h+1}d"] = [risk(y_pred[h])]

out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

if IN_COLAB:
    files.download(OUTPUT_CSV)



### 경북·대구 대표산 위험도 예측 및 지도 시각화

import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import matplotlib.pyplot as plt

try:
    from google.colab import files
except ImportError:
    files = None

import tensorflow as tf
from tensorflow.keras.models import load_model

import geopandas as gpd
from shapely.geometry import Point
from matplotlib.colors import ListedColormap, BoundaryNorm

print("TF version:", tf.__version__)

# 0) 사용자 설정
SEASON   = "fall"
SEQ_LEN  = 30
HORIZON  = 7
TARGET_H = 3
TIMEZONE = "Asia/Seoul"
NEIGHBOR_KM = 5

today = datetime.now().date()

SITES_INFO = [
    {"name": "Hamji-san",       "lat": 35.9181780434288, "lon": 128.566700685493},
    {"name": "Palgong-san",     "lat": 36.0225318418839, "lon": 128.736206404928},
    {"name": "Geumo-san",       "lat": 36.0923500736404, "lon": 128.301361851721},
    {"name": "Juwang-san",      "lat": 36.3938590686737, "lon": 129.142072525794},
    {"name": "Cheongnyang-san", "lat": 36.8013774501963, "lon": 128.939160702182},
]

def get_season_from_date(d):
    m = d.month
    return "spring" if 1 <= m <= 7 else "fall"

# 1) lstm_dataset.csv 업로드
print("\ncsv 파일을 업로드하세요.")
uploaded = files.upload()
csv_name = list(uploaded.keys())[0]
df = pd.read_csv(csv_name, parse_dates=["date"])
df = df.sort_values(["pid", "date"])

if "season" in df.columns and SEASON in ("spring", "fall"):
    df = df[df["season"] == SEASON]

df["doy"] = df["date"].dt.dayofyear

# 2) FFDRI_new 계산
sun_cols = [c for c in df.columns if "sun" in c.lower()]
ndvi_cols = [c for c in df.columns if "ndvi" in c.lower()]

SUN_COL = sun_cols[0]
NDVI_COL = ndvi_cols[0]

def compute_FFDRI_new(dwi, fmi, tmi, sun, ndvi):
    return (
        5.3012
        + 6.6232*dwi
        + 0.7657*fmi
        + 1.0994*tmi
        + 0.0906*sun
        - 10.6796*ndvi
    )

df["FFDRI_new"] = compute_FFDRI_new(
    df["DWI"], df["FMI"], df["TMI"], df[SUN_COL], df[NDVI_COL]
)

ffdri_mean = df["FFDRI_new"].mean()
ffdri_std  = df["FFDRI_new"].std()

def standardize(x):
    return (x - ffdri_mean) / ffdri_std

def inv_standardize(x):
    return x * ffdri_std + ffdri_mean

# 3) GRU 모델 업로드
print("\nGRU 모델(.h5)을 업로드하세요.")
uploaded_model = files.upload()
model_name = list(uploaded_model.keys())[0]
model = load_model(model_name, compile=False)

# 4) Open-Meteo 기반 오늘 기상 → DWI 계산
def fetch_today_weather(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "forecast_days": 1,
        "timezone": TIMEZONE,
    }
    data = requests.get(url, params=params).json()
    h = pd.DataFrame(data["hourly"])
    h["time"] = pd.to_datetime(h["time"])
    h["date"] = h["time"].dt.date
    block = h[h["date"] == today]
    if block.empty:
        block = h
    return {
        "T_max": block["temperature_2m"].max(),
        "RH_mean": block["relative_humidity_2m"].mean(),
        "WSPD_mean": block["wind_speed_10m"].mean(),
        "TP_sum": block["precipitation"].sum(),
    }

def pre_spring(T, RH, EH):
    return 1/(1+np.exp(2.706 + 0.088*T - 0.055*RH - 0.023*EH))

def pre_fall(T, RH, EH):
    return 1/(1+np.exp(1.099 + 0.117*T - 0.069*RH - 0.182*EH))

def compute_DWI_from_weather(w, season):
    if season == "auto":
        season = get_season_from_date(today)
    if season == "spring":
        pre = pre_spring(w["T_max"], w["RH_mean"], w["WSPD_mean"])
    else:
        pre = pre_fall(w["T_max"], w["RH_mean"], w["WSPD_mean"])
    tp = w["TP_sum"]
    if tp >= 10:
        rne = 0.3
    elif tp >= 1:
        rne = 0.7
    else:
        rne = 1.0
    return pre * rne

# 5) 5개 산에 대해 7일 예측 수행
pid_locs = df.groupby("pid")[["lat", "lon"]].first().reset_index()
clim_ffdri = df.groupby(["pid", "doy"])["FFDRI_new"].mean()

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

q25 = df["FFDRI_new"].quantile(0.25)
q50 = df["FFDRI_new"].quantile(0.50)
q75 = df["FFDRI_new"].quantile(0.75)
q90 = df["FFDRI_new"].quantile(0.90)

def risk_level(v):
    if v < q25:
        return "low"
    elif v < q50:
        return "moderate"
    elif v < q75:
        return "high"
    elif v < q90:
        return "very_high"
    else:
        return "extreme"

sites_records = []

for site in SITES_INFO:
    lat, lon = site["lat"], site["lon"]

    pid_locs["dist"] = pid_locs.apply(
        lambda r: haversine(lat, lon, r["lat"], r["lon"]), axis=1
    )
    near = pid_locs[pid_locs["dist"] <= NEIGHBOR_KM]
    if near.empty:
        near_pids = pid_locs["pid"].tolist()
    else:
        near_pids = near["pid"].tolist()

    w = fetch_today_weather(lat, lon)
    dwi_today = compute_DWI_from_weather(w, SEASON)

    doy_today = today.timetuple().tm_yday
    sub = df[df["pid"].isin(near_pids)]
    sub_today = sub[sub["doy"] == doy_today]

    if sub_today.empty:
        FMI_  = sub["FMI"].mean()
        TMI_  = sub["TMI"].mean()
        NDVI_ = sub[NDVI_COL].mean()
        SUN_  = sub[SUN_COL].mean()
    else:
        FMI_  = sub_today["FMI"].mean()
        TMI_  = sub_today["TMI"].mean()
        NDVI_ = sub_today[NDVI_COL].mean()
        SUN_  = sub_today[SUN_COL].mean()

    ff_today = compute_FFDRI_new(dwi_today, FMI_, TMI_, SUN_, NDVI_)

    seq = []
    for off in range(SEQ_LEN-1, 0, -1):
        d = today - timedelta(days=off)
        doy = d.timetuple().tm_yday
        vals = []
        for pid in near_pids:
            key = (pid, doy)
            if key in clim_ffdri.index:
                vals.append(float(clim_ffdri.loc[key]))
        seq.append(np.mean(vals) if vals else clim_ffdri.mean())
    seq.append(ff_today)

    seq_std = standardize(np.array(seq)).reshape(1, SEQ_LEN, 1)
    y_std = model.predict(seq_std, verbose=0)
    y_pred = inv_standardize(y_std)[0]

    rec = {"name": site["name"], "lat": lat, "lon": lon, "FFDRI_today": ff_today}
    for i in range(HORIZON):
        rec[f"FFDRI_D{i+1}"] = float(y_pred[i])
        rec[f"risk_D{i+1}"] = risk_level(float(y_pred[i]))
    sites_records.append(rec)

sites_df = pd.DataFrame(sites_records)

# 6) D+TARGET_H 위험도 인덱스
risk_order = ["low", "moderate", "high", "very_high", "extreme"]
risk_to_idx = {r: i+1 for i, r in enumerate(risk_order)}

sites_df["risk_index"] = sites_df[f"risk_D{TARGET_H}"].map(risk_to_idx)

# 7) 행정경계 Shapefile 업로드 및 좌표계 설정
print("\nShapefile (.shp, .shx, .dbf, .prj)을 모두 업로드하세요.")
uploaded_shp = files.upload()

shp_files = [n for n in uploaded_shp.keys() if n.lower().endswith(".shp")]
boundary_shp = shp_files[0]

os.environ["SHAPE_RESTORE_SHX"] = "YES"
boundary = gpd.read_file(boundary_shp)

boundary = boundary.set_crs(epsg=5179, allow_override=True)
boundary_4326 = boundary.to_crs(epsg=4326)

b = boundary_4326.geometry.bounds
boundary_4326 = boundary_4326[b["maxx"] < 131]

g_sites = gpd.GeoDataFrame(
    sites_df,
    geometry=[Point(xy) for xy in zip(sites_df["lon"], sites_df["lat"])],
    crs="EPSG:4326"
)

# 8) 지도 시각화 (영문 타이틀)
fig, ax = plt.subplots(figsize=(10, 7))

boundary_4326.plot(ax=ax, color="white", edgecolor="black")

risk_colors = [
    "#FFCCCC",
    "#FF6666",
    "#FF0000",
    "#CC0000",
    "#800000",
]
cmap = ListedColormap(risk_colors)
bounds = np.arange(0.5, 6.5, 1)
norm = BoundaryNorm(bounds, cmap.N)

g_sites.plot(
    ax=ax,
    column="risk_index",
    cmap=cmap,
    norm=norm,
    markersize=180,
    edgecolor="black",
    linewidth=0.8,
    legend=True,
    vmin=1,
    vmax=5,
)

for _, row in g_sites.iterrows():
    ax.text(
        row.geometry.x + 0.02,
        row.geometry.y + 0.02,
        row["name"],
        fontsize=12
    )

ax.set_aspect("equal", adjustable="datalim")
ax.set_xlabel("Longitude", fontsize=12)
ax.set_ylabel("Latitude", fontsize=12)

forecast_date = today + timedelta(days=TARGET_H)

ax.set_title(
    f"FFDRI_new Risk Map (Forecast Date: {forecast_date}, D+{TARGET_H})",
    fontsize=16
)

minx, miny, maxx, maxy = boundary_4326.total_bounds
ax.set_xlim(minx-0.05, maxx+0.05)
ax.set_ylim(miny+0.02, maxy-0.02)

plt.tight_layout()
plt.show()

# 9) CSV 저장
out_csv = f"FFD_forecast_D{TARGET_H}_{today}.csv"
sites_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
files.download(out_csv)

# 10) PNG 저장
out_png = f"FFD_RISKMAP_D{TARGET_H}_{today}.png"
fig.savefig(out_png, dpi=300, bbox_inches="tight")
files.download(out_png)

