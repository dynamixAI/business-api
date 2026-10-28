import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse

from pipeline.clean import DataError, clean_transactions
from pipeline.segment import segment_customers, segments_by_store, summarise_segments

# biggest upload we accept, in MB
MAX_UPLOAD_MB = 50

# the app itself; title and description appear on the /docs page
app = FastAPI(
    title="Business API",
    description="Customer segmentation for a 10-store Greater Manchester retail chain",
    version="0.2.0",
)


@app.get("/", include_in_schema=False)
def home():
    # anyone opening the bare link lands on the interactive docs
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    # simple check that the API is up; the keep-awake ping will call this
    return {"status": "ok"}


@app.post("/segment")
def segment(file: UploadFile = File(..., description="Till export as a CSV file")):
    # only accept CSV files
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="please upload a .csv file")

    # refuse files that are too big before reading them
    size_mb = (file.size or 0) / 1_000_000
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(status_code=413, detail=f"file is {size_mb:.0f} MB, limit is {MAX_UPLOAD_MB} MB")

    # read everything as text so ids like 000161 keep their leading zeros
    try:
        raw = pd.read_csv(file.file, dtype=str, keep_default_na=False, na_values=[""])
    except Exception:
        raise HTTPException(status_code=400, detail="couldn't read the file as a CSV")

    # run the same pipeline stages as run_pipeline.py
    try:
        clean, report = clean_transactions(raw)
        rfm, snapshot = segment_customers(clean)
    except (DataError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e))

    # send back the cleaning report and the two summaries
    return {
        "file": file.filename,
        "snapshot_date": str(snapshot.date()),
        "customers_segmented": len(rfm),
        "cleaning_report": report,
        "segments": summarise_segments(rfm).to_dict(orient="records"),
        "segments_by_store": segments_by_store(rfm).to_dict(orient="records"),
    }
