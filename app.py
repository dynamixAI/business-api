from fastapi import FastAPI
from fastapi.responses import RedirectResponse

# the app itself; title and description appear on the /docs page
app = FastAPI(
    title="Business API",
    description="Customer segmentation for a 10-store Greater Manchester retail chain",
    version="0.1.0",
)


@app.get("/", include_in_schema=False)
def home():
    # anyone opening the bare link lands on the interactive docs
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    # simple check that the API is up; the keep-awake ping will call this
    return {"status": "ok"}
