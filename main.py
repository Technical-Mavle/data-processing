# In main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from processing import run_ingestion_pipeline

# --- Create the FastAPI app ---
app = FastAPI(
    title="Project SAGAR Ingestion Service",
    description="An API to trigger the data ingestion and processing pipeline.",
    version="1.0.0"
)

# --- Define the request data model ---
# This ensures that any request to our endpoint MUST have a 'filename' field.
class FileInput(BaseModel):
    filename: str

# --- Create the API endpoint ---
@app.post("/ingest-file")
async def create_ingestion_job(file: FileInput):
    """
    Receives a filename, triggers the ingestion pipeline, and returns the result.
    """
    print(f"Received request to ingest file: {file.filename}")
    
    # Run the orchestrator function from processing.py
    result = run_ingestion_pipeline(file.filename)
    
    # If the pipeline failed, return an HTTP error
    if result.get("status") == "failed":
        raise HTTPException(
            status_code=500, 
            detail=f"Pipeline failed. Error: {result.get('error')}"
        )
        
    # If successful, return a success message
    return {
        "message": "Ingestion pipeline completed successfully.",
        "details": result
    }

# --- Add a root endpoint for basic health check ---
@app.get("/")
def read_root():
    return {"status": "ok", "message": "SAGAR Ingestion Service is running."}