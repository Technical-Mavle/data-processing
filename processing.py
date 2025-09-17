# In processing.py
import pandas as pd
import io
from config import supabase # Import the client from your config file

def download_file_from_storage(bucket_name: str, remote_file_path: str):
    """Downloads a file from Supabase Storage."""
    # Logic:
    # 1. Use supabase.storage.from_(bucket_name).download(remote_file_path)
    # 2. This returns the file content in bytes.
    # 3. Return the file content.
    response = supabase.storage.from_(bucket_name).download(remote_file_path)
    # Depending on the supabase-py version, download can return bytes directly
    # or a response-like object with .data.
    if isinstance(response, (bytes, bytearray)):
        return bytes(response)
    # Newer client returns a dict { data: bytes, error: None }
    data = getattr(response, "data", None) if not isinstance(response, dict) else response.get("data")
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    raise RuntimeError(f"Failed to download {remote_file_path} from bucket {bucket_name}")

def process_occurrence_data(file_bytes: bytes) -> pd.DataFrame:
    """Reads raw file bytes into a DataFrame and cleans it."""
    # Logic:
    # 1. Use io.BytesIO to treat the file_bytes as a file.
    # 2. Read the data into a Pandas DataFrame using pd.read_csv or pd.read_table.
    # 3. Perform cleaning steps:
    #    - Convert 'eventDate' column using pd.to_datetime.
    #    - Check if 'decimalLatitude' and 'decimalLongitude' are not null.
    #    - Drop rows with missing critical data.
    # 4. Return the cleaned DataFrame.
    buffer = io.BytesIO(file_bytes)
    # Try CSV first; if it fails, fallback to tab-delimited
    try:
        df = pd.read_csv(buffer)
    except Exception:
        buffer.seek(0)
        df = pd.read_csv(buffer, sep="\t")

    # Normalize expected columns if present
    if 'eventDate' in df.columns:
        df['eventDate'] = pd.to_datetime(df['eventDate'], errors='coerce', utc=True)

    # Ensure lat/lon are numeric
    if 'decimalLatitude' in df.columns:
        df['decimalLatitude'] = pd.to_numeric(df['decimalLatitude'], errors='coerce')
    if 'decimalLongitude' in df.columns:
        df['decimalLongitude'] = pd.to_numeric(df['decimalLongitude'], errors='coerce')

    # Drop rows missing critical fields if they exist
    critical_cols = [c for c in ['decimalLatitude', 'decimalLongitude'] if c in df.columns]
    if critical_cols:
        df = df.dropna(subset=critical_cols)

    # Optionally drop completely empty columns
    df = df.dropna(axis=1, how='all')
    return df


def upload_parquet_to_storage(bucket_name: str, remote_file_path: str, data: pd.DataFrame):
    """Converts a DataFrame to Parquet and uploads it to Supabase Storage."""
    # Logic:
    # 1. Convert the DataFrame to a Parquet file in memory (bytes)
    #    - Use data.to_parquet(engine='pyarrow')
    # 2. Upload the Parquet bytes to Supabase Storage.
    #    - Use supabase.storage.from_(bucket_name).upload(...)
    #    - You might need to set 'upsert=True' to allow overwriting files.
    parquet_bytes = data.to_parquet(engine='pyarrow')
    # Ensure bytes-like object
    if not isinstance(parquet_bytes, (bytes, bytearray)):
        # Some versions may return None and write to a buffer; handle explicitly
        buffer = io.BytesIO()
        data.to_parquet(buffer, engine='pyarrow')
        parquet_bytes = buffer.getvalue()

    # Upload with upsert to overwrite if exists
    result = supabase.storage.from_(bucket_name).upload(
        path=remote_file_path,
        file=parquet_bytes,
        file_options={"content-type": "application/octet-stream", "upsert": "true"}
    )

    # Return the uploaded path or raise on error-like responses
    if isinstance(result, dict) and result.get('error'):
        raise RuntimeError(result['error'])
    return result


def update_metadata_in_db(original_filename: str, processed_file_location: str, status: str):
    """Inserts a record into the 'datasets' table."""
    # Logic:
    # 1. Define the new row as a dictionary, e.g.,
    #    {'original_filename': original_filename, 'processed_file_location': processed_file_location, 'status': status}
    # 2. Use supabase.table('datasets').insert(new_row).execute()
    new_row = {
        'original_filename': original_filename,
        'processed_file_location': processed_file_location,
        'status': status,
    }
    # Match your provided schema/table name: public.file_metadata
    response = supabase.table('file_metadata').insert(new_row).execute()
    # Basic validation
    if getattr(response, 'error', None):
        raise RuntimeError(str(response.error))
    return response

# In processing.py (add this at the end)

def run_ingestion_pipeline(filename: str):
    """
    Orchestrates the entire file ingestion process.
    """
    raw_bucket = 'raw-uploads'
    processed_bucket = 'processed-data'
    processed_filename = filename.split('.')[0] + '.parquet'

    try:
        # 1. Download
        print(f"Downloading {filename}...")
        file_bytes = download_file_from_storage(raw_bucket, filename)

        # 2. Process
        print("Processing data...")
        cleaned_df = process_occurrence_data(file_bytes)

        # 3. Upload
        print(f"Uploading {processed_filename}...")
        upload_parquet_to_storage(processed_bucket, processed_filename, cleaned_df)

        # 4. Update Metadata
        print("Updating metadata...")
        update_metadata_in_db(
            original_filename=filename,
            processed_file_location=processed_filename,
            status='processed'
        )
        print("Pipeline completed successfully!")
        return {"status": "success", "processed_file": processed_filename}

    except Exception as e:
        print(f"Pipeline failed: {e}")
        # Optionally, update the database with a 'failed' status
        # update_metadata_in_db(original_filename=filename, processed_file_location='', status='failed')
        return {"status": "failed", "error": str(e)}