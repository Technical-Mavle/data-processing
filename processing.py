# In processing.py
import pandas as pd
import io
import json
from config import supabase # Import the client from your config file
from typing import Tuple

# Optional deps are imported lazily inside functions to avoid import errors at startup

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


def process_tabular_data(file_bytes: bytes) -> pd.DataFrame:
    """Compatibility wrapper for tabular data (CSV/TXT)."""
    return process_occurrence_data(file_bytes)


def process_netcdf_data(file_bytes: bytes) -> pd.DataFrame:
    """Reads NetCDF bytes using xarray and returns a normalized DataFrame.

    The function attempts to open the in-memory bytes as a dataset and flattens
    it into a tabular form using .to_dataframe().
    """
    # Lazy import so the app can start even if optional deps aren't installed yet
    import xarray as xr  # type: ignore

    buffer = io.BytesIO(file_bytes)
    # Let xarray pick the appropriate engine (netcdf4 or h5netcdf) based on availability
    ds = xr.open_dataset(buffer)
    try:
        df = ds.to_dataframe().reset_index()
    finally:
        ds.close()

    # Drop completely empty columns for cleanliness
    df = df.dropna(axis=1, how='all')
    return df


def process_otolith_image(file_bytes: bytes) -> pd.DataFrame:
    """Extract basic image metadata (width, height, format, mode)."""
    from PIL import Image  # type: ignore

    buffer = io.BytesIO(file_bytes)
    with Image.open(buffer) as img:
        width, height = img.size
        info = {
            'width': width,
            'height': height,
            'format': img.format,
            'mode': img.mode,
        }

        # Attempt EXIF extraction if available
        try:
            exif = getattr(img, "_getexif", lambda: None)() or {}
            # Flatten minimal EXIF keys if present
            for k, v in list(exif.items())[:10]:
                info[f'exif_{k}'] = str(v)
        except Exception:
            pass

    return pd.DataFrame([info])


def _sanitize_dataframe_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce DataFrame to Parquet-friendly schema.

    - Ensure string column names
    - Convert objects, dicts, lists to JSON strings
    - Convert categorical to string
    - Convert timedeltas to integer milliseconds
    """
    sanitized = df.copy()
    # Normalize column names
    sanitized.columns = [str(c).strip().replace("\n", " ").replace("\t", " ") for c in sanitized.columns]

    for col in sanitized.columns:
        series = sanitized[col]
        dtype = series.dtype
        if pd.api.types.is_categorical_dtype(dtype):
            sanitized[col] = series.astype(str)
        elif pd.api.types.is_timedelta64_dtype(dtype):
            # store as integer milliseconds
            sanitized[col] = (series / pd.Timedelta(milliseconds=1)).astype("Int64")
        elif pd.api.types.is_object_dtype(dtype):
            def to_jsonable(value):
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    return None
                # Simple scalars
                if isinstance(value, (str, int, float, bool)):
                    return value
                # Try JSON dump for lists/dicts and other objects
                try:
                    return json.loads(json.dumps(value, default=str))
                except Exception:
                    return str(value)

            sanitized[col] = sanitized[col].map(to_jsonable)
        # datetime64 (with or without tz) is handled by pyarrow; leave as-is

    # Drop columns that are entirely NaN and unnamed artifacts
    sanitized = sanitized.dropna(axis=1, how='all')
    return sanitized


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

        # 2. Route & Process based on file type
        print("Detecting file type and processing...")
        lower_name = filename.lower()
        if lower_name.endswith('.csv') or lower_name.endswith('.txt') or lower_name.endswith('.tsv'):
            cleaned_df = process_tabular_data(file_bytes)
        elif lower_name.endswith('.nc') or lower_name.endswith('.nc4') or lower_name.endswith('.cdf'):
            cleaned_df = process_netcdf_data(file_bytes)
        elif lower_name.endswith('.jpg') or lower_name.endswith('.jpeg') or lower_name.endswith('.png') or lower_name.endswith('.tif') or lower_name.endswith('.tiff'):
            cleaned_df = process_otolith_image(file_bytes)
        else:
            raise ValueError(f"Unsupported file type: {filename}")

        # Ensure Parquet-safe schema
        cleaned_df = _sanitize_dataframe_for_parquet(cleaned_df)

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