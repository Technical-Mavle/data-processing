# In processing.py
import pandas as pd
import io
import json
from config import supabase # Import the client from your config file
from typing import Tuple, Dict, Any

# Optional deps are imported lazily inside functions to avoid import errors at startup

def download_file_from_storage(bucket_name: str, remote_file_path: str):
    """Downloads a file from Supabase Storage."""
    response = supabase.storage.from_(bucket_name).download(remote_file_path)
    if isinstance(response, (bytes, bytearray)):
        return bytes(response)
    data = getattr(response, "data", None) if not isinstance(response, dict) else response.get("data")
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    raise RuntimeError(f"Failed to download {remote_file_path} from bucket {bucket_name}")

def process_occurrence_data(file_bytes: bytes) -> pd.DataFrame:
    """Reads raw tabular file bytes into a DataFrame and cleans it."""
    buffer = io.BytesIO(file_bytes)
    try:
        df = pd.read_csv(buffer)
    except Exception:
        buffer.seek(0)
        df = pd.read_csv(buffer, sep="\t")

    if 'eventDate' in df.columns:
        df['eventDate'] = pd.to_datetime(df['eventDate'], errors='coerce', utc=True)
    if 'decimalLatitude' in df.columns:
        df['decimalLatitude'] = pd.to_numeric(df['decimalLatitude'], errors='coerce')
    if 'decimalLongitude' in df.columns:
        df['decimalLongitude'] = pd.to_numeric(df['decimalLongitude'], errors='coerce')

    critical_cols = [c for c in ['decimalLatitude', 'decimalLongitude'] if c in df.columns]
    if critical_cols:
        df = df.dropna(subset=critical_cols)

    df = df.dropna(axis=1, how='all')
    return df

def process_tabular_data(file_bytes: bytes) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    """Processes tabular data and extracts metadata."""
    df = process_occurrence_data(file_bytes)
    metadata_payload = {
        "columns": list(df.columns),
        "inferred_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "total_rows": len(df)
    }
    return df, "tabular", metadata_payload

def process_netcdf_data(file_bytes: bytes) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    """Reads NetCDF bytes, extracts metadata, and returns a normalized DataFrame."""
    import xarray as xr
    buffer = io.BytesIO(file_bytes)
    ds = xr.open_dataset(buffer)
    try:
        df = ds.to_dataframe().reset_index()
        metadata_payload = {
            "dimensions": dict(ds.dims),
            "variables": list(ds.data_vars.keys()),
            "attributes": ds.attrs
        }
    finally:
        ds.close()
    df = df.dropna(axis=1, how='all')
    return df, "netcdf", metadata_payload

def process_otolith_image(file_bytes: bytes) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    """Extracts image metadata and returns it in a single-row DataFrame."""
    from PIL import Image
    buffer = io.BytesIO(file_bytes)
    with Image.open(buffer) as img:
        width, height = img.size
        metadata_payload = {
            'width': width,
            'height': height,
            'format': img.format,
            'mode': img.mode,
        }
        try:
            exif = getattr(img, "_getexif", lambda: None)() or {}
            for k, v in list(exif.items())[:10]:
                metadata_payload[f'exif_{k}'] = str(v)
        except Exception:
            pass
    return pd.DataFrame([metadata_payload]), "image", metadata_payload

def _sanitize_dataframe_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Coerces DataFrame to a Parquet-friendly schema."""
    sanitized = df.copy()
    sanitized.columns = [str(c).strip().replace("\n", " ").replace("\t", " ") for c in sanitized.columns]
    for col in sanitized.columns:
        series = sanitized[col]
        dtype = series.dtype
        if pd.api.types.is_categorical_dtype(dtype):
            sanitized[col] = series.astype(str)
        elif pd.api.types.is_timedelta64_dtype(dtype):
            sanitized[col] = (series / pd.Timedelta(milliseconds=1)).astype("Int64")
        elif pd.api.types.is_object_dtype(dtype):
            def to_jsonable(value):
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    return None
                if isinstance(value, (str, int, float, bool)):
                    return value
                try:
                    return json.loads(json.dumps(value, default=str))
                except Exception:
                    return str(value)
            sanitized[col] = sanitized[col].map(to_jsonable)
    sanitized = sanitized.dropna(axis=1, how='all')
    return sanitized

def upload_parquet_to_storage(bucket_name: str, remote_file_path: str, data: pd.DataFrame):
    """Converts a DataFrame to Parquet and uploads it to Supabase Storage."""
    parquet_bytes = data.to_parquet(engine='pyarrow')
    if not isinstance(parquet_bytes, (bytes, bytearray)):
        buffer = io.BytesIO()
        data.to_parquet(buffer, engine='pyarrow')
        parquet_bytes = buffer.getvalue()
    result = supabase.storage.from_(bucket_name).upload(
        path=remote_file_path,
        file=parquet_bytes,
        file_options={"content-type": "application/octet-stream", "upsert": "true"}
    )
    if isinstance(result, dict) and result.get('error'):
        raise RuntimeError(result['error'])
    return result

def update_metadata_in_db(original_filename: str, processed_file_location: str, status: str, file_type: str, metadata_payload: dict, geom: str | None):
    """Inserts a record with rich metadata and geometry into the 'file_metadata' table."""
    new_row = {
        'original_filename': original_filename,
        'processed_file_location': processed_file_location,
        'status': status,
        'file_type': file_type,
        'metadata_payload': metadata_payload,
        'geom': geom
    }
    response = supabase.table('file_metadata').insert(new_row).execute()
    if getattr(response, 'error', None):
        raise RuntimeError(str(response.error))
    return response

def create_bounding_box(df: pd.DataFrame) -> str | None:
    """
    Creates a WKT POLYGON representation of the data's bounding box.
    Returns None if coordinate columns are not found or data is invalid.
    """
    lat_col, lon_col = None, None
    for col in df.columns:
        if 'lat' in str(col).lower():
            lat_col = col
        if 'lon' in str(col).lower() or 'lng' in str(col).lower():
            lon_col = col

    if not (lat_col and lon_col):
        return None

    # Drop rows with invalid coordinate data before calculating min/max
    df_coords = df.dropna(subset=[lat_col, lon_col])
    if df_coords.empty:
        return None

    min_lon, max_lon = df_coords[lon_col].min(), df_coords[lon_col].max()
    min_lat, max_lat = df_coords[lat_col].min(), df_coords[lat_col].max()

    wkt_bbox = f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, {max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"
    return wkt_bbox

def run_ingestion_pipeline(filename: str):
    """Orchestrates the entire file ingestion process."""
    raw_bucket = 'raw-uploads'
    processed_bucket = 'processed-data'
    processed_filename = filename.split('.')[0] + '.parquet'
    try:
        print(f"Downloading {filename}...")
        file_bytes = download_file_from_storage(raw_bucket, filename)

        print("Detecting file type and processing...")
        lower_name = filename.lower()
        if lower_name.endswith(('.csv', '.txt', '.tsv')):
            cleaned_df, file_type, metadata_payload = process_tabular_data(file_bytes)
        elif lower_name.endswith(('.nc', '.nc4', '.cdf')):
            cleaned_df, file_type, metadata_payload = process_netcdf_data(file_bytes)
        elif lower_name.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
            cleaned_df, file_type, metadata_payload = process_otolith_image(file_bytes)
        else:
            raise ValueError(f"Unsupported file type: {filename}")

        # Create a bounding box from the processed data
        geom_wkt = create_bounding_box(cleaned_df)
        
        cleaned_df = _sanitize_dataframe_for_parquet(cleaned_df)
        
        print(f"Uploading {processed_filename}...")
        upload_parquet_to_storage(processed_bucket, processed_filename, cleaned_df)

        print("Updating metadata...")
        update_metadata_in_db(
            original_filename=filename,
            processed_file_location=processed_filename,
            status='processed',
            file_type=file_type,
            metadata_payload=metadata_payload,
            geom=geom_wkt
        )
        print("Pipeline completed successfully!")
        return {"status": "success", "processed_file": processed_filename}

    except Exception as e:
        print(f"Pipeline failed: {e}")
        return {"status": "failed", "error": str(e)}