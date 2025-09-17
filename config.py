# In config.py

import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Get Supabase credentials from environment variables
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

# Create a single Supabase client instance
supabase: Client = create_client(url, key)