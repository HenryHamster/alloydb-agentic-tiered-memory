import os

import google.auth
import psycopg2
import redis
from google.auth.transport.requests import Request

# Initialize database & Valkey connection clients from environment variables
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER")

VALKEY_HOST = os.getenv("VALKEY_HOST", "127.0.0.1")
VALKEY_PORT = int(os.getenv("VALKEY_PORT", "6379"))


def get_db_connection(*, credentials=None):
    # ADC uses the attached VM identity when no local credentials were configured.
    # The developer worker passes Compute Engine credentials explicitly, so a user's
    # gcloud login or ADC file cannot silently replace its service identity.
    if credentials is None:
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    if not credentials.valid:
        credentials.refresh(Request())
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=credentials.token,
        sslmode="require",
        connect_timeout=20,
    )


def get_valkey_client():
    return redis.Redis(
        host=VALKEY_HOST, port=VALKEY_PORT, db=0, socket_timeout=5.0, socket_connect_timeout=5.0
    )


if __name__ == "__main__":
    print(
        f"Connecting to AlloyDB via IAM Auth ({DB_USER}) at {DB_HOST}:{DB_PORT} and Valkey at {VALKEY_HOST}:{VALKEY_PORT}..."
    )
    print("Database and Valkey connection client modules loaded successfully.")
