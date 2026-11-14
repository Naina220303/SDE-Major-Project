
import os
import sqlite3

from fastapi import FastAPI, HTTPException

from shared.schemas import PumpRecord
from simulation.pump_simulator import FEATURES


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DB = os.getenv(
    "DB_PATH",
    "/data/operational_data.db"
)

app = FastAPI(
    title="Data and Model Handler Service",
    version="3.0"
)


# --------------------------------------------------
# DATABASE CONNECTION
# --------------------------------------------------

def connect():
    os.makedirs(
        os.path.dirname(DB) or ".",
        exist_ok=True
    )

    con = sqlite3.connect(DB)

    cols = ", ".join(
        f"{feature} REAL"
        for feature in FEATURES
    )

    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {cols},
            failure INTEGER,
            scenario TEXT,
            severity REAL,
            elapsed_seconds INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    con.commit()

    return con


# --------------------------------------------------
# STARTUP
# --------------------------------------------------

@app.on_event("startup")
def startup():

    con = connect()
    con.close()


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "service": "data-handler",
        "status": "ok",
        "features": len(FEATURES)
    }


# --------------------------------------------------
# INSERT SENSOR READING
# --------------------------------------------------

@app.post("/readings")
def add_reading(record: PumpRecord):

    con = connect()

    try:
        data = record.model_dump()

        keys = FEATURES + [
            "failure",
            "scenario",
            "severity",
            "elapsed_seconds"
        ]

        names = ",".join(keys)
        marks = ",".join("?" for _ in keys)

        values = [
            data.get(key)
            for key in keys
        ]

        cursor = con.execute(
            f"""
            INSERT INTO readings ({names})
            VALUES ({marks})
            """,
            values
        )

        con.commit()

        reading_id = cursor.lastrowid

        return {
            "id": reading_id,
            "stored": True
        }

    finally:
        con.close()


# --------------------------------------------------
# RETRIEVE SENSOR READINGS
# --------------------------------------------------

@app.get("/readings")
def get_readings(
    limit: int = 10000,
    labelled_only: bool = False
):

    if limit < 1 or limit > 20000:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 20000."
        )

    con = connect()

    try:

        # IMPORTANT: Include id in SELECT
        selected_columns = (
            ["id"]
            + FEATURES
            + [
                "failure",
                "scenario",
                "severity",
                "elapsed_seconds"
            ]
        )

        query = (
            f"SELECT {','.join(selected_columns)} "
            "FROM readings"
        )

        if labelled_only:
            query += " WHERE failure IS NOT NULL"

        query += " ORDER BY id ASC LIMIT ?"

        rows = con.execute(
            query,
            (limit,)
        ).fetchall()

        records = [
            dict(zip(selected_columns, row))
            for row in rows
        ]

        return {
            "count": len(records),
            "records": records
        }

    finally:
        con.close()


# --------------------------------------------------
# OPERATOR FEEDBACK
# --------------------------------------------------

@app.post("/feedback/{reading_id}")
def feedback(
    reading_id: int,
    label: int = 0
):

    if label not in (0, 1):
        raise HTTPException(
            status_code=400,
            detail="Label must be 0 or 1."
        )

    con = connect()

    try:

        cursor = con.execute(
            """
            UPDATE readings
            SET failure = ?
            WHERE id = ?
            """,
            (label, reading_id)
        )

        con.commit()

        if cursor.rowcount == 0:
            raise HTTPException(
                status_code=404,
                detail="Reading not found."
            )

        return {
            "reading_id": reading_id,
            "operator_label": label,
            "updated": True
        }

    finally:
        con.close()


# --------------------------------------------------
# DATABASE STATISTICS
# --------------------------------------------------

@app.get("/stats")
def stats():

    con = connect()

    try:

        total = con.execute(
            "SELECT COUNT(*) FROM readings"
        ).fetchone()[0]

        labelled = con.execute(
            """
            SELECT COUNT(*)
            FROM readings
            WHERE failure IS NOT NULL
            """
        ).fetchone()[0]

        return {
            "total_readings": total,
            "labelled_readings": labelled
        }

    finally:
        con.close()
