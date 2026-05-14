import sqlite3

import pandas as pd
import streamlit as st

from utils.config import DB_PATH, TABLE_COLUMNS, CHROMA_DB_PATH, CHROMA_SETTINGS


@st.cache_data(ttl=300)
def load_all_data():
    """Load all tables from SQLite."""
    tables = {name: pd.DataFrame(columns=cols) for name, cols in TABLE_COLUMNS.items()}
    if not DB_PATH.exists():
        return tables

    conn = sqlite3.connect(DB_PATH)
    for table, columns in TABLE_COLUMNS.items():
        try:
            tables[table] = pd.read_sql(f"SELECT * FROM {table}", conn)
        except Exception:
            tables[table] = pd.DataFrame(columns=columns)
    conn.close()

    if not tables["supplier_kpis"].empty:
        tables["supplier_kpis"]["year_month"] = pd.to_datetime(
            tables["supplier_kpis"]["year_month"])

    for _tbl, _col in [
        ("claims",          "creation_date"),
        ("audits",          "audit_date"),
        ("apqp_projects",   "customer_sop_date"),
        ("apqp_projects",   "supplier_sop_date"),
        ("external_events", "event_date"),
        ("external_events", "response_due_date"),
    ]:
        if not tables[_tbl].empty and _col in tables[_tbl].columns:
            tables[_tbl][_col] = pd.to_datetime(tables[_tbl][_col], errors="coerce")

    return tables


@st.cache_resource(show_spinner=False)
def get_kb_chunk_count() -> int:
    """Return live ChromaDB chunk count; falls back to last-known value."""
    try:
        import chromadb
        _client = chromadb.PersistentClient(
            path=str(CHROMA_DB_PATH), settings=CHROMA_SETTINGS)
        return _client.get_collection("supplier_kb").count()
    except Exception:
        return 264
