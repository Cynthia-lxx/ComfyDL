"""
Dataset readers — load popular data formats into the universal ``DATASET`` type.

    text / string / csv / json / xlsx / db(sqlite) / accdb  ->  DATASET

Every reader is efficiency-minded: the heavy lifting is done by pandas (C engine,
``usecols``/``dtype`` honoured, ``low_memory=False`` to avoid silent upcasts) and
the resulting ``DataFrame`` is turned into tensors with ``torch.from_numpy`` (a
zero-copy view when the dtype already matches).  Optional dependencies (openpyxl
for xlsx, pyodbc for accdb) are imported lazily inside ``execute`` so a missing
driver raises a readable ``pip install`` hint instead of crashing module load.

Sharding (SP-6c): every reader takes ``start_idx``/``num_samples`` —
``num_samples=-1`` (the default) reads all rows, a positive value reads that
many rows starting at ``start_idx``.  The shard is applied to the parsed
DataFrame *after* the parse, so ``df.iloc[start:end]`` semantics are exact and
a shard can never collide with the cache (see below).

Caching (SP-6c): the raw parsed ``DataFrame`` is cached under the hot switch
``comfydl.dataio.cache`` (see :mod:`._dataio_cache` for the key construction,
the byte budget and the self-metering).  The shard is applied *after* the
cache, so shard changes are structurally immune to stale-cache bugs and N
shards share one cached parse.  Every parameter that affects the parsed
DataFrame (delimiter/header/sheet/query text, the file's mtime_ns+size)
IS part of the key — a changed file or a changed parse parameter produces a
new key.  ``target``/``dtype`` are applied after the parse and are therefore
not part of it.

Empty-value rule: a source with 0 rows, a ``start_idx`` beyond the table, or
a shard that selects nothing raises a readable ``ValueError`` — readers never
emit an empty ``DATASET``.

All readers live in the ``d2l/Datasets`` category and output a single ``DATASET``.
Writers live in :mod:`comfydl.nodes.data_io_writers` (batch 3).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

from . import _dataio_cache as _dcache
from .data_types import CdlDataset

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Declare the hot switch (idempotent; no-op outside the host registry).
_dcache.register()


def _target(target: str) -> Optional[str]:
    """Empty target string means "unlabelled dataset"."""
    return target if target else None


def _resolve_input_path(file: str) -> str:
    """Resolve a user-supplied path the ComfyUI way (SP-6c hotfix).

    ComfyUI convention: file inputs live in the ``input/`` directory and users
    type bare filenames.  Resolution order:

    1. absolute path — used as-is (backwards compatible, sandbox/bench safe);
    2. bare/relative name — first tried against ``folder_paths``
       ``get_input_directory()`` (the ``<ComfyUI>\\input\\`` root);
    3. then CWD-relative (legacy behaviour, what the smoke harness relies on
       with its ``chdir`` sandbox).

    Raises a readable ``FileNotFoundError`` naming every candidate when
    nothing resolves — a bare ``pandas`` ENOENT never says where it looked.
    """
    p = str(file or "").strip()
    if not p:
        return p
    cand = Path(p)
    if cand.is_absolute():
        return p
    tried = []
    try:
        import folder_paths

        merged = Path(folder_paths.get_input_directory()) / p
        tried.append(str(merged))
        if merged.is_file():
            return str(merged)
    except Exception:  # noqa: BLE001 - no host: straight to the CWD fallback
        pass
    tried.append(str(Path.cwd() / p))
    if cand.is_file():
        return p
    raise FileNotFoundError(
        f"input file {p!r} not found — tried: " + "; ".join(tried) +
        " (absolute paths are used as-is; bare filenames are looked up in "
        "the ComfyUI input directory first)"
    )


def _shard(df, start_idx, num_samples, source: str):
    """Apply the ``[start : start+num]`` row window (``num<0`` = to the end).

    Raises a readable ``ValueError`` instead of ever returning an empty
    selection — an empty ``DATASET`` downstream is always a bug the user
    should hear about at the reader, not a silent 0-row tensor.
    """
    n = len(df)
    start = int(start_idx)
    num = int(num_samples)
    if start < 0:
        raise ValueError(f"{source}: start_idx must be >= 0, got {start}")
    if num == 0:
        raise ValueError(
            f"{source}: num_samples=0 is ambiguous — use -1 for 'all rows' "
            "or a positive count."
        )
    if n == 0:
        raise ValueError(f"{source}: the source has 0 rows (empty file/query).")
    if start >= n:
        raise ValueError(
            f"{source}: start_idx {start} is out of range for a {n}-row source."
        )
    end = n if num < 0 else min(n, start + num)
    out = df.iloc[start:end]
    if len(out) == 0:
        raise ValueError(
            f"{source}: the shard [{start}:{end}) selects no rows "
            f"(source has {n} rows)."
        )
    return out


class CdlReadCSV:
    """Read a CSV file into a ``DATASET`` (optionally pinned to a label column)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file": ("STRING", {"default": "cdl_dataset.csv"}),
                "target": ("STRING", {"default": ""}),
            },
            "optional": {
                "delimiter": ("STRING", {"default": ","}),
                "dtype": (["float32", "float64"], {"default": "float32"}),
                "header": ("BOOLEAN", {"default": True}),
                "start_idx": ("INT", {"default": 0, "min": 0, "max": 1000000000}),
                "num_samples": ("INT", {"default": -1, "min": -1, "max": 1000000000}),
            },
        }

    RETURN_TYPES = ("DATASET",)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "execute"
    CATEGORY = "d2l/Datasets"

    def execute(self, file, target, delimiter=",", dtype="float32", header=True,
                start_idx=0, num_samples=-1):
        import pandas as pd

        file = _resolve_input_path(file)
        key = None
        hit, df = False, None
        if _dcache.enabled():
            with _dcache.measure():
                key = _dcache.cache_key("csv", _dcache.file_stamp(file),
                                        delimiter, bool(header))
                hit, df = _dcache.lookup(key)
        if not hit:
            df = pd.read_csv(
                file, sep=delimiter, header=0 if header else None, low_memory=False
            )
            if key is not None:
                with _dcache.measure():
                    _dcache.store(key, df)
        df = _shard(df, start_idx, num_samples, source=f"CSV file {file!r}")
        return (CdlDataset.from_dataframe(df, target=_target(target), dtype=dtype),)


class CdlReadText:
    """Read whitespace/comma/tab-delimited text into a ``DATASET``.

    Useful for plain ``.txt`` dumps that hold tabular numbers (no header by
    default).  The Python CSV engine is used so any single-character delimiter
    (including arbitrary whitespace) is honoured.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file": ("STRING", {"default": "cdl_dataset.txt"}),
                "target": ("STRING", {"default": ""}),
            },
            "optional": {
                "delimiter": ("STRING", {"default": "\t"}),
                "dtype": (["float32", "float64"], {"default": "float32"}),
                "header": ("BOOLEAN", {"default": False}),
                "start_idx": ("INT", {"default": 0, "min": 0, "max": 1000000000}),
                "num_samples": ("INT", {"default": -1, "min": -1, "max": 1000000000}),
            },
        }

    RETURN_TYPES = ("DATASET",)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "execute"
    CATEGORY = "d2l/Datasets"

    def execute(self, file, target, delimiter="\t", dtype="float32", header=False,
                start_idx=0, num_samples=-1):
        import pandas as pd

        file = _resolve_input_path(file)
        key = None
        hit, df = False, None
        if _dcache.enabled():
            with _dcache.measure():
                key = _dcache.cache_key("text", _dcache.file_stamp(file),
                                        delimiter, bool(header))
                hit, df = _dcache.lookup(key)
        if not hit:
            df = pd.read_csv(
                file,
                sep=delimiter,
                header=0 if header else None,
                engine="python",
            )
            if key is not None:
                with _dcache.measure():
                    _dcache.store(key, df)
        df = _shard(df, start_idx, num_samples, source=f"text file {file!r}")
        return (CdlDataset.from_dataframe(df, target=_target(target), dtype=dtype),)


class CdlReadString:
    """Parse an inline ``STRING`` of delimited numbers into a ``DATASET``.

    Handy for pasting a small sample directly into the node without a file.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "a\tb\tc\ty\n0\t0\t0\t0.0\n1\t2\t3\t0.5\n2\t4\t6\t1.0\n",
                    },
                ),
                "target": ("STRING", {"default": ""}),
            },
            "optional": {
                "delimiter": ("STRING", {"default": "\t"}),
                "dtype": (["float32", "float64"], {"default": "float32"}),
                "header": ("BOOLEAN", {"default": True}),
                "start_idx": ("INT", {"default": 0, "min": 0, "max": 1000000000}),
                "num_samples": ("INT", {"default": -1, "min": -1, "max": 1000000000}),
            },
        }

    RETURN_TYPES = ("DATASET",)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "execute"
    CATEGORY = "d2l/Datasets"

    def execute(self, text, target, delimiter="\t", dtype="float32", header=True,
                start_idx=0, num_samples=-1):
        import pandas as pd

        key = None
        hit, df = False, None
        if _dcache.enabled():
            with _dcache.measure():
                key = _dcache.cache_key("string", _dcache.md5_hex(text),
                                        delimiter, bool(header))
                hit, df = _dcache.lookup(key)
        if not hit:
            df = pd.read_csv(
                io.StringIO(text),
                sep=delimiter,
                header=0 if header else None,
                engine="python",
            )
            if key is not None:
                with _dcache.measure():
                    _dcache.store(key, df)
        df = _shard(df, start_idx, num_samples, source="inline STRING table")
        return (CdlDataset.from_dataframe(df, target=_target(target), dtype=dtype),)


class CdlReadJSON:
    """Read a JSON file (list-of-records or column dictionary) into a ``DATASET``."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file": ("STRING", {"default": "cdl_dataset.json"}),
                "target": ("STRING", {"default": ""}),
            },
            "optional": {
                "dtype": (["float32", "float64"], {"default": "float32"}),
                "start_idx": ("INT", {"default": 0, "min": 0, "max": 1000000000}),
                "num_samples": ("INT", {"default": -1, "min": -1, "max": 1000000000}),
            },
        }

    RETURN_TYPES = ("DATASET",)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "execute"
    CATEGORY = "d2l/Datasets"

    def execute(self, file, target, dtype="float32", start_idx=0, num_samples=-1):
        import pandas as pd

        file = _resolve_input_path(file)
        key = None
        hit, df = False, None
        if _dcache.enabled():
            with _dcache.measure():
                key = _dcache.cache_key("json", _dcache.file_stamp(file))
                hit, df = _dcache.lookup(key)
        if not hit:
            df = pd.read_json(file)
            if key is not None:
                with _dcache.measure():
                    _dcache.store(key, df)
        df = _shard(df, start_idx, num_samples, source=f"JSON file {file!r}")
        return (CdlDataset.from_dataframe(df, target=_target(target), dtype=dtype),)


class CdlReadXLSX:
    """Read an Excel ``.xlsx`` / ``.xls`` workbook into a ``DATASET``."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file": ("STRING", {"default": "cdl_dataset.xlsx"}),
                "target": ("STRING", {"default": ""}),
            },
            "optional": {
                "sheet": ("STRING", {"default": "0"}),
                "dtype": (["float32", "float64"], {"default": "float32"}),
                "start_idx": ("INT", {"default": 0, "min": 0, "max": 1000000000}),
                "num_samples": ("INT", {"default": -1, "min": -1, "max": 1000000000}),
            },
        }

    RETURN_TYPES = ("DATASET",)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "execute"
    CATEGORY = "d2l/Datasets"

    def execute(self, file, target, sheet="0", dtype="float32",
                start_idx=0, num_samples=-1):
        import pandas as pd

        file = _resolve_input_path(file)
        key = None
        hit, df = False, None
        if _dcache.enabled():
            with _dcache.measure():
                key = _dcache.cache_key("xlsx", _dcache.file_stamp(file), sheet)
                hit, df = _dcache.lookup(key)
        if not hit:
            sheet_name = sheet if sheet not in ("", "0") else 0
            df = pd.read_excel(file, sheet_name=sheet_name)
            if key is not None:
                with _dcache.measure():
                    _dcache.store(key, df)
        df = _shard(df, start_idx, num_samples, source=f"XLSX file {file!r}")
        return (CdlDataset.from_dataframe(df, target=_target(target), dtype=dtype),)


class CdlReadDB:
    """Run a SQL query against a SQLite database file into a ``DATASET``.

    Uses the already-bundled SQLAlchemy engine; the ``query`` may be any
    ``SELECT`` returning columns of numbers (optionally one label column).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "db_path": ("STRING", {"default": "cdl_dataset.db"}),
                "query": ("STRING", {"multiline": True, "default": "SELECT * FROM data"}),
                "target": ("STRING", {"default": ""}),
            },
            "optional": {
                "dtype": (["float32", "float64"], {"default": "float32"}),
                "start_idx": ("INT", {"default": 0, "min": 0, "max": 1000000000}),
                "num_samples": ("INT", {"default": -1, "min": -1, "max": 1000000000}),
            },
        }

    RETURN_TYPES = ("DATASET",)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "execute"
    CATEGORY = "d2l/Datasets"

    def execute(self, db_path, query, target, dtype="float32",
                start_idx=0, num_samples=-1):
        import pandas as pd
        from sqlalchemy import create_engine, text

        db_path = _resolve_input_path(db_path)
        key = None
        hit, df = False, None
        if _dcache.enabled():
            with _dcache.measure():
                key = _dcache.cache_key("db", _dcache.file_stamp(db_path),
                                        _dcache.md5_hex(query))
                hit, df = _dcache.lookup(key)
        if not hit:
            engine = create_engine(f"sqlite:///{db_path}")
            with engine.connect() as conn:
                df = pd.read_sql_query(text(query), conn)
            if key is not None:
                with _dcache.measure():
                    _dcache.store(key, df)
        df = _shard(df, start_idx, num_samples, source=f"SQLite {db_path!r}")
        return (CdlDataset.from_dataframe(df, target=_target(target), dtype=dtype),)


class CdlReadAccDB:
    """Read a Microsoft Access ``.accdb`` / ``.mdb`` database via pyodbc.

    Requires the ``pyodbc`` package **and** the Microsoft Access Database Engine
    driver, neither of which ships with the project.  The import is lazy so a
    missing driver yields a readable ``pip install pyodbc`` hint instead of a
    module-load crash.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "db_path": ("STRING", {"default": ""}),
                "query": ("STRING", {"multiline": True, "default": "SELECT * FROM data"}),
                "target": ("STRING", {"default": ""}),
            },
            "optional": {
                "dtype": (["float32", "float64"], {"default": "float32"}),
                "start_idx": ("INT", {"default": 0, "min": 0, "max": 1000000000}),
                "num_samples": ("INT", {"default": -1, "min": -1, "max": 1000000000}),
            },
        }

    RETURN_TYPES = ("DATASET",)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "execute"
    CATEGORY = "d2l/Datasets"

    def execute(self, db_path, query, target, dtype="float32",
                start_idx=0, num_samples=-1):
        try:
            import pyodbc
        except ImportError as exc:  # pragma: no cover - optional driver
            raise ImportError(
                "CdlReadAccDB requires pyodbc and the Microsoft Access Database "
                "Engine driver. Install pyodbc with `pip install pyodbc` and "
                "install the Access driver from Microsoft."
            ) from exc
        import pandas as pd

        db_path = _resolve_input_path(db_path)
        key = None
        hit, df = False, None
        if _dcache.enabled():
            with _dcache.measure():
                key = _dcache.cache_key("accdb", _dcache.file_stamp(db_path),
                                        _dcache.md5_hex(query))
                hit, df = _dcache.lookup(key)
        if not hit:
            conn = pyodbc.connect(
                f"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={db_path};"
            )
            try:
                df = pd.read_sql(query, conn)
            finally:
                conn.close()
            if key is not None:
                with _dcache.measure():
                    _dcache.store(key, df)
        df = _shard(df, start_idx, num_samples, source=f"Access DB {db_path!r}")
        return (CdlDataset.from_dataframe(df, target=_target(target), dtype=dtype),)


NODE_CLASS_MAPPINGS.update({
    "CdlReadCSV": CdlReadCSV,
    "CdlReadText": CdlReadText,
    "CdlReadString": CdlReadString,
    "CdlReadJSON": CdlReadJSON,
    "CdlReadXLSX": CdlReadXLSX,
    "CdlReadDB": CdlReadDB,
    "CdlReadAccDB": CdlReadAccDB,
})
NODE_DISPLAY_NAME_MAPPINGS.update({
    "CdlReadCSV": "Read CSV → Dataset",
    "CdlReadText": "Read Text → Dataset",
    "CdlReadString": "Read String → Dataset",
    "CdlReadJSON": "Read JSON → Dataset",
    "CdlReadXLSX": "Read XLSX → Dataset",
    "CdlReadDB": "Read DB → Dataset",
    "CdlReadAccDB": "Read AccDB → Dataset",
})
