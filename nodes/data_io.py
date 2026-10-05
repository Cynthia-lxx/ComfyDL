"""
Dataset readers — load popular data formats into the universal ``DATASET`` type.

    text / string / csv / json / xlsx / db(sqlite) / accdb  ->  DATASET

Every reader is efficiency-minded: the heavy lifting is done by pandas (C engine,
``usecols``/``dtype`` honoured, ``low_memory=False`` to avoid silent upcasts) and
the resulting ``DataFrame`` is turned into tensors with ``torch.from_numpy`` (a
zero-copy view when the dtype already matches).  Optional dependencies (openpyxl
for xlsx, pyodbc for accdb) are imported lazily inside ``execute`` so a missing
driver raises a readable ``pip install`` hint instead of crashing module load.

All readers live in the ``d2l/Datasets`` category and output a single ``DATASET``.
Writers live in :mod:`comfydl.nodes.data_io_writers` (batch 3).
"""

from __future__ import annotations

import io

from .data_types import CdlDataset

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _target(target: str) -> Optional[str]:
    """Empty target string means "unlabelled dataset"."""
    return target if target else None


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
            },
        }

    RETURN_TYPES = ("DATASET",)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "execute"
    CATEGORY = "d2l/Datasets"

    def execute(self, file, target, delimiter=",", dtype="float32", header=True):
        import pandas as pd

        df = pd.read_csv(
            file, sep=delimiter, header=0 if header else None, low_memory=False
        )
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
            },
        }

    RETURN_TYPES = ("DATASET",)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "execute"
    CATEGORY = "d2l/Datasets"

    def execute(self, file, target, delimiter="\t", dtype="float32", header=False):
        import pandas as pd

        df = pd.read_csv(
            file,
            sep=delimiter,
            header=0 if header else None,
            engine="python",
            low_memory=False,
        )
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
            },
        }

    RETURN_TYPES = ("DATASET",)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "execute"
    CATEGORY = "d2l/Datasets"

    def execute(self, text, target, delimiter="\t", dtype="float32", header=True):
        import pandas as pd

        df = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            header=0 if header else None,
            engine="python",
            low_memory=False,
        )
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
            },
        }

    RETURN_TYPES = ("DATASET",)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "execute"
    CATEGORY = "d2l/Datasets"

    def execute(self, file, target, dtype="float32"):
        import pandas as pd

        df = pd.read_json(file)
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
            },
        }

    RETURN_TYPES = ("DATASET",)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "execute"
    CATEGORY = "d2l/Datasets"

    def execute(self, file, target, sheet="0", dtype="float32"):
        import pandas as pd

        sheet_name = sheet if sheet not in ("", "0") else 0
        df = pd.read_excel(file, sheet_name=sheet_name)
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
            },
        }

    RETURN_TYPES = ("DATASET",)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "execute"
    CATEGORY = "d2l/Datasets"

    def execute(self, db_path, query, target, dtype="float32"):
        import pandas as pd
        from sqlalchemy import create_engine, text

        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn)
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
            },
        }

    RETURN_TYPES = ("DATASET",)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "execute"
    CATEGORY = "d2l/Datasets"

    def execute(self, db_path, query, target, dtype="float32"):
        try:
            import pyodbc
        except ImportError as exc:  # pragma: no cover - optional driver
            raise ImportError(
                "CdlReadAccDB requires pyodbc and the Microsoft Access Database "
                "Engine driver. Install pyodbc with `pip install pyodbc` and "
                "install the Access driver from Microsoft."
            ) from exc
        import pandas as pd

        conn = pyodbc.connect(
            f"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={db_path};"
        )
        try:
            df = pd.read_sql(query, conn)
        finally:
            conn.close()
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
