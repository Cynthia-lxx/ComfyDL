"""
Dataset writers — export a ``DATASET`` to popular formats (batch 3).

    DATASET  ->  csv / json / xlsx / db  (+ a STRING preview)

All writers are thin, efficient wrappers over pandas: the ``CdlDataset`` is
materialised with :meth:`CdlDataset.to_dataframe` (a copy, so the caller's data
is never mutated) and then serialised with the matching ``DataFrame.to_*`` call.
``CdlWriteDB`` streams via ``to_sql(chunksize=...)`` so a large table never has
to be buffered twice.  As with the readers, optional deps (openpyxl for xlsx)
are imported lazily inside ``execute``.

Because the only required input is a ``DATASET`` (which the smoke harness cannot
synthesise), these nodes are naturally skipped during the auto smoke run and are
covered by ``cdl_smoke_tests/test_data_io.py``.
"""

from __future__ import annotations

from .data_types import CdlDataset

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


class CdlWriteCSV:
    """Write a ``DATASET`` to a CSV file; returns the path for chaining."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("DATASET", {"forceInput": True}),
                "file": ("STRING", {"default": "dataset_out.csv"}),
            },
            "optional": {
                "index": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file",)
    FUNCTION = "execute"
    CATEGORY = "d2l/Datasets"

    def execute(self, dataset: CdlDataset, file: str, index: bool = False):
        df = dataset.to_dataframe()
        df.to_csv(file, index=index)
        return (file,)


class CdlWriteJSON:
    """Write a ``DATASET`` to a JSON file (records orientation); returns the path."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("DATASET", {"forceInput": True}),
                "file": ("STRING", {"default": "dataset_out.json"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file",)
    FUNCTION = "execute"
    CATEGORY = "d2l/Datasets"

    def execute(self, dataset: CdlDataset, file: str):
        df = dataset.to_dataframe()
        df.to_json(file, orient="records")
        return (file,)


class CdlWriteXLSX:
    """Write a ``DATASET`` to an Excel ``.xlsx`` file; returns the path.

    Needs ``openpyxl`` (imported lazily so a missing driver raises a readable
    ``pip install openpyxl`` hint instead of crashing module load).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("DATASET", {"forceInput": True}),
                "file": ("STRING", {"default": "dataset_out.xlsx"}),
            },
            "optional": {
                "sheet": ("STRING", {"default": "Sheet1"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file",)
    FUNCTION = "execute"
    CATEGORY = "d2l/Datasets"

    def execute(self, dataset: CdlDataset, file: str, sheet: str = "Sheet1"):
        try:
            import openpyxl  # noqa: F401  (ensures the engine is importable)
        except ImportError as exc:  # pragma: no cover - optional driver
            raise ImportError(
                "CdlWriteXLSX requires openpyxl. Install it with `pip install openpyxl`."
            ) from exc
        df = dataset.to_dataframe()
        df.to_excel(file, sheet_name=sheet, index=False)
        return (file,)


class CdlWriteDB:
    """Write a ``DATASET`` to a SQLite table; returns the db path for chaining.

    Uses the bundled SQLAlchemy engine and streams with ``chunksize`` so a large
    table is written in batches rather than buffered in memory twice.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("DATASET", {"forceInput": True}),
                "db_path": ("STRING", {"default": "dataset_out.db"}),
                "table": ("STRING", {"default": "data"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("db_path",)
    FUNCTION = "execute"
    CATEGORY = "d2l/Datasets"

    def execute(self, dataset: CdlDataset, db_path: str, table: str = "data"):
        df = dataset.to_dataframe()
        from sqlalchemy import create_engine

        engine = create_engine(f"sqlite:///{db_path}")
        df.to_sql(table, engine, if_exists="replace", index=False, chunksize=1000)
        return (db_path,)


class CdlDatasetPreview:
    """Render a ``DATASET`` head as a ``STRING`` for quick inspection."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("DATASET", {"forceInput": True}),
            },
            "optional": {
                "head": ("INT", {"default": 5, "min": 1, "max": 1000, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "execute"
    CATEGORY = "d2l/Datasets"

    def execute(self, dataset: CdlDataset, head: int = 5):
        df = dataset.to_dataframe()
        return (df.head(head).to_string(),)


NODE_CLASS_MAPPINGS.update({
    "CdlWriteCSV": CdlWriteCSV,
    "CdlWriteJSON": CdlWriteJSON,
    "CdlWriteXLSX": CdlWriteXLSX,
    "CdlWriteDB": CdlWriteDB,
    "CdlDatasetPreview": CdlDatasetPreview,
})
NODE_DISPLAY_NAME_MAPPINGS.update({
    "CdlWriteCSV": "Dataset → Write CSV",
    "CdlWriteJSON": "Dataset → Write JSON",
    "CdlWriteXLSX": "Dataset → Write XLSX",
    "CdlWriteDB": "Dataset → Write DB",
    "CdlDatasetPreview": "Dataset → Preview",
})
