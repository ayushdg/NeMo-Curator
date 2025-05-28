"""Task data structures for the ray-curator pipeline framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, Union

import pandas as pd
import pyarrow as pa

T = TypeVar("T")


@dataclass
class Task(ABC, Generic[T]):
    """Abstract base class for tasks in the pipeline.

    A task represents a batch of data to be processed. Different modalities
    (text, audio, video) can implement their own task types.

    Attributes:
        task_id: Unique identifier for this task
        dataset_name: Name of the dataset this task belongs to
        data: The actual data (type depends on modality)
        metadata: Task-level metadata
        stage_history: List of stages this task has passed through
    """

    task_id: str
    dataset_name: str
    data: T
    metadata: dict[str, Any] = field(default_factory=dict)
    stage_history: list[str] = field(default_factory=list)

    @abstractmethod
    def num_items(self) -> int:
        """Get the number of items in this task."""

    def add_stage(self, stage_name: str) -> None:
        """Add a stage to the processing history."""
        self.stage_history.append(stage_name)

    @abstractmethod
    def validate(self) -> bool:
        """Validate the task data."""


@dataclass
class DocumentBatch(Task[Union[pa.Table, pd.DataFrame]]):
    """Task for processing batches of text documents.

    Documents are stored as a dataframe (PyArrow table or Pandas DataFrame).
    The schema is flexible - users can specify which columns contain text
    and other relevant data.

    Attributes:
        text_column: Name of the column containing text content
        id_column: Name of the column containing document IDs (optional)
        additional_columns: List of other columns to preserve during processing
    """

    text_column: str = "content"
    id_column: str | None = "id"
    additional_columns: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate the document batch."""
        self.validate()

    def validate(self) -> bool:
        """Validate that required columns exist."""
        columns = self._get_columns()

        if self.text_column not in columns:
            raise ValueError(f"Text column '{self.text_column}' not found in data. Available columns: {columns}")

        if self.id_column and self.id_column not in columns:
            raise ValueError(f"ID column '{self.id_column}' not found in data. Available columns: {columns}")

        for col in self.additional_columns:
            if col not in columns:
                raise ValueError(f"Additional column '{col}' not found in data. Available columns: {columns}")

        return True

    def _get_columns(self) -> list[str]:
        """Get column names from the data."""
        if isinstance(self.data, pd.DataFrame):
            return list(self.data.columns)
        elif isinstance(self.data, pa.Table):
            return self.data.column_names
        else:
            raise TypeError(f"Unsupported data type: {type(self.data)}")

    def to_pyarrow(self) -> pa.Table:
        """Convert data to PyArrow table."""
        if isinstance(self.data, pa.Table):
            return self.data
        elif isinstance(self.data, pd.DataFrame):
            return pa.Table.from_pandas(self.data)
        else:
            raise ValueError(f"Cannot convert {type(self.data)} to PyArrow table")

    def to_pandas(self) -> pd.DataFrame:
        """Convert data to Pandas DataFrame."""
        if isinstance(self.data, pd.DataFrame):
            return self.data
        elif isinstance(self.data, pa.Table):
            return self.data.to_pandas()
        else:
            raise ValueError(f"Cannot convert {type(self.data)} to Pandas DataFrame")

    @property
    def num_items(self) -> int:
        """Get the number of documents in this batch."""
        return len(self.data)

    def get_text_series(self) -> pd.Series:
        """Get the text column as a pandas Series."""
        df = self.to_pandas()
        return df[self.text_column]

    def get_id_series(self) -> pd.Series | None:
        """Get the ID column as a pandas Series if it exists."""
        if not self.id_column:
            return None
        df = self.to_pandas()
        return df[self.id_column]


@dataclass
class ImageObject:
    """Represents a single image with metadata."""

    image_path: str
    image_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageBatch(Task[list[ImageObject]]):
    """Task for processing batches of images.

    Images are stored as a list of ImageObject instances, each containing
    the path to the image and associated metadata.
    """

    def validate(self) -> bool:
        """Validate that all image objects are properly formed."""
        if not isinstance(self.data, list):
            raise TypeError(f"ImageBatch data must be a list, got {type(self.data)}")

        for i, img in enumerate(self.data):
            if not isinstance(img, ImageObject):
                raise TypeError(f"Item {i} is not an ImageObject: {type(img)}")

        return True

    @property
    def num_items(self) -> int:
        """Get the number of images in this batch."""
        return len(self.data)

    def get_image_paths(self) -> list[str]:
        """Get list of all image paths in this batch."""
        return [img.image_path for img in self.data]

    def get_image_ids(self) -> list[str]:
        """Get list of all image IDs in this batch."""
        return [img.image_id for img in self.data]
