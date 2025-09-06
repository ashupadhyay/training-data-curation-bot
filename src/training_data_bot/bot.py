"""
Main Training Data Bot Class

This module contains the core TrainingDataBot class
This class orchestrates all functionality which includes
- Document loading
- Processing
- Quality Assessment
- Dataset export
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from uuid import UUID

from .core.config import settings
from .core.logging import get_logger, LogContext
from .core.exceptions import TrainingDataBotError, ConfigurationError
from .core.models import (
    Document, 
    Dataset, 
    TrainingExample,
    TaskType,
    DocumentType,
    ProcessingJob,
    ProcessingStatus,
    QualityReport,
    ExportFormat)

# Import modules
from .sources import UnifiedLoader
from .decodo import DecodoClient # WebScraping
from .ai import AIClient  # LLM interactions, AI Text Generation
from .tasks import TaskManager
from .preprocessing import TextPreprocessor
from .evaluation import QualityEvaluator
from .storage import DatasetExplorer, DatabaseManager

class TrainingDataBot:
    """
    Main Training Data Bot Class

    This class provides a high-level interface for 
    - Loading documents from various sources
    Preprocessing text with task templates
    - Quality assessment and filtering
    - Dataset creation and export
    """

    def __init__(
            self,
            config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Training Data Bot

        Args:
            config (Optional[Dict[str, Any]]): Optional configuration dictionary
        """
        self.logger = get_logger("TrainingDataBot")
        self.config = config or {}
        self._init_components()
        self.logger.info("TrainingDataBot initialized successfully.")

    def _init_components(self):
        """
        Initialize core components based on configuration
        """
        try:
            # Initialize Unified Loader for document loading
            self.loader = UnifiedLoader()

            # Initialize Decodo Client for web scraping
            self.decodo_client = DecodoClient()

            # Initialize AI Client for LLM interactions
            self.ai_client = AIClient()


            # Initialize Task Manager for handling various tasks
            self.task_manager = TaskManager()

            # Initialize Text Preprocessor for cleaning text
            self.preprocessor = TextPreprocessor()

            # Initialize Quality Evaluator for assessing data quality
            self.quality_evaluator = QualityEvaluator()

            # Initialize Dataset Explorer for dataset management
            self.dataset_explorer = DatasetExplorer()

            # Initialize Database Manager for persistent storage
            self.db_manager = DatabaseManager(db_url=settings.DATABASE_URL)


            # state(memoery boxes)
            self.documents: Dict[UUID, Document] = {}
            self.datasets: Dict[UUID, Dataset] = {}
            self.jobs: Dict[UUID, ProcessingJob] = {}

        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise ConfigurationError("Failed to initialize TrainingDataBot components.", 
                                     context = {"error": str(e)},
                                     cause=e
                                     )
        

        