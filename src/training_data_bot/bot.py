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
        
    async def load_documents(
            self,
            sources: List[Union[str, Path, List[Union[str, Path]]]],
            doc_types: Optional[DocumentType] = None,
            **kwargs
    ) -> List[Document]:
        """
        Load documents from specified sources

        Args:
            sources: Single Source or list of sources (file paths, URLs, directories, etc.)
            doc_types (Optional[DocumentType]): Optional document type hint
            **kwargs: Additional parameters for loading
        Returns:
            List[Document]: List of loaded Document objects
        """
        
        with LogContext("document_loading", sources=str(sources)):
            try:
                # Ensuring that sources is a list
                if isinstance(sources, (str, Path)):
                    sources = [sources]
                
                # Check if sources is a directory
                documents = []
                for source in sources:
                    source_path = Path(source)

                    if source_path.is_dir():
                        # If it's a directory, load all documents within it
                        dir_docs = await self.loader.load_from_directory(source, doc_types=doc_types, **kwargs)
                        documents.extend(dir_docs)
                    else:
                        # If it's a file, load it directly
                        doc = await self.loader.load_single(source, doc_types=doc_types, **kwargs)
                        documents.append(doc)
                
                # Store documents in memory and database
                for doc in documents:
                    self.documents[doc.id] = doc
                    # await self.db_manager.save_document(doc)
                
                self.logger.info(f"Loaded {len(documents)} documents from sources.")

                return documents
            except Exception as e:
                self.logger.error(f"Error loading documents: {e}")
                return []
            
    async def process_documents(
            self,
            documents: Optional[List[Document]] = None,
            task_types: Optional[List[TaskType]] = None,
            quality_filter: bool = True,
            **kwargs
    ) -> Dataset:
        """
        Process documents to generate training data

        Args:
            document_ids (List[UUID]): List of document IDs to process (default value: all loaded documents)
            task_type (TaskType): Type of task to perform (e.g., QA, Classification)
            preprocess (bool): Whether to preprocess text before processing
            **kwargs: Additional parameters for processing
        Returns:
            Dataset: Generated Dataset containing the processed training data
        """
        with LogContext("document_processing"):
            try:
                # Use all loaded documents if none specified
                if documents is None:
                    documents = list(self.documents.values())
                if not documents: # If the list is empty
                    raise TrainingDataBotError("No documents available for processing.", context={"documents": documents})

                # Use default task types if none specified
                if task_types is None:
                    task_types = [TaskType.QA_GENERATION, TaskType.CLASSIFICATION, TaskType.SUMMARIZATION]
            
                # Create a new processing job
                job = ProcessingJob(
                    name=f'Process {len(documents)} documents',
                    job_type='document_processing',
                    total_items=len(documents)*len(task_types),
                    
                    input_data={
                        "document_count": len(documents),
                        "task_types": [t.value for t in task_types],
                        "quality_filter": quality_filter
                    }
                )

                self.jobs[job.id] = job
                
                job.status = ProcessingStatus.PROCESSING

                # Process each document
                all_examples = []

                for doc in documents:
                    # Preprocess document (chunking, cleaning, etc.)
                    chunks = await self.preprocessor.process_document(doc)
                    
                    for task_type in task_types:
                        # Generate training examples for each chunk
                        for chunk in chunks:
                            try:
                                self.logger.info(f"Processing chunk for task {task_type.value}")
                                # Execute task
                                result = await self.task_manager.execute_task(
                                    task_type=task_type,
                                    input_text=chunk,
                                    client=self.ai_client
                                )
                               
                               # Create training example
                                example = TrainingExample(
                                    input_text = chunk.content,
                                    output_text = result.output,
                                    task_type = task_type,
                                    source_document_id = chunk.id,
                                    template_id = result.template_id,
                                    quality_scores = result.quality_scores
                                )

                                # Apply quality filtering if enabled

                                if quality_filter:
                                    # Evaluate quality
                                    quality_report = await self.evaluator.evaluate_example(example)

                                    if quality_report.passed:
                                        all_examples.append(example)
                                        example.quality_approved = True
                                    else:
                                        example.quality_approved = False
                                        self.logger.warning(f"Example filtered out by quality check, due to poor quality")
                                else:
                                    all_examples.append(example)
                                
                                job.processed_items += 1
                            
                            except Exception as e:
                                self.logger.error(f"Error processing documents: {e}")
                                job.failed_items += 1
                                continue
                #Create dataset
                dataset=Dataset(
                    name=f"Generated Dataset {len(self.datasets)+1}",
                    description=f"Dataset generated from {len(documents)} document",
                    examples=all_examples,
                )

                # Store dataset
                self.datasets[dataset.id]=dataset
                
                #Update job status
                job.status=ProcessingStatus.COMPLETED
                job.output_data={
                    "dataset_id":str(dataset.id),
                    "examples_generated":len(all_examples),
                    "quality_filtered":quality_filter,
                }
                self.logger.info(f"Processing completed, Generated{len(all_examples)}")
                return dataset
                        
            except Exception as e:
                if 'job' in locals():
                    job.status=ProcessingStatus.FAILED
                    job.error_message=str(e)
                self.logger.error(f"Document processing failed: {e}")
                raise
        