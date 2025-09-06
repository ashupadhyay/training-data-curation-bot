"""
Training Data Curation Bot

We are using Decodo and Python for curation of training data for 
LLM fine-tuning
"""

__version__ = "0.1.0"
__author__ = "Aishwarya Upadhyay"
__description__ = "Enterprise-grade training data curation bot for LLM fine-tuning"

# Core imports for easy access
from .core.config import settings
from .core.logging import get_logger
from .core.exceptions import TrainingDataBotError

# Main bot class
from .bot import TrainingDataBot

# Key modules for external use
from .sources import(
    PDFLoader,   # worker reading PDF Files
    WebLoader,   # worker reading websites
    DocumentLoader, # Worker reading text files
    UnifiedLoader # # Boss who decides which worker to use
)

from .tasks import(
    QAGenerator,   # Worker making questions and answers
    ClassficationGenerator,   # Worker sorting things into categories
    SummarizationGenerator,   # Worker for summarization
    TaskTemplate    # Instruction sheet for workers
)

from .decodo import DecodoClient   # used for scraping internet
from .preprocessing import TextPreprocessor   # The text cleaner
from .evaluation import QualityEvaluator   # The quality checker
from .storage import DatasetExplorer   # The packager

__all__ = [
    # Core
    "TrainingDataBot",
    "settings",
    "get_logger",
    "TrainingDataBotError",

    # Sources
    'PDFLoader',
    "WebLoader",
    "DocumentLoader",
    "UnifiedLoader",

    # Tasks
    'QAGenerator',
    'ClassficationGenerator',
    'SummarizationGenerator',
    'TaskTemplate',

    # Services
    "DecodoClient",
    'TextPreprocessor',
    'QualityEvaluator',
    'DatasetExplorer'
]