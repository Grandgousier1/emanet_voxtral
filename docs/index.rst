Emanet Voxtral Documentation
============================

**Emanet Voxtral v3.0** - Production-Ready AI Subtitle Generator optimized for B200 GPUs.

.. image:: https://img.shields.io/badge/python-3.11%2B-blue.svg
   :alt: Python Version
   :target: https://python.org

.. image:: https://img.shields.io/badge/GPU-B200%20Optimized-green.svg
   :alt: B200 Optimized

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :alt: Code Style
   :target: https://github.com/psf/black

Overview
--------

Emanet Voxtral is a sophisticated AI-powered subtitle generation system designed for high-performance
audio transcription and translation. The system leverages state-of-the-art models (Voxtral Small/Mini)
and is specifically optimized for NVIDIA B200 GPUs with 180GB VRAM.

Key Features
~~~~~~~~~~~~

- 🚀 **B200 GPU Optimization** - Specialized batching and memory management
- 🛡️ **Error Boundary System** - Automatic recovery and graceful degradation  
- 📊 **Type-Safe Architecture** - Complete TypedDict domain models
- 🧪 **Production Testing** - 80%+ test coverage with unit & integration tests
- ⚡ **High Performance** - Optimized hot paths with O(n log n) algorithms
- 🔧 **Modular Services** - Clean architecture with dependency injection

Architecture
------------

The system follows a clean architecture pattern with clear separation of concerns:

.. code-block::

    voxtral/
    ├── main.py              # Entry point with refactored functions
    ├── domain_models.py     # Type-safe data structures  
    ├── error_boundary.py    # Unified error handling
    ├── parallel_processor.py # Optimized GPU processing
    ├── services/            # Business logic services
    │   ├── processing_service.py
    │   └── validation_service.py
    └── utils/               # Utility modules

Quick Start
-----------

Installation::

    # Clone and setup
    git clone <repository>
    cd emanet_voxtral
    make setup

Basic Usage::

    # Single video processing
    python main.py --url "https://youtube.com/watch?v=..." --output "output.srt"

    # Batch processing
    python main.py --batch-list videos.txt --output-dir ./subtitles/

    # Validation
    python main.py --validate-only

API Reference
=============

.. toctree::
   :maxdepth: 2
   :caption: Core Modules

   api/main
   api/domain_models
   api/error_boundary
   api/parallel_processor

.. toctree::
   :maxdepth: 2
   :caption: Services

   api/services

.. toctree::
   :maxdepth: 2  
   :caption: Utilities

   api/utils

Development Guide
=================

.. toctree::
   :maxdepth: 2
   :caption: Development

   development/setup
   development/testing
   development/contributing
   development/architecture

Performance & Optimization
===========================

.. toctree::
   :maxdepth: 2
   :caption: Performance

   performance/b200_optimization
   performance/benchmarks
   performance/profiling

Deployment
==========

.. toctree::
   :maxdepth: 2
   :caption: Deployment

   deployment/runpod_b200
   deployment/monitoring
   deployment/troubleshooting

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`