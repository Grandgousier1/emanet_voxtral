domain_models module
===================

.. automodule:: domain_models
   :members:
   :undoc-members:
   :show-inheritance:

Core Types
----------

Audio Domain Models
~~~~~~~~~~~~~~~~~~~

.. autoclass:: AudioSegment
   :members:

.. autoclass:: ProcessingResult
   :members:

Configuration Models
~~~~~~~~~~~~~~~~~~~~ 

.. autoclass:: ModelConfig
   :members:

.. autoclass:: ProcessingConfig
   :members:

.. autoclass:: HardwareInfo
   :members:

Error Models
~~~~~~~~~~~~

.. autoclass:: ErrorSeverity
   :members:

.. autoclass:: ErrorContext
   :members:

Batch Processing Models
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: BatchMetrics
   :members:

.. autoclass:: BatchStatus
   :members:

.. autoclass:: ProcessingBatch
   :members:

Protocols
~~~~~~~~~

.. autoclass:: AudioProcessor
   :members:

.. autoclass:: ModelManager
   :members:

.. autoclass:: FeedbackProvider
   :members:

Validation Functions
~~~~~~~~~~~~~~~~~~~~ 

.. autofunction:: validate_audio_segment
.. autofunction:: validate_processing_result