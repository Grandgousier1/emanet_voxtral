# CODE QUALITY ANALYSIS REPORT
==================================================

## PROJECT OVERVIEW
Total Lines of Code: 8,248
Total Functions: 333
Total Classes: 55
Total Complexity: 1,413
Average Maintainability Index: 8.8
Average Documentation Coverage: 14.8%
Files Analyzed: 35
Files with Issues: 20

## MOST COMPLEX FILES
- validator.py: 129 complexity
- parallel_processor.py: 85 complexity
- code_quality_analyzer.py: 75 complexity
- utils/b200_optimizer.py: 74 complexity
- utils/model_utils.py: 70 complexity

## LARGEST FILES
- validator.py: 591 lines
- parallel_processor.py: 503 lines
- code_quality_analyzer.py: 423 lines
- tests/test_end_to_end.py: 390 lines
- cli_feedback.py: 366 lines

## RECOMMENDATIONS
### HIGH PRIORITY
- **validator.py**: High cyclomatic complexity (129)
  *Recommendation*: Consider breaking down complex functions into smaller, focused functions
- **validator.py**: Low maintainability index (0.0)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **test_improvements.py**: Low maintainability index (19.9)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **test_parallel_processor.py**: Low maintainability index (44.7)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **config.py**: Low maintainability index (0.5)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **monitor.py**: High cyclomatic complexity (56)
  *Recommendation*: Consider breaking down complex functions into smaller, focused functions
- **monitor.py**: Low maintainability index (0.0)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **cli_feedback.py**: High cyclomatic complexity (64)
  *Recommendation*: Consider breaking down complex functions into smaller, focused functions
- **cli_feedback.py**: Low maintainability index (0.0)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **test_timing_sync.py**: Low maintainability index (0.2)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **test_main.py**: Low maintainability index (38.8)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **test_complete.py**: Low maintainability index (12.2)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **constants.py**: Low maintainability index (0.0)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **main.py**: Low maintainability index (1.1)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **voxtral_prompts.py**: Low maintainability index (0.0)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **parallel_processor.py**: High cyclomatic complexity (85)
  *Recommendation*: Consider breaking down complex functions into smaller, focused functions
- **parallel_processor.py**: Low maintainability index (0.0)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **benchmark.py**: Low maintainability index (0.0)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **code_quality_analyzer.py**: High cyclomatic complexity (75)
  *Recommendation*: Consider breaking down complex functions into smaller, focused functions
- **code_quality_analyzer.py**: Low maintainability index (0.0)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **utils/gpu_utils.py**: Low maintainability index (49.9)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **utils/security_utils.py**: Low maintainability index (19.7)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **utils/antibot_utils.py**: Low maintainability index (13.1)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **utils/audio_cache.py**: Low maintainability index (0.5)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **utils/srt_utils.py**: Low maintainability index (24.3)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **utils/processing_utils.py**: High cyclomatic complexity (69)
  *Recommendation*: Consider breaking down complex functions into smaller, focused functions
- **utils/processing_utils.py**: Low maintainability index (0.0)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **utils/validation_utils.py**: Low maintainability index (36.5)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **utils/error_messages.py**: Low maintainability index (23.4)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **utils/memory_manager.py**: Low maintainability index (5.9)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **utils/tensor_validation.py**: Low maintainability index (7.5)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **utils/audio_utils.py**: High cyclomatic complexity (53)
  *Recommendation*: Consider breaking down complex functions into smaller, focused functions
- **utils/audio_utils.py**: Low maintainability index (0.0)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **utils/reproducibility.py**: Low maintainability index (9.0)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **utils/translation_quality.py**: Low maintainability index (1.2)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **utils/b200_optimizer.py**: High cyclomatic complexity (74)
  *Recommendation*: Consider breaking down complex functions into smaller, focused functions
- **utils/b200_optimizer.py**: Low maintainability index (0.0)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **utils/model_utils.py**: High cyclomatic complexity (70)
  *Recommendation*: Consider breaking down complex functions into smaller, focused functions
- **utils/model_utils.py**: Low maintainability index (0.0)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **utils/performance_profiler.py**: High cyclomatic complexity (61)
  *Recommendation*: Consider breaking down complex functions into smaller, focused functions
- **utils/performance_profiler.py**: Low maintainability index (0.0)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **tests/test_ml_validation.py**: Low maintainability index (0.4)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **tests/test_edge_cases.py**: Low maintainability index (0.0)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **tests/test_integration_modules.py**: Low maintainability index (0.0)
  *Recommendation*: Refactor to reduce complexity and improve code structure
- **tests/test_end_to_end.py**: High cyclomatic complexity (51)
  *Recommendation*: Consider breaking down complex functions into smaller, focused functions
- **tests/test_end_to_end.py**: Low maintainability index (0.0)
  *Recommendation*: Refactor to reduce complexity and improve code structure

### MEDIUM PRIORITY
- **test_timing_sync.py**: Low documentation coverage (9.5%)
  *Recommendation*: Add docstrings and comments to improve code documentation
- **main.py**: Low documentation coverage (8.3%)
  *Recommendation*: Add docstrings and comments to improve code documentation
- **voxtral_prompts.py**: Low documentation coverage (0.0%)
  *Recommendation*: Add docstrings and comments to improve code documentation
- **parallel_processor.py**: Very long function (178 lines)
  *Recommendation*: Break down large functions into smaller, more manageable pieces
- **benchmark.py**: Low documentation coverage (8.3%)
  *Recommendation*: Add docstrings and comments to improve code documentation
- **utils/gpu_utils.py**: Low documentation coverage (5.0%)
  *Recommendation*: Add docstrings and comments to improve code documentation
- **utils/processing_utils.py**: Very long function (138 lines)
  *Recommendation*: Break down large functions into smaller, more manageable pieces
- **utils/error_messages.py**: Low documentation coverage (8.7%)
  *Recommendation*: Add docstrings and comments to improve code documentation
- **utils/tensor_validation.py**: Low documentation coverage (8.8%)
  *Recommendation*: Add docstrings and comments to improve code documentation
- **utils/audio_utils.py**: Very long function (159 lines)
  *Recommendation*: Break down large functions into smaller, more manageable pieces
- **utils/audio_utils.py**: Low documentation coverage (9.6%)
  *Recommendation*: Add docstrings and comments to improve code documentation
- **utils/model_utils.py**: Very long function (152 lines)
  *Recommendation*: Break down large functions into smaller, more manageable pieces
- **Multiple files**: Duplicate function: main()
  *Recommendation*: Extract common functionality into shared utilities or base classes
- **Multiple files**: Duplicate function: __init__(self)
  *Recommendation*: Extract common functionality into shared utilities or base classes
- **Multiple files**: Duplicate function: detect_hardware()
  *Recommendation*: Extract common functionality into shared utilities or base classes
- **Multiple files**: Duplicate function: __init__(self, feedback)
  *Recommendation*: Extract common functionality into shared utilities or base classes
- **Multiple files**: Duplicate function: get_stats(self)
  *Recommendation*: Extract common functionality into shared utilities or base classes
- **Multiple files**: Duplicate function: wrapper()
  *Recommendation*: Extract common functionality into shared utilities or base classes
- **Multiple files**: Duplicate function: setUp(self)
  *Recommendation*: Extract common functionality into shared utilities or base classes
- **Multiple files**: Duplicate function: tearDown(self)
  *Recommendation*: Extract common functionality into shared utilities or base classes
- **Multiple files**: Duplicate code block (hash: 1697)
  *Recommendation*: Extract common functionality into shared utilities or base classes
- **Multiple files**: Duplicate code block (hash: 2835)
  *Recommendation*: Extract common functionality into shared utilities or base classes
- **Multiple files**: Duplicate code block (hash: 7365)
  *Recommendation*: Extract common functionality into shared utilities or base classes
- **Multiple files**: Duplicate code block (hash: 2323)
  *Recommendation*: Extract common functionality into shared utilities or base classes
- **Multiple files**: Duplicate code block (hash: 532)
  *Recommendation*: Extract common functionality into shared utilities or base classes
- **Multiple files**: Duplicate code block (hash: 2310)
  *Recommendation*: Extract common functionality into shared utilities or base classes
- **Multiple files**: Duplicate code block (hash: 4960)
  *Recommendation*: Extract common functionality into shared utilities or base classes
- **Multiple files**: Duplicate code block (hash: 8645)
  *Recommendation*: Extract common functionality into shared utilities or base classes
- **Multiple files**: Duplicate code block (hash: 626)
  *Recommendation*: Extract common functionality into shared utilities or base classes
- **Multiple files**: Duplicate code block (hash: 6395)
  *Recommendation*: Extract common functionality into shared utilities or base classes
- **Multiple files**: Duplicate code block (hash: 3744)
  *Recommendation*: Extract common functionality into shared utilities or base classes
- **Multiple files**: Duplicate code block (hash: 3847)
  *Recommendation*: Extract common functionality into shared utilities or base classes
- **Multiple files**: Duplicate code block (hash: 4717)
  *Recommendation*: Extract common functionality into shared utilities or base classes
- **Multiple files**: Duplicate code block (hash: 5952)
  *Recommendation*: Extract common functionality into shared utilities or base classes
- **Multiple files**: Duplicate code block (hash: 2630)
  *Recommendation*: Extract common functionality into shared utilities or base classes

## CODE DUPLICATIONS
### Duplicate function: main()
- validator.py
- main.py
- benchmark.py
- code_quality_analyzer.py

### Duplicate function: __init__(self)
- validator.py
- parallel_processor.py
- parallel_processor.py
- code_quality_analyzer.py
- code_quality_analyzer.py
- utils/translation_quality.py
- utils/performance_profiler.py

### Duplicate function: detect_hardware()
- config.py
- tests/test_end_to_end.py

### Duplicate function: __init__(self, feedback)
- cli_feedback.py
- utils/security_utils.py
- utils/error_messages.py
- utils/memory_manager.py
- utils/model_utils.py

### Duplicate function: get_stats(self)
- utils/audio_cache.py
- utils/memory_manager.py

### Duplicate function: wrapper()
- utils/b200_optimizer.py
- utils/performance_profiler.py

### Duplicate function: setUp(self)
- tests/test_ml_validation.py
- tests/test_ml_validation.py
- tests/test_edge_cases.py
- tests/test_integration_modules.py
- tests/test_integration_modules.py
- tests/test_end_to_end.py
- tests/test_end_to_end.py

### Duplicate function: tearDown(self)
- tests/test_integration_modules.py
- tests/test_integration_modules.py
- tests/test_end_to_end.py
- tests/test_end_to_end.py

### Duplicate code block (hash: 1697)
- validator.py
- validator.py

### Duplicate code block (hash: 2835)
- main.py
- utils/processing_utils.py

### Duplicate code block (hash: 7365)
- main.py
- parallel_processor.py
- utils/processing_utils.py
- utils/audio_utils.py

### Duplicate code block (hash: 2323)
- benchmark.py
- benchmark.py
- utils/performance_profiler.py
- utils/performance_profiler.py

### Duplicate code block (hash: 532)
- benchmark.py
- tests/test_edge_cases.py

### Duplicate code block (hash: 2310)
- utils/audio_cache.py
- utils/audio_cache.py

### Duplicate code block (hash: 4960)
- utils/audio_cache.py
- utils/audio_cache.py

### Duplicate code block (hash: 8645)
- utils/audio_utils.py
- utils/audio_utils.py

### Duplicate code block (hash: 626)
- utils/reproducibility.py
- utils/model_utils.py
- utils/model_utils.py
- utils/model_utils.py

### Duplicate code block (hash: 6395)
- utils/reproducibility.py
- utils/model_utils.py
- utils/model_utils.py
- utils/model_utils.py

### Duplicate code block (hash: 3744)
- utils/b200_optimizer.py
- utils/model_utils.py

### Duplicate code block (hash: 3847)
- utils/model_utils.py
- tests/test_edge_cases.py

### Duplicate code block (hash: 4717)
- tests/test_ml_validation.py
- tests/test_edge_cases.py
- tests/test_integration_modules.py
- tests/test_end_to_end.py

### Duplicate code block (hash: 5952)
- tests/test_edge_cases.py
- tests/test_integration_modules.py

### Duplicate code block (hash: 2630)
- tests/test_integration_modules.py
- tests/test_end_to_end.py
- tests/test_end_to_end.py
