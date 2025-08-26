# Changelog

## [Improvements] - 2025-08-26

### Added
- **Setup Script** (`setup.sh`): Automated installation script for easier onboarding
  - Checks Python version requirements
  - Creates and activates virtual environment
  - Installs all dependencies including spaCy models
  - Creates `.env` file template
  - Verifies installation

- **Missing Method**: Added `anonymize_transactions()` to `AdaptiveAnonymizer` class
  - Processes lists of transaction dictionaries
  - Anonymizes sensitive fields (concept, description, notes)

- **Test Script** (`test_lightrag_api.py`): Comprehensive testing for LightRAG functionality
  - Tests OpenAI API connectivity
  - Verifies LightRAG initialization
  - Tests query execution with real API keys

### Fixed
- **Dependencies**: Updated `requirements.txt` with missing packages
  - Added `langgraph>=0.6.0`
  - Added `langchain-openai>=0.3.0`
  - Enabled `tavily-python>=0.3.0` for web search

- **Import Errors**: Fixed module references in demo scripts
  - Updated `demo_queries.py` to use correct module paths
  - Fixed initialization parameters for LightRAG

- **Data Extraction**: Enhanced `BBVAExtractor` for better compatibility
  - Added `status` field to extraction results
  - Added `concept` field mapping for backward compatibility
  - Improved metadata structure

### Changed
- **Error Handling**: Improved error messages and fallback behavior
- **Documentation**: Updated inline documentation for clarity

### Testing Results
- ✅ Data extraction: Successfully processes CSV files
- ✅ API Integration: OpenAI API connection verified
- ✅ LightRAG: Knowledge graph loaded (11 files)
- ⚠️ Query execution: Async context issues need resolution

### Known Issues
- LightRAG query execution has async context manager issues
- Some advanced categorization features require additional configuration
- PDF extraction not yet implemented (placeholder exists)

## Branch Structure
Created feature branches for organized development:
- `fix/dependencies` - Dependency updates
- `feat/setup-script` - Automated setup
- `fix/anonymizer-methods` - Missing method implementations
- `docs/improvements` - Documentation updates