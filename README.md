# Radiology AI Agent

A Python AI agent that uses OpenAI (ChatGPT) API to transform de-identified radiology narratives into structured data, with a confirmation step before extraction.

## Features

- **Natural Language Query Processing**: Parse user questions like "How many patients had left kidney stones > 1 cm?"
- **Intent → Plan → Confirm Workflow**: Always confirm extraction plans before processing
- **Dual Extraction Methods**: Regex baseline + LLM extraction with OpenAI API
- **Structured Data Output**: Patient-imaging level data with stone status, sizes, and measurements
- **Interactive Mode**: Query and visualize data with natural language
- **Safety Features**: PHI detection and redaction before API calls
- **Caching**: SQLite cache for API responses to reduce costs
- **Visualizations**: Matplotlib and Plotly charts for data analysis

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd medical-ai-agent
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp env_example.txt .env
   # Edit .env and add your OpenAI API key
   ```

## Quick Start

1. **Create sample data** (for testing):
   ```bash
   python create_sample_data.py
   ```

2. **Run with sample data**:
   ```bash
   # Interactive mode
   python main.py sample_radiology_data.xlsx
   
   # Specific query
   python main.py sample_radiology_data.xlsx --query "How many patients had left kidney stones > 1 cm?"
   
   # Dry run (regex only, no API calls)
   python main.py sample_radiology_data.xlsx --dry-run --limit 10
   ```

## Data Format

### Input Excel File
Required columns:
- `recordid`: Patient record ID
- `surg_date`: Surgery date (optional)
- `imaging_date`: Imaging study date
- `narrative`: Radiology report text

### Output Structured Data
Columns (patient-imaging grain):
- `recordid`, `imaging_date`
- `right_stone` (present/absent/unclear), `right_stone_size_cm` (float|NULL), `right_kidney_size_cm` (string|NULL)
- `left_stone` (present/absent/unclear), `left_stone_size_cm` (float|NULL), `left_kidney_size_cm` (string|NULL)
- `bladder_volume_ml` (float|NULL)
- `history_summary` (string|NULL)
- `matched_reason` (brief string explaining filter matches)

## Usage Examples

### Command Line Interface

```bash
# Interactive mode - ask questions naturally
python main.py data.xlsx

# Specific query
python main.py data.xlsx --query "Show me all patients with bilateral stones"

# Dry run for testing (no API calls)
python main.py data.xlsx --dry-run --limit 5

# Process specific number of records
python main.py data.xlsx --limit 100
```

### Natural Language Queries

The agent understands queries like:
- "How many patients had left kidney stones > 1 cm?"
- "Show me all patients with bilateral stones between 2020-2022"
- "Count patients with bladder volume > 200 ml"
- "Plot stone distribution by side"
- "Find patients with right kidney stones larger than 1.5 cm"

### Interactive Mode Features

1. **Plan Confirmation**: Always shows extraction plan before processing
2. **Filter Editing**: Modify filters before execution
3. **Results Options**: Show table, download CSV, create plots
4. **Visualization**: Generate charts and interactive dashboards

## Architecture

### Core Components

- `src/config.py`: Configuration and environment settings
- `src/safety.py`: PHI detection and data safety utilities
- `src/io_utils.py`: Excel loading and data processing
- `src/regex_baseline.py`: Rule-based extraction baseline
- `src/llm_schema.py`: Pydantic models for JSON validation
- `src/llm_extractor.py`: OpenAI API integration with caching
- `src/intent_parser.py`: Natural language query parsing
- `src/confirm_flow.py`: User confirmation workflow
- `src/query_tools.py`: Data filtering and analysis
- `src/viz.py`: Visualization and plotting
- `src/orchestrator.py`: Main pipeline orchestration

### LLM JSON Schema

The agent uses structured JSON extraction with Pydantic validation:

```json
{
  "right": {
    "stone_status": "present|absent|unclear",
    "stone_size_cm": number|null,
    "kidney_size_cm": "L x W x AP cm"|null
  },
  "left": {
    "stone_status": "present|absent|unclear", 
    "stone_size_cm": number|null,
    "kidney_size_cm": "L x W x AP cm"|null
  },
  "bladder": {
    "volume_ml": number|null,
    "wall": "normal|abnormal"|null
  },
  "history_summary": string|null,
  "key_sentences": [string]|null
}
```

## Safety Features

- **PHI Detection**: Automatically detects and redacts potential PHI
- **Conservative Extraction**: No guessing - unclear findings marked as "unclear"
- **Negation Handling**: Properly handles "no stones" vs "stones present"
- **API Safety**: Never echoes original PHI back to user

## Output Files

The agent creates several output files in the `out/` directory:

- `structured.parquet`: Complete structured dataset
- `filtered.csv`: Query-specific filtered results
- `out/plots/`: Visualization files (PNG, HTML)
- `out/cache/`: SQLite cache for API responses
- `radiology_agent.log`: Processing logs

## Configuration

Environment variables (`.env` file):
```bash
OPENAI_API_KEY=your_openai_api_key_here
MODEL_NAME=gpt-4o-mini
MAX_RETRIES=3
BATCH_SIZE=10
TIMEOUT_SECONDS=30
```

## Cost Management

- **Caching**: SQLite cache prevents duplicate API calls
- **Token Tracking**: Logs token usage and estimated costs
- **Model Selection**: Supports gpt-4o-mini (cheaper) and gpt-4o
- **Dry Run Mode**: Test with regex only, no API costs

## Error Handling

- **Retry Logic**: Exponential backoff for API failures
- **Validation**: Pydantic models ensure data quality
- **Fallback**: Regex baseline when LLM extraction fails
- **Logging**: Comprehensive logging for debugging

## Development

### Running Tests
```bash
# Create sample data
python create_sample_data.py

# Test with dry run
python main.py sample_radiology_data.xlsx --dry-run --limit 5

# Test specific query
python main.py sample_radiology_data.xlsx --query "count stones" --dry-run
```

### Adding New Features

1. **New Extraction Fields**: Update `llm_schema.py` and `regex_baseline.py`
2. **New Query Types**: Extend `intent_parser.py` patterns
3. **New Visualizations**: Add methods to `viz.py`
4. **New Filters**: Update `query_tools.py` filter logic

## License

This project is licensed under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions, please open an issue on the repository.



