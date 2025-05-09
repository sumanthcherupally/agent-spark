# SWE-bench Model

This repository contains code for training and evaluating models on the SWE-bench dataset, which tests AI systems' ability to solve real-world software engineering tasks.

## Project Structure

```
.
├── data/                   # Data processing and storage
├── models/                 # Model implementations
├── training/              # Training scripts and utilities
├── evaluation/            # Evaluation scripts and metrics
├── utils/                 # Utility functions
├── configs/               # Configuration files
└── tests/                 # Test files
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

### Training

```bash
python training/train.py --config configs/training_config.yaml
```

### Evaluation

```bash
python evaluation/evaluate.py --model_path path/to/model --test_data path/to/test_data
```

### Inference

```bash
python models/predict.py --model_path path/to/model --input_file path/to/input
```

## Development

- Run tests: `pytest tests/`
- Format code: `black .`
- Sort imports: `isort .`
- Lint code: `flake8`

## License

MIT License 