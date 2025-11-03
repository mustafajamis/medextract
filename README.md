# MedExtract

MedExtract is a clinical datapoint extraction system that uses Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to extract specific information from medical reports.

## 🌟 New: Web Dashboard

MedExtract now includes a **user-friendly web dashboard** that makes AI-powered medical data extraction accessible to everyone—no coding required!

### Quick Start with Dashboard

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install Ollama and download a model:
   ```bash
   ollama pull llama3
   ```

3. Start the dashboard:
   ```bash
   python app.py
   ```

4. Open your browser to `http://localhost:5000`

5. Upload your CSV file, configure settings, and extract data!

📖 **For complete instructions, see [DASHBOARD_TUTORIAL.md](DASHBOARD_TUTORIAL.md)**

## Features

- Supports multiple LLM models (llama3, mistral)
- Implements RAG with various embedding models and retriever types
- Configurable processing options and evaluation metrics
- Supports few-shot learning and prompt engineering
- Benchmarking capabilities for model comparison

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/medextract.git
   cd medextract
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Place your input CSV file in the `data/input/` directory.

2. Modify the `config/config.yaml` file to suit your needs or use the default configuration.

3. Run the main script:
   ```
   python medextract.py --config config/config.yaml
   ```

4. Results will be saved in the `data/output/results/` directory, and figures in the `data/output/figures/` directory.

## Configuration

The `config/config.yaml` file contains all the configurable parameters for the pipeline. You can modify this file to change the behavior of the system, including:

- Input/output file paths
- Processing options
- RAG settings
- Model selection
- Prompt engineering options
- Evaluation settings

## Common Issues & Troubleshooting

### Dashboard Issues

**Problem: RAG showing as disabled in results even when enabled**
- **Solution**: This has been fixed in the latest version. Make sure you're using the updated code where `medextract.main()` accepts a `config_path` parameter.

**Problem: Uploaded files not being processed**
- **Solution**: The dashboard now correctly uses uploaded files. Your file is saved with a unique name in `data/input/` and automatically used for processing. You no longer need to manually place files in `data/input/input_reports.csv`.

**Problem: Results show all "NR" (Not Reported) values**
- **Cause**: This usually means RAG was disabled or the LLM couldn't read the reports properly.
- **Solution**: 
  1. Ensure RAG is enabled in the configuration page
  2. Check that Ollama is running: `ollama list` should show your model
  3. Verify your CSV has a "Report Text" column with actual medical text
  4. Try with a smaller batch size first (10-20 reports) to test

**Problem: Low accuracy results**
- **Solutions**:
  - Enable RAG for better context understanding
  - Use the reranker option for improved relevance
  - Enable few-shot learning
  - Try the ensemble retriever
  - Ensure ground truth labels are properly formatted

### General Issues

**Problem: Ollama not found**
- Install Ollama from [ollama.ai](https://ollama.ai)
- Run `ollama serve` to start the server
- Pull the model: `ollama pull llama3`

**Problem: Out of memory**
- Reduce batch size in configuration
- Disable RAG temporarily
- Use a lighter embedding model
- Close other applications

For more detailed troubleshooting, see [DASHBOARD_TUTORIAL.md](DASHBOARD_TUTORIAL.md).

## Docker

To run MedExtract using Docker:

1. Build the Docker image:
   ```
   docker build -t medextract .
   ```

2. Run the container:
   ```
   docker run -v $(pwd)/data:/app/data medextract
   ```

## Contributing

We welcome contributions to MedExtract! Please see our Contributing Guide for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

If you have any questions or feedback, please open an issue on this GitHub repository.
