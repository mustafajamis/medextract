# MedExtract Dashboard Tutorial

## Complete Guide to Using the MedExtract Web Dashboard

### Table of Contents
1. [Introduction](#introduction)
2. [What is MedExtract Dashboard?](#what-is-medextract-dashboard)
3. [System Requirements](#system-requirements)
4. [Installation Guide](#installation-guide)
5. [Quick Start](#quick-start)
6. [Detailed User Guide](#detailed-user-guide)
7. [Configuration Options](#configuration-options)
8. [Understanding Results](#understanding-results)
9. [Advanced Features](#advanced-features)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)
12. [Technical Architecture](#technical-architecture)

---

## Introduction

This tutorial provides comprehensive documentation for the MedExtract Web Dashboard, a user-friendly interface that transforms the powerful MedExtract clinical datapoint extraction engine into an accessible tool for biologists, researchers, and health scientists.

**Problem Solved:** MedExtract was originally a complex Python tool requiring technical expertise. This dashboard bridges the gap between the sophisticated backend (`medextract.py`) and non-technical users, making AI-powered medical data extraction accessible to everyone.

---

## What is MedExtract Dashboard?

The MedExtract Dashboard is a web-based application that provides:

- **Intuitive Interface:** Upload CSV files through a simple drag-and-drop interface
- **Visual Configuration:** Configure AI models and parameters without editing code
- **Real-time Processing:** Watch your data being processed with live progress updates
- **Rich Visualizations:** View confusion matrices, performance metrics, and extracted data
- **Easy Export:** Download results in standard CSV format

### Key Benefits

1. **No Coding Required:** Complete workflow through web interface
2. **Accessible to Non-Experts:** Designed for medical researchers without AI expertise
3. **Flexible Configuration:** Customize models, RAG settings, and processing parameters
4. **Comprehensive Results:** View accuracy metrics, visualizations, and raw data
5. **Production-Ready:** Built with Flask for reliability and scalability

---

## System Requirements

### Hardware Requirements
- **Minimum:**
  - 8 GB RAM
  - 20 GB free disk space
  - Dual-core processor
  
- **Recommended:**
  - 16+ GB RAM
  - 50+ GB free disk space
  - Quad-core processor (or better)
  - GPU (optional, for faster processing with CUDA-enabled models)

### Software Requirements
- **Operating System:** Linux, macOS, or Windows 10+
- **Python:** 3.7 or higher (3.9+ recommended)
- **Ollama:** Required for LLM models (download from [ollama.ai](https://ollama.ai))
- **Web Browser:** Chrome, Firefox, Safari, or Edge (latest versions)

---

## Installation Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/mustafajamis/medextract.git
cd medextract
```

### Step 2: Create Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- Flask (web framework)
- pandas (data manipulation)
- scikit-learn (metrics)
- LangChain (RAG framework)
- sentence-transformers (embeddings)
- ollama (LLM interface)
- And other required packages

### Step 4: Install Ollama and Download Models

1. **Install Ollama:**
   - Visit [ollama.ai](https://ollama.ai)
   - Download and install for your operating system
   
2. **Download Language Models:**
   ```bash
   # Install Llama 3 (recommended)
   ollama pull llama3
   
   # Optional: Install other models
   ollama pull mistral
   ollama pull llama2
   ```

3. **Verify Installation:**
   ```bash
   ollama list
   ```
   You should see the downloaded models listed.

### Step 5: Verify Installation

Test that MedExtract is properly installed:

```bash
python medextract.py --help
```

---

## Quick Start

### Starting the Dashboard

1. **Activate Virtual Environment** (if not already activated):
   ```bash
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate  # Windows
   ```

2. **Start the Dashboard:**
   ```bash
   python app.py
   ```

3. **Access the Dashboard:**
   - Open your web browser
   - Navigate to: `http://localhost:5000`
   - You should see the MedExtract Dashboard homepage

### Your First Extraction

1. **Upload a CSV File:**
   - Click "Get Started" or "Upload" in the navigation
   - Select a CSV file containing medical reports
   - CSV must have a column named "Report Text" or similar
   - Click "Upload File"

2. **Configure Parameters:**
   - Select your preferred LLM model (Llama 3 recommended)
   - Enable/disable RAG (Retrieval-Augmented Generation)
   - Adjust batch size (start with 10 for testing)
   - Click "Save Configuration & Start Processing"

3. **Monitor Processing:**
   - Watch the progress bar as reports are processed
   - Processing time varies based on:
     - Number of reports
     - Model complexity
     - Hardware capabilities
   - Wait for completion message

4. **View Results:**
   - Click "View Results" when processing completes
   - Review accuracy metrics
   - Examine confusion matrices
   - Preview extracted data
   - Download complete results as CSV

---

## Detailed User Guide

### Home Page

The home page provides:
- Overview of MedExtract capabilities
- Quick access to upload functionality
- Key features list
- Navigation to other sections

### Upload Page

**Accepted File Format:**
- CSV files (.csv)
- Maximum size: 16 MB
- Required columns:
  - A column with medical report text (e.g., "Report Text")
  - Optional: Ground truth labels (e.g., "BTFU Score (Updated)")
  
**Example CSV Structure:**
```csv
Report_ID,Report Text,BTFU Score (Updated)
001,"Patient presents with mild symptoms...",1
002,"Severe condition requiring hospitalization...",3
003,"No significant findings...",0
```

**Tips:**
- Ensure your CSV is properly formatted
- Remove any special characters that might break parsing
- Include ground truth labels for evaluation metrics

### Configure Page

#### Processing Options

**Batch Size:**
- Number of reports to process
- Start small (10-50) for testing
- Increase for production runs
- Check "Process all" to process entire file

**Timeout Duration:**
- Maximum seconds per report
- Default: 100 seconds
- Increase for complex reports

**Verbose Output:**
- Enable to see detailed processing logs
- Useful for debugging

**Rule-Based Extraction:**
- Faster processing without LLM
- Uses pattern matching
- Lower accuracy but faster

#### AI Model Settings

**LLM Model:**
- **Llama 3:** Best accuracy, moderate speed
- **Mistral:** Fast, good accuracy
- **Llama 2:** Legacy option

**Simple Prompting:**
- Use straightforward prompts
- Recommended for most cases

**Few-Shot Learning:**
- Include examples in prompts
- Improves extraction accuracy
- Slightly slower

#### RAG Settings

**Enable RAG:**
- Improves accuracy by retrieving relevant text chunks
- Requires more processing time
- Highly recommended

**Chunk Size:**
- Size of text segments (characters)
- Default: 70
- Larger = more context, slower processing

**Chunk Overlap:**
- Overlap between chunks
- Default: 20
- Prevents information loss at boundaries

**Retriever Type:**
- **Vector Store:** Semantic similarity search
- **Ensemble:** Combines vector + keyword search (slower, more accurate)

**Embedding Model:**
- **all-MiniLM-L6-v2:** Fast, good quality (recommended)
- **all-mpnet-base-v2:** Better quality, slower
- **mistral:** Uses Ollama (requires model download)

**Reranker:**
- Improves relevance of retrieved chunks
- Slower but more accurate
- Uses cross-encoder model

#### Sampling Parameters

**Temperature:**
- Controls randomness (0.0 - 1.0)
- Lower = more deterministic
- Recommended: 0.1 for medical extraction

**Top K:**
- Number of tokens to consider
- Default: 40
- Higher = more diverse outputs

**Top P:**
- Nucleus sampling threshold
- Default: 0.9
- Controls output diversity

### Processing Page

**Features:**
- Real-time progress bar
- Status messages
- Automatic status updates
- Error handling
- Completion notification

**What Happens During Processing:**
1. Configuration is loaded
2. CSV file is read
3. Text is preprocessed
4. For each report:
   - Text is chunked (if RAG enabled)
   - Embeddings are created
   - Relevant chunks are retrieved
   - LLM extracts datapoints
   - Results are validated
5. Metrics are calculated
6. Visualizations are generated
7. Results are saved

### Results Page

**Performance Metrics:**
- **Accuracy:** Overall correctness
- **Macro Precision:** Per-class precision (averaged)
- **Macro Recall:** Per-class recall (averaged)
- **Macro F1:** Harmonic mean of precision and recall
- **Reports Evaluated:** Number of reports with ground truth

**Visualizations:**
- **Confusion Matrix:** Shows prediction vs. actual labels
- Diagonal = correct predictions
- Off-diagonal = errors
- Color intensity = frequency

**Data Preview:**
- First 10 rows of results
- Shows original data + predictions
- Includes all extracted fields

**Download:**
- Complete results in CSV format
- Includes all columns and metadata
- Ready for further analysis

### About Page

- System overview
- Technology stack
- Target audience
- How it works
- Feature list
- Links to documentation

---

## Configuration Options

### Recommended Configurations

#### Fast Processing (Testing)
```yaml
Batch Size: 10
LLM Model: mistral
RAG Enabled: No
Temperature: 0.1
Rule-Based: Optional (if no LLM needed)
```

#### High Accuracy (Production)
```yaml
Batch Size: 100+
LLM Model: llama3
RAG Enabled: Yes
Chunk Size: 70
Retriever: Ensemble
Reranker: Yes
Temperature: 0.1
Few-Shot: Yes
```

#### Balanced (General Use)
```yaml
Batch Size: 50
LLM Model: llama3
RAG Enabled: Yes
Chunk Size: 70
Retriever: Vector Store
Reranker: No
Temperature: 0.1
```

---

## Understanding Results

### Interpreting Metrics

**Accuracy:**
- Overall percentage of correct predictions
- Good starting point for evaluation
- Can be misleading with imbalanced classes

**Precision:**
- Of predictions made, how many were correct
- Important when false positives are costly
- Macro: average across classes
- Micro: weighted by class frequency

**Recall:**
- Of actual instances, how many were found
- Important when false negatives are costly
- Macro vs. Micro as above

**F1 Score:**
- Harmonic mean of precision and recall
- Balances both metrics
- Good overall quality measure

### Confusion Matrix Analysis

**Reading the Matrix:**
- Rows = Actual labels
- Columns = Predicted labels
- Diagonal = Correct predictions
- Off-diagonal = Errors

**Common Patterns:**
- Concentrated diagonal = Good performance
- Scattered predictions = Poor model
- Specific off-diagonal clusters = Systematic errors

**Example:**
```
          Predicted
          0  1  2  3  NR
Actual 0  45 2  0  0  1    <- Class 0 predicted correctly 45/48 times
       1  1  38 3  0  0    <- 3 instances of class 1 mispredicted as class 2
       2  0  2  42 1  0
       3  0  0  1  44 0
       NR 0  0  0  0  5
```

---

## Advanced Features

### Custom Target Variables

Edit `config/config.yaml` to extract different datapoints:

```yaml
evaluation:
  target_variable: "Pain Score"  # Change this
  valid_values: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
```

### Batch Processing Multiple Files

Process multiple CSV files sequentially:
1. Upload and process first file
2. Download results
3. Upload next file
4. Repeat

### API Integration

The dashboard provides a REST API:

**Check Status:**
```bash
curl http://localhost:5000/status/<job_id>
```

**Download Results:**
```bash
curl http://localhost:5000/download/<job_id> -o results.csv
```

### Customizing Prompts

Edit `config/config.yaml` to customize extraction prompts:

```yaml
system_prompts:
  simple: "Extract the {target_variable} from the medical report."
  complex: "You are an expert medical AI. Carefully analyze the report and extract the {target_variable}. If not found, return 'NR'."
```

### Few-Shot Examples

Add examples in `config/config.yaml`:

```yaml
few_shot_examples:
  example1:
    input: "Patient shows mild symptoms with no complications."
    output: '{"BTFU Score (Updated)": "1"}'
  example2:
    input: "Severe symptoms requiring hospitalization."
    output: '{"BTFU Score (Updated)": "3"}'
```

---

## Troubleshooting

### Common Issues

#### 1. "Ollama not found" Error

**Problem:** LLM model unavailable

**Solution:**
```bash
# Install Ollama from ollama.ai
# Then pull the model
ollama pull llama3
ollama serve  # Start Ollama server
```

#### 2. RAG Shows as Disabled in Results (Even When Enabled)

**Problem:** Results file shows `rag_enabled_(False)` in column name even though RAG was enabled on dashboard

**Solution:** ✅ **FIXED in latest version!** This was a bug where the configuration wasn't being passed correctly from the dashboard to the backend. The fix ensures that:
- When you check "Enable RAG" on the configuration page, it's now properly passed to the processing engine
- The column names in your results will correctly show `rag_enabled_(True)` when RAG is enabled
- Your results should now have far fewer "NR" (Not Reported) values when RAG is working

**What was wrong:** The backend was using command-line argument parsing which ignored the dashboard configuration. Now it accepts the config directly from the dashboard.

#### 3. Uploaded CSV Not Being Used

**Problem:** System only works when manually placing CSV in `data/input/input_reports.csv`

**Solution:** ✅ **Already working correctly!** The uploaded file is automatically used. Here's what happens:
1. When you upload a file, it's saved to `data/input/` with a unique filename
2. The dashboard configuration automatically points to your uploaded file
3. The processing uses your uploaded file directly
4. Results are saved to `data/output/predictions.csv`

**If you're still having issues:**
- Make sure you complete the full workflow: Upload → Configure → Process
- Don't manually edit files in `data/input/` - let the dashboard handle it
- Check that your CSV has a "Report Text" column

#### 4. Results Show All "NR" (Not Reported)

**Problem:** All extractions return "NR" instead of actual values

**Cause:** This happens when:
- RAG is disabled (see issue #2 above - now fixed!)
- The AI can't access the medical reports
- The report text column is empty or misnamed

**Solutions:**
- ✅ Use the latest version with RAG fix
- Enable RAG in configuration (checkbox should be checked)
- Verify your CSV has actual text in the "Report Text" column
- Check Ollama is running: `ollama list`
- Try a smaller test batch first (10 reports)

#### 5. Out of Memory Error

**Problem:** Not enough RAM

**Solutions:**
- Reduce batch size
- Disable RAG
- Use lighter embedding model
- Close other applications
- Use rule-based extraction

#### 6. Processing Takes Too Long

**Problem:** Slow extraction

**Solutions:**
- Reduce batch size
- Use faster model (mistral)
- Disable reranker
- Increase timeout
- Use rule-based extraction for speed

#### 7. Low Accuracy

**Problem:** Poor extraction quality

**Solutions:**
- Enable RAG (now working correctly!)
- Enable reranker
- Use ensemble retriever
- Enable few-shot learning
- Reduce temperature
- Add custom examples
- Verify CSV format

#### 8. File Upload Fails

**Problem:** CSV not accepted

**Solutions:**
- Check file size (<16 MB)
- Verify CSV format
- Ensure proper encoding (UTF-8)
- Check for special characters
- Validate column names

#### 9. Dashboard Won't Start

**Problem:** Flask app error

**Solutions:**
```bash
# Check Python version
python --version  # Should be 3.7+

# Reinstall dependencies
pip install -r requirements.txt

# Check port availability
# Try different port
python app.py --port 5001

# Check error logs
python app.py --debug
```

### Getting Help

1. Check error messages in browser console (F12)
2. Review Flask logs in terminal
3. Check Ollama logs: `ollama logs`
4. Verify configuration in `config/config.yaml`
5. Test with small sample first
6. Review this documentation
7. Check GitHub issues

---

## Best Practices

### Data Preparation

1. **Clean Your Data:**
   - Remove duplicate reports
   - Fix encoding issues
   - Standardize column names
   - Remove irrelevant columns

2. **Include Ground Truth:**
   - Always include labels when available
   - Enables accuracy evaluation
   - Use consistent label format

3. **Sample First:**
   - Test with 10-20 reports
   - Verify results quality
   - Adjust parameters
   - Then scale up

### Configuration

1. **Start Simple:**
   - Use default settings
   - Gradually add features
   - Test each change

2. **Use RAG for Accuracy:**
   - RAG significantly improves results
   - Worth the extra processing time
   - Essential for long reports

3. **Adjust Temperature:**
   - Lower (0.0-0.2) for extraction tasks
   - Higher (0.5-0.8) for creative tasks
   - 0.1 is optimal for medical data

4. **Enable Few-Shot:**
   - Improves accuracy
   - Minimal speed impact
   - Add domain-specific examples

### Processing

1. **Monitor Progress:**
   - Watch for errors
   - Check intermediate saves
   - Verify output format

2. **Save Frequently:**
   - Results auto-save every 10 reports
   - Check `data/output/predictions.csv`
   - Resume if interrupted

3. **Validate Results:**
   - Spot-check predictions
   - Review confusion matrix
   - Analyze error patterns

### Production Deployment

1. **Change Secret Key:**
   ```bash
   export SECRET_KEY='your-secure-random-key'
   ```

2. **Use Production Server:**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

3. **Enable HTTPS:**
   - Use reverse proxy (nginx)
   - Configure SSL certificates
   - Secure sensitive data

4. **Set Resource Limits:**
   - Configure timeouts
   - Limit concurrent users
   - Monitor system resources

5. **Backup Data:**
   - Regular backups of uploads
   - Archive processed results
   - Version control configs

---

## Technical Architecture

### System Components

```
┌─────────────────────────────────────────────┐
│         Web Browser (User Interface)        │
└────────────────┬────────────────────────────┘
                 │ HTTP/HTTPS
                 ▼
┌─────────────────────────────────────────────┐
│           Flask Web Application             │
│  ┌──────────────────────────────────────┐   │
│  │  Routes: /, /upload, /configure,     │   │
│  │          /process, /results          │   │
│  └──────────────────────────────────────┘   │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│      MedExtract Core Engine (Python)        │
│  ┌──────────────────────────────────────┐   │
│  │  • Text Preprocessing                │   │
│  │  • LLM Integration (Ollama)          │   │
│  │  • RAG Pipeline                      │   │
│  │  • Evaluation & Metrics              │   │
│  └──────────────────────────────────────┘   │
└───┬─────────────────────────────────────┬───┘
    │                                     │
    ▼                                     ▼
┌──────────────┐                  ┌──────────────┐
│   Ollama     │                  │  LangChain   │
│   Server     │                  │  Components  │
│              │                  │              │
│  • Llama 3   │                  │  • FAISS DB  │
│  • Mistral   │                  │  • Retrievers│
│  • Llama 2   │                  │  • Embeddings│
└──────────────┘                  └──────────────┘
```

### Data Flow

1. **Upload Phase:**
   - User uploads CSV → Flask → Saved to `data/input/`
   - Unique filename generated (UUID + original name)
   - Session stores file reference

2. **Configuration Phase:**
   - User sets parameters → Flask form processing
   - Config saved as YAML → `config/temp_config_<id>.yaml`
   - Session stores config reference

3. **Processing Phase:**
   - Background thread spawned
   - Config loaded → MedExtract initialized
   - For each report:
     - Text preprocessed
     - If RAG: chunked → embedded → stored in FAISS
     - Query sent to LLM via Ollama
     - Response parsed and validated
     - Result saved to dataframe
   - Dataframe saved incrementally

4. **Results Phase:**
   - Metrics calculated (accuracy, precision, recall, F1)
   - Confusion matrix generated
   - Results displayed in web interface
   - CSV available for download

### File Structure

```
medextract/
├── app.py                      # Flask web application
├── medextract.py               # Core extraction engine
├── requirements.txt            # Python dependencies
├── config/
│   ├── config.yaml            # User configuration
│   ├── default_config.yaml    # Default settings
│   └── temp_config_*.yaml     # Session configs
├── templates/                  # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── upload.html
│   ├── configure.html
│   ├── process.html
│   ├── results.html
│   └── about.html
├── data/
│   ├── input/                 # Uploaded CSV files
│   └── output/
│       ├── predictions.csv    # Main results
│       ├── figures/          # Confusion matrices
│       └── results/
│           ├── metrics.csv   # Performance metrics
│           └── log.csv       # Processing log
└── DASHBOARD_TUTORIAL.md      # This file
```

### Technology Stack

**Backend:**
- **Flask:** Web framework (lightweight, flexible)
- **pandas:** Data manipulation and CSV handling
- **LangChain:** RAG framework and LLM orchestration
- **Ollama:** Local LLM serving (privacy-preserving)
- **sentence-transformers:** Text embeddings
- **FAISS:** Vector similarity search
- **scikit-learn:** Metrics and evaluation

**Frontend:**
- **Bootstrap 5:** Responsive UI framework
- **Bootstrap Icons:** Icon library
- **Vanilla JavaScript:** Interactive features
- **Jinja2:** Template engine

**AI Models:**
- **Llama 3:** Primary LLM (Meta)
- **Mistral:** Alternative LLM (Mistral AI)
- **all-MiniLM-L6-v2:** Fast embedding model
- **BGE-reranker-v2-m3:** Cross-encoder reranker

---

## Conclusion

The MedExtract Dashboard successfully bridges the gap between a powerful but complex clinical data extraction tool and its intended users—researchers, biologists, and health scientists who need AI capabilities without coding expertise.

### What You've Learned

- How to install and configure MedExtract Dashboard
- How to upload and process medical reports
- How to interpret accuracy metrics and visualizations
- How to troubleshoot common issues
- Best practices for production use

### Next Steps

1. **Test with Sample Data:** Start with a small CSV file
2. **Experiment with Configurations:** Try different models and settings
3. **Evaluate Results:** Review metrics and confusion matrices
4. **Scale Up:** Process larger datasets
5. **Customize:** Adapt to your specific use case
6. **Deploy:** Move to production environment

### Support and Resources

- **GitHub Repository:** [github.com/mustafajamis/medextract](https://github.com/mustafajamis/medextract)
- **Issues:** Report bugs or request features on GitHub
- **Documentation:** This tutorial and README.md
- **Ollama Docs:** [ollama.ai/docs](https://ollama.ai/docs)
- **LangChain Docs:** [python.langchain.com](https://python.langchain.com)

### Final Notes

This dashboard represents a "novel application system" built on top of the MedExtract technology, making AI-powered medical data extraction accessible to everyone. Whether you're conducting research, analyzing patient outcomes, or building datasets for machine learning, MedExtract Dashboard provides the tools you need in an intuitive, user-friendly package.

**Happy Extracting! 🚀**

---

*Last Updated: November 2024*
*Version: 1.0*
*Authors: MedExtract Development Team*
