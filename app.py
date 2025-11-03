"""
MedExtract Web Dashboard
A user-friendly web interface for the MedExtract clinical datapoint extraction system.
"""
import os
import json
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file, jsonify
from werkzeug.utils import secure_filename
import yaml
import threading
import uuid
from datetime import datetime
import traceback

# Import medextract functionality
import medextract

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['UPLOAD_FOLDER'] = 'data/input'
app.config['OUTPUT_FOLDER'] = 'data/output'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Store processing status in memory (for demo purposes)
processing_status = {}

def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Upload CSV file page."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Generate unique filename to avoid conflicts
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)
            
            # Store in session
            session['uploaded_file'] = unique_filename
            session['original_filename'] = filename
            
            flash(f'File "{filename}" uploaded successfully!', 'success')
            return redirect(url_for('configure'))
        else:
            flash('Invalid file type. Please upload a CSV file.', 'error')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/configure', methods=['GET', 'POST'])
def configure():
    """Configure processing parameters."""
    if 'uploaded_file' not in session:
        flash('Please upload a file first.', 'warning')
        return redirect(url_for('upload'))
    
    # Load default config
    with open('config/default_config.yaml', 'r') as f:
        default_config = yaml.safe_load(f)
    
    if request.method == 'POST':
        # Get configuration from form
        config = {
            'file_paths': {
                'input': os.path.join(app.config['UPLOAD_FOLDER'], session['uploaded_file']),
                'figures': 'data/output/figures/',
                'results': 'data/output/results/',
                'metrics': 'data/output/results/metrics.csv',
                'log': 'data/output/results/log.csv',
                'predictions': 'data/output/predictions.csv'
            },
            'processing': {
                'batch_size': int(request.form.get('batch_size', 10)),
                'process_all': request.form.get('process_all') == 'on',
                'verbose': request.form.get('verbose') == 'on',
                'timeout_duration': int(request.form.get('timeout_duration', 100)),
                'csv_save_frequency': 10,
                'force_rule_based': request.form.get('use_rule_based') == 'on'
            },
            'rag': {
                'enabled': request.form.get('rag_enabled') == 'on',
                'chunk_size': int(request.form.get('chunk_size', 70)),
                'chunk_overlap': int(request.form.get('chunk_overlap', 20))
            },
            'models': {
                'llm_models': [request.form.get('llm_model', 'llama3:latest')]
            },
            'prompting': {
                'simple_prompting': request.form.get('simple_prompting') == 'on',
                'fewshots_method': request.form.get('fewshots_method') == 'on',
                'fewshots_with_NR_method': False,
                'fewshots_with_NR_extended_method': False
            },
            'output': {
                'json_format': True
            },
            'sampling': {
                'temperatures': [float(request.form.get('temperature', 0.1))],
                'top_ks': [int(request.form.get('top_k', 40))],
                'top_ps': [float(request.form.get('top_p', 0.9))]
            },
            'embedding_models': [request.form.get('embedding_model', 'all-MiniLM-L6-v2')],
            'retriever': {
                'types': [request.form.get('retriever_type', 'vectorstore')],
                'use_reranker': request.form.get('use_reranker') == 'on',
                'reranker_model_name': 'BAAI/bge-reranker-v2-m3',
                'reranker_top_n': 2
            },
            'evaluation': default_config['evaluation'],
            'advanced_llm': default_config['advanced_llm'],
            'few_shot_examples': default_config['few_shot_examples'],
            'system_prompts': default_config['system_prompts'],
            'metrics': default_config['metrics'],
            'column_name_format': default_config['column_name_format'],
            'library_versions': default_config['library_versions'],
            'run_benchmark': False
        }
        
        # Save config to temporary file
        config_path = f'config/temp_config_{session["uploaded_file"].split("_")[0]}.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        session['config_file'] = config_path
        
        flash('Configuration saved successfully!', 'success')
        return redirect(url_for('process'))
    
    return render_template('configure.html', config=default_config)

@app.route('/process')
def process():
    """Start processing the uploaded file."""
    if 'uploaded_file' not in session or 'config_file' not in session:
        flash('Please upload a file and configure parameters first.', 'warning')
        return redirect(url_for('upload'))
    
    job_id = session['uploaded_file'].split('_')[0]
    
    # Check if already processing
    if job_id in processing_status and processing_status[job_id]['status'] == 'processing':
        return render_template('process.html', job_id=job_id, status='processing')
    
    # Initialize processing status
    processing_status[job_id] = {
        'status': 'processing',
        'progress': 0,
        'message': 'Starting processing...',
        'started_at': datetime.now().isoformat()
    }
    
    # Start processing in background thread
    thread = threading.Thread(target=run_processing, args=(job_id, session['config_file']))
    thread.daemon = True
    thread.start()
    
    return render_template('process.html', job_id=job_id, status='processing')

def run_processing(job_id, config_path):
    """Run the medextract processing in background."""
    try:
        processing_status[job_id]['message'] = 'Loading configuration...'
        processing_status[job_id]['progress'] = 10
        
        # Load config
        medextract.config = medextract.load_config(config_path)
        
        processing_status[job_id]['message'] = 'Processing medical reports...'
        processing_status[job_id]['progress'] = 30
        
        # Run main processing
        medextract.main()
        
        processing_status[job_id]['status'] = 'completed'
        processing_status[job_id]['progress'] = 100
        processing_status[job_id]['message'] = 'Processing completed successfully!'
        processing_status[job_id]['completed_at'] = datetime.now().isoformat()
        
    except Exception as e:
        processing_status[job_id]['status'] = 'error'
        processing_status[job_id]['message'] = f'Error: {str(e)}'
        processing_status[job_id]['error'] = traceback.format_exc()
        processing_status[job_id]['completed_at'] = datetime.now().isoformat()

@app.route('/status/<job_id>')
def status(job_id):
    """Get processing status via API."""
    if job_id not in processing_status:
        return jsonify({'status': 'not_found'}), 404
    
    return jsonify(processing_status[job_id])

@app.route('/results/<job_id>')
def results(job_id):
    """Display results page."""
    if job_id not in processing_status:
        flash('Job not found', 'error')
        return redirect(url_for('index'))
    
    if processing_status[job_id]['status'] != 'completed':
        flash('Processing not completed yet', 'warning')
        return redirect(url_for('process'))
    
    # Load results
    try:
        predictions_path = 'data/output/predictions.csv'
        if os.path.exists(predictions_path):
            df = pd.read_csv(predictions_path)
            results_data = {
                'total_reports': len(df),
                'columns': df.columns.tolist(),
                'preview': df.head(10).to_html(classes='table table-striped', index=False)
            }
        else:
            results_data = {'error': 'Results file not found'}
        
        # Load metrics if available
        metrics_path = 'data/output/results/metrics.csv'
        if os.path.exists(metrics_path):
            metrics_df = pd.read_csv(metrics_path)
            results_data['metrics'] = metrics_df.tail(1).to_dict('records')[0] if not metrics_df.empty else {}
        
        # Get list of generated figures
        figures_path = 'data/output/figures/'
        if os.path.exists(figures_path):
            figures = [f for f in os.listdir(figures_path) if f.endswith('.png')]
            results_data['figures'] = figures
        else:
            results_data['figures'] = []
        
        return render_template('results.html', job_id=job_id, results=results_data)
    
    except Exception as e:
        flash(f'Error loading results: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/download/<job_id>')
def download(job_id):
    """Download results CSV."""
    predictions_path = 'data/output/predictions.csv'
    if os.path.exists(predictions_path):
        return send_file(predictions_path, as_attachment=True, download_name='medextract_results.csv')
    else:
        flash('Results file not found', 'error')
        return redirect(url_for('results', job_id=job_id))

@app.route('/figure/<filename>')
def figure(filename):
    """Serve figure image."""
    figures_path = os.path.join('data/output/figures', secure_filename(filename))
    if os.path.exists(figures_path):
        return send_file(figures_path, mimetype='image/png')
    else:
        return "Figure not found", 404

@app.route('/about')
def about():
    """About page."""
    return render_template('about.html')

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('data/input', exist_ok=True)
    os.makedirs('data/output/figures', exist_ok=True)
    os.makedirs('data/output/results', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
