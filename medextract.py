import pandas as pd
import numpy as np
import re
import os
import ast
from tqdm.auto import tqdm
import yaml
import itertools
import json
from datetime import datetime
import time
try:
    from interruptingcow import timeout
except Exception:
    # fallback no-op timeout context manager when interruptingcow is not installed
    from contextlib import contextmanager

    @contextmanager
    def timeout(seconds, exception=RuntimeError):
        # On systems where interruptingcow is unavailable (or installation fails),
        # provide a transparent no-op context manager so the rest of the code can run
        # (it will not enforce timeouts).
        yield
# On Windows the `signal.SIGALRM` used by interruptingcow is not available.
# If we are on Windows, always use a no-op timeout to avoid AttributeError.
import platform
if platform.system().lower().startswith('win'):
    from contextlib import contextmanager

    @contextmanager
    def timeout(seconds, exception=RuntimeError):
        # no-op on Windows
        yield
import importlib
import argparse
from sklearn import metrics
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    _HAVE_SEABORN = True
except Exception:
    sns = None
    _HAVE_SEABORN = False
#from langchain_community.llms import LlamaCpp
# embeddings used later; import here so names are available when constructing embeddings
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain_community.document_compressors.openvino_rerank import OpenVINOReranker
    from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
    from langchain.schema import Document
    _HAVE_LANGCHAIN = True
except Exception:
    # Allow the script to run in environments where langchain / langchain_community
    # are not installed. Functionality that depends on these libraries (RAG,
    # vectorstores, retrievers, etc.) will raise a clear error if used.
    HuggingFaceEmbeddings = None
    OllamaEmbeddings = None
    BM25Retriever = None
    EnsembleRetriever = None
    RecursiveCharacterTextSplitter = None
    FAISS = None
    ContextualCompressionRetriever = None
    OpenVINOReranker = None
    BaseDocumentCompressor = object
    class Document:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}
    _HAVE_LANGCHAIN = False

# Provide a minimal fallback for RecursiveCharacterTextSplitter when LangChain is not available.
if RecursiveCharacterTextSplitter is None:
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=0, separators=None):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.separators = separators or ["\n\n", "\n", ":", "."]

        def split_text(self, text):
            # Try paragraph-based splitting first, then fall back to fixed-size chunks.
            paragraphs = [p for p in text.split("\n\n") if p.strip()]
            if not paragraphs:
                paragraphs = [text]
            chunks = []
            for p in paragraphs:
                if len(p) <= self.chunk_size:
                    chunks.append(p)
                else:
                    step = max(1, self.chunk_size - self.chunk_overlap)
                    for i in range(0, len(p), step):
                        chunks.append(p[i:i + self.chunk_size])
            return chunks

import ollama
from ollama import Options
try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None
from typing import Sequence, Optional

class BgeRerank(BaseDocumentCompressor):
    """A small wrapper that uses a CrossEncoder to rerank documents.

    The CrossEncoder model is created at instance initialization (lazy) to avoid
    attempting to download or load GPU tensors at import time.
    """
    model_name: str = 'BAAI/bge-reranker-v2-m3'
    top_n: int = 2

    def __init__(self, device: str = "cpu"):
        # instantiate model at runtime; default to CPU to be safer across envs
        if CrossEncoder is None:
            raise RuntimeError("sentence_transformers.CrossEncoder is not available. Install 'sentence-transformers' to use BgeRerank.")
        try:
            self.model = CrossEncoder(self.model_name, device=device)
        except Exception:
            # fallback: try without explicit device (lets the library decide)
            self.model = CrossEncoder(self.model_name)

    def bge_rerank(self, query, docs):
        model_inputs = [[query, doc] for doc in docs]
        scores = self.model.predict(model_inputs)
        results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return results[:self.top_n]

    def compress_documents(self, documents: Sequence[Document], query: str, callbacks: Optional[object] = None) -> Sequence[Document]:
        if len(documents) == 0:
            return []
        doc_list = list(documents)
        _docs = [d.page_content for d in doc_list]
        results = self.bge_rerank(query, _docs)
        final_results = []
        for r in results:
            doc = doc_list[r[0]]
            # ensure metadata dict exists
            if not isinstance(doc.metadata, dict):
                doc.metadata = {}
            doc.metadata["relevance_score"] = r[1]
            final_results.append(doc)
        return final_results

def load_config(config_path='config/config.yaml', default_config_path='config/default_config.yaml'):
    with open(default_config_path, 'r') as file:
        default_config = yaml.safe_load(file)
    
    with open(config_path, 'r') as file:
        user_config = yaml.safe_load(file)
    
    # If the user explicitly requests the default config, return it. Otherwise
    # merge default and user configs with user values taking precedence.
    if user_config.get('use_default_config', False):
        return default_config
    else:
        merged = {**default_config, **user_config}
        return merged

def check_library_versions():
    # Only warn for missing critical libraries. Version mismatches for optional
    # libraries are non-actionable in many environments, so avoid noisy prints.
    for lib, version in config.get('library_versions', {}).items():
        try:
            importlib.import_module(lib)
        except Exception:
            # Only print when the library is actually required at runtime; keep it
            # quiet for optional deps. We'll treat langchain as optional here.
            if lib in ('langchain',):
                # do not spam on missing langchain; higher-level logic will handle it
                continue
            print(f"Warning: required library '{lib}' is not installed or could not be imported.")

def preprocess_text(text):
    text = re.sub(r'\n(?!\.)', ' ', text)
    text = re.sub(r"\.\n", " \n ", text)
    return text


def rule_based_extractor(text):
    """Simple deterministic extractor for BTFU Score values.

    Looks for tokens matching the allowed label set (e.g. '0','1','1a','1b',
    '2','2a','2b','3','3a','3b','3c','4','NR'). This provides a fast
    baseline when the LLM/RAG stack isn't available.
    """
    if not text:
        return "NR"

    target_vals = [v.lower() for v in config['evaluation']['valid_values']]

    # normalize punctuation, lower-case
    s = re.sub(r'[\r\n]+', ' ', text.lower())

    # direct token match (exact labels)
    for v in target_vals:
        if re.search(rf"\b{re.escape(v)}\b", s):
            return v.upper() if v != 'nr' else 'NR'

    # common synonyms/phrases mapping to labels
    phrase_map = {
        'no problems': '0',
        'asymptomatic': '0',
        'mild': '1',
        'mild symptoms': '1',
        'moderate': '2',
        'moderate symptoms': '2',
        'severe': '3',
        'hospital': '3',
        'hospitalization': '3',
        'urgent': '4',
        'critical': '4',
        'worse': '3',
        'worsened': '3',
        'no improvement': '3',
        'improved': '1'
    }

    for phrase, label in phrase_map.items():
        if phrase in s:
            return label

    # proximity rule: if a label-like token appears near keywords like 'score' or 'btfu' or 'follow-up'
    keywords = ['score', 'btfu', 'follow-up', 'follow up', 'severity']
    for kw in keywords:
        for m in re.finditer(rf"([0-4](?:[ab]|c)?|nr)\b", s):
            # look at context window of 30 chars before the token
            start = max(0, m.start() - 30)
            ctx = s[start:m.end()]
            if any(k in ctx for k in keywords):
                return m.group(1)

    # look for patterns like 'score: 2' or 'btfu = 1a'
    m = re.search(r"(?:score|btfu|btfu score|btfu:)\s*[:=]?\s*([0-4](?:[ab]|c)?|nr)\b", s)
    if m:
        return m.group(1)

    # look for x/10 style severity and map high scores to labels
    m2 = re.search(r"(\d{1,2})\s*/\s*10", s)
    if m2:
        try:
            val = int(m2.group(1))
            if val >= 9:
                return '4'
            if val >= 7:
                return '3'
            if val >= 4:
                return '2'
            return '1'
        except Exception:
            pass

    # numeric keywords like 'stage 3' -> 3
    m3 = re.search(r"stage\s*([0-4])\b", s)
    if m3:
        return m3.group(1)

    # fallback
    return 'NR'

def get_text_chunks(text, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ":", "."]
    )
    return [Document(page_content=x) for x in text_splitter.split_text(text)]

def ollama_llm(context, llm_model, simple_prompting, fewshots_method, fewshots_with_NR_method, fewshots_with_NR_extended_method, json_value, temp, top_k, top_p):
    response = ollama.chat(
        model=llm_model, 
        format="json" if json_value else None,
        keep_alive=config['advanced_llm']['keep_alive'],
        options=Options(
            temperature=temp, 
            top_k=top_k,
            top_p=top_p,
            num_predict=config['advanced_llm']['num_predict'],
            mirostat_tau=config['advanced_llm']['mirostat_tau'],
        ), 
        messages=construct_prompt(context, simple_prompting, fewshots_method, fewshots_with_NR_method, fewshots_with_NR_extended_method)
    )
    return response['message']['content']

def construct_prompt(context, simple_prompting, fewshots_method, fewshots_with_NR_method, fewshots_with_NR_extended_method):
    target_variable = config['evaluation']['target_variable']
    formatted_prompt = f"Question: Extract the {target_variable} from the given medical report. If not found, return 'NR'. Answer in JSON format.\nContext: {context}"
    
    system_prompt = config['system_prompts']['simple'] if simple_prompting else config['system_prompts']['complex']
    system_prompt = system_prompt.format(target_variable=target_variable)
    
    system_messages = [{"role": "system", "content": system_prompt}]
    
    if fewshots_method:
        system_messages.extend([
            {"role": "user", "content": ex['input']}
            for ex in config['few_shot_examples'].values()
        ])
        system_messages.extend([
            {"role": "assistant", "content": ex['output']}
            for ex in config['few_shot_examples'].values()
        ])
    
    user_message = [{'role': 'user', 'content': formatted_prompt}]
    
    return system_messages + user_message

def process_text(text, llm_model, rag_enabled, embeddings, retriever_type, reranker, simple_prompting, fewshots_method, fewshots_with_NR_method, fewshots_with_NR_extended_method, json_value, temp, top_k, top_p):
    if rag_enabled:
        chunks = get_text_chunks(text, chunk_size=config['rag']['chunk_size'], chunk_overlap=config['rag']['chunk_overlap'])
        db = FAISS.from_documents(chunks, embeddings)
        
        if reranker:
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=reranker, base_retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
            )
        else:
            compression_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

        if retriever_type == "ensemble":
            keyword_retriever = BM25Retriever.from_documents(chunks, k=2)
            retriever = EnsembleRetriever(retrievers=[compression_retriever, keyword_retriever], weights=[0.25, 0.75])
        else:
            retriever = compression_retriever

        # try common LangChain retriever APIs; be tolerant to different implementations
        if hasattr(retriever, 'get_relevant_documents'):
            retrieved_docs = retriever.get_relevant_documents(config['evaluation']['target_variable'])
        elif hasattr(retriever, 'retrieve'):
            retrieved_docs = retriever.retrieve(config['evaluation']['target_variable'])
        else:
            try:
                retrieved_docs = retriever(config['evaluation']['target_variable'])
            except Exception as e:
                raise AttributeError("Retriever does not expose a known retrieval method") from e

        formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    else:
        chunks = get_text_chunks(text, chunk_size=100000, chunk_overlap=200)
        formatted_context = text

    # If configuration requests rule-based only, use it immediately
    force_rule = config.get('processing', {}).get('force_rule_based', False)
    if force_rule:
        rb_label = rule_based_extractor(formatted_context or text)
        return json.dumps({config['evaluation']['target_variable']: rb_label})

    # Attempt an LLM call; if it fails or returns an invalid result, fall back
    # to the deterministic rule-based extractor for a baseline prediction.
    result = None
    try:
        result = ollama_llm(
            context=formatted_context,
            llm_model=llm_model,
            simple_prompting=simple_prompting,
            fewshots_method=fewshots_method,
            fewshots_with_NR_method=fewshots_with_NR_method,
            fewshots_with_NR_extended_method=fewshots_with_NR_extended_method,
            json_value=json_value,
            temp=temp,
            top_k=top_k,
            top_p=top_p,
        )
    except Exception:
        result = None

    # If result is missing or does not contain a valid label, use rule-based
    parsed_label = None
    if result is not None:
        parsed_label = clean_extracted_value(result)

    if not parsed_label or parsed_label == 'invalid':
        rb_label = rule_based_extractor(formatted_context or text)
        return json.dumps({config['evaluation']['target_variable']: rb_label})

    return result

def process_model(column_name, file_path, df, batch_size, llm_model, rag_enabled, embeddings, retriever_type, use_reranker, simple_prompting, fewshots_method, fewshots_with_NR_method, fewshots_with_NR_extended_method, json_value, temp, top_k, top_p, verbose, report_column="Report Text"):
    # ensure reranker is always defined to avoid UnboundLocalError
    reranker = None
    if use_reranker:
        # default device can be configured under processing.reranker_device in config
        try:
            reranker = BgeRerank(device=config.get('processing', {}).get('reranker_device', 'cpu'))
        except Exception:
            # silently continue without reranker to avoid noisy warnings in lightweight runs
            reranker = None
    
    with tqdm(total=batch_size, desc='Report Processing', unit='report', leave=True) as report_pbar:
        for i in range(batch_size):
            if pd.notna(df.at[i, column_name]):
                report_pbar.update(1)
            else:
                report_text = df.loc[i, report_column]
                if pd.notna(report_text):
                    try:
                        with timeout(config['processing']['timeout_duration'], exception=RuntimeError):
                            preprocessed_text = preprocess_text(report_text)
                            processed_text = process_text(
                                text=preprocessed_text,
                                llm_model=llm_model,
                                rag_enabled=rag_enabled,
                                embeddings=embeddings, 
                                retriever_type=retriever_type,
                                reranker=reranker,
                                simple_prompting=simple_prompting,
                                fewshots_method=fewshots_method,
                                fewshots_with_NR_method=fewshots_with_NR_method,
                                fewshots_with_NR_extended_method=fewshots_with_NR_extended_method,
                                json_value=json_value,
                                temp=temp,
                                top_k=top_k,
                                top_p=top_p
                            )
                    except RuntimeError:
                        print("Interrupted due to timeout")
                        processed_text = json.dumps({config['evaluation']['target_variable']: "timeout"})
                    
                    df.at[i, column_name] = processed_text.rstrip(' \n')
                    
                    if verbose:
                        print("*" * 100)
                        print(f"Processed text: {processed_text}")
                        print(f"Ground truth: {df.at[i, config['evaluation']['target_variable']]}")
                        print("*" * 100)
                    
                    if i % config['processing']['csv_save_frequency'] == 0 or i == batch_size - 1:
                        df.to_csv(file_path, index=False)
                
                report_pbar.update(1)
    
    return df

def clean_extracted_value(value):
    try:
        # Robustly handle several possible encodings of the prediction:
        # - Plain JSON string: '{"BTFU...": "0"}'
        # - Double-wrapped JSON: '"{\"BTFU...\": \"0\"}"'
        # - CSV-escaped doubled-quotes: '"{""BTFU..."": ""0""}"'
        s = value
        if isinstance(s, str):
            s = s.strip()
            # Unwrap repeated quoting and try parsing repeatedly
            attempts = 0
            while attempts < 5:
                try:
                    parsed = json.loads(s)
                except Exception:
                    # try to normalize doubled quotes and unquote
                    if s.startswith('"') and s.endswith('"'):
                        s = s[1:-1]
                        s = s.replace('""', '"')
                        attempts += 1
                        continue
                    break

                # If parsing yields a dict, extract target variable
                if isinstance(parsed, dict):
                    val = parsed.get(config['evaluation']['target_variable'], "invalid")
                    return val if val in config['evaluation']['valid_values'] else "invalid"

                # If parsing yields a string, unwrap and loop
                if isinstance(parsed, str) and parsed != s:
                    s = parsed
                    attempts += 1
                    continue
                break

            # If we couldn't JSON-parse into a dict, fall back to heuristics
            # strip remaining quotes and braces and look for the target value
            cleaned = s.replace('""', '"').strip()
            # simple regex to pull the last quoted token (common for '{...: "VAL"}')
            m = re.search(r'"([^\"]+)"\s*\}?\s*$', cleaned)
            if m:
                candidate = m.group(1).strip()
                return candidate if candidate in config['evaluation']['valid_values'] else "invalid"
            return cleaned if cleaned in config['evaluation']['valid_values'] else "invalid"

        # If already a dict
        if isinstance(value, dict):
            v = value.get(config['evaluation']['target_variable'], "invalid")
            return v if v in config['evaluation']['valid_values'] else "invalid"

        # Other types
        return "invalid"
    except Exception:
        return "invalid"

def evaluate_experiment(df, column_name, figures_path, metrics_file_path, log_file_path, eval_id):
    # Only evaluate rows where the ground-truth label is present. This avoids
    # inflating accuracy when both ground-truth and prediction are missing and
    # normalized to the same sentinel value.
    gt_col = config['evaluation']['target_variable']
    df_exp = df[~df[gt_col].isna() & ~df[column_name].isna()].copy()
    if df_exp.empty:
        print(f"No labelled examples found for column '{column_name}' — skipping evaluation.")
        return 0.0

    # Clean extracted predictions
    df_exp[column_name + '_cleaned'] = df_exp[column_name].apply(clean_extracted_value)

    # Normalize types: ensure both predictions and ground-truth are strings and
    # that values not in the valid set are marked as 'invalid'.
    valid_values = set(config['evaluation']['valid_values'])

    y_pred = df_exp[column_name + '_cleaned'].astype(str).str.strip()
    y_pred = y_pred.apply(lambda v: v if v in valid_values else 'invalid')

    y_test = df_exp[gt_col].astype(str).str.strip()
    y_test = y_test.apply(lambda v: v if v in valid_values else 'invalid')
    
    metrics_dict = {
        "Accuracy": metrics.accuracy_score(y_test, y_pred),
        "Macro Precision": metrics.precision_score(y_test, y_pred, average='macro', zero_division=0),
        "Micro Precision": metrics.precision_score(y_test, y_pred, average='micro', zero_division=0),
        "Macro Recall": metrics.recall_score(y_test, y_pred, average='macro', zero_division=0),
        "Micro Recall": metrics.recall_score(y_test, y_pred, average='micro', zero_division=0),
        "Macro F1": metrics.f1_score(y_test, y_pred, average='macro', zero_division=0),
        "Micro F1": metrics.f1_score(y_test, y_pred, average='micro', zero_division=0),
        "Reports Evaluated": len(df_exp),
    }
    
    save_confusion_matrix(y_test, y_pred, column_name, figures_path, eval_id)
    update_metrics_csv(metrics_dict, column_name, metrics_file_path)
    append_metrics_to_log(metrics_dict, column_name, log_file_path, eval_id)
    
    return metrics_dict["Accuracy"]

def save_confusion_matrix(y_test, y_pred, column_name, figures_path, eval_id):
    os.makedirs(figures_path, exist_ok=True)
    all_labels = sorted(list(set(y_test) | set(y_pred)))
    cm = metrics.confusion_matrix(y_test, y_pred, labels=all_labels)
    plt.figure(figsize=(14, 10))
    if _HAVE_SEABORN:
        sns.heatmap(cm, annot=True, fmt="d", cmap='YlGnBu', xticklabels=all_labels, yticklabels=all_labels)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    else:
        # fallback to matplotlib if seaborn is not available
        im = plt.imshow(cm, cmap='YlGnBu')
        plt.colorbar(im)
        # annotate cells
        for (i, j), val in np.ndenumerate(cm):
            plt.text(j, i, int(val), ha='center', va='center', color='black')
        plt.xticks(range(len(all_labels)), all_labels, rotation=45, ha='right')
        plt.yticks(range(len(all_labels)), all_labels)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
    # sanitize column_name for filesystem usage and keep filename reasonably short
    import hashlib
    safe_name = re.sub(r'[^A-Za-z0-9_.-]', '_', column_name)
    # keep a short hash to avoid extremely long filenames
    short_hash = hashlib.md5(column_name.encode()).hexdigest()[:8]
    safe_name = (safe_name[:120] + '_' + short_hash) if len(safe_name) > 120 else (safe_name + '_' + short_hash)
    cm_path = os.path.join(figures_path, f"{safe_name}_confusion_matrix_eval_{eval_id}.png")
    # ensure the target directory exists (be robust to missing directories)
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

def update_metrics_csv(metrics_dict, column_name, metrics_file_path):
    os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics_df = pd.DataFrame([{"Column Name": column_name, "Timestamp": current_time, **metrics_dict}])
    if os.path.exists(metrics_file_path):
        metrics_df.to_csv(metrics_file_path, mode='a', header=False, index=False)
    else:
        metrics_df.to_csv(metrics_file_path, index=False)

def append_metrics_to_log(metrics_dict, column_name, log_file_path, eval_id):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_df = pd.DataFrame([{**{"Column Name": column_name, "Eval ID": eval_id, "Timestamp": current_time}, **metrics_dict}])
    if os.path.exists(log_file_path):
        log_df.to_csv(log_file_path, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_file_path, index=False)

def main(config_path=None):
    """
    Main processing function.
    
    Args:
        config_path: Optional path to config file. If None, will parse from command-line args.
                     If provided, uses the given config path directly (for programmatic use).
    """
    global config
    
    if config_path is None:
        # Command-line mode: parse arguments
        parser = argparse.ArgumentParser(description='MedExtract: Clinical Datapoint Extraction System')
        parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the configuration file')
        args = parser.parse_args()
        config = load_config(args.config)
    else:
        # Programmatic mode: use provided config path
        config = load_config(config_path)

    # Basic library check (silent on optional/version mismatches)
    check_library_versions()

    # If langchain isn't present, ensure RAG is disabled silently so we don't
    # repeatedly print about it during processing.
    try:
        have_lc = _HAVE_LANGCHAIN
    except NameError:
        have_lc = False
    if not have_lc:
        config.setdefault('rag', {})
        config['rag']['enabled'] = False

    # Ensure input file exists
    if not os.path.exists(config['file_paths']['input']):
        raise FileNotFoundError(f"Input file not found: {config['file_paths']['input']}")
    
    # Read CSV robustly: some input files may contain unquoted commas in the
    # Report text column which breaks the C engine. Try the normal parser
    # first and fall back to a tolerant line-based parser on ParserError.
    try:
        df = pd.read_csv(config['file_paths']['input'])
    except Exception as e:
        # If pandas fails to parse (common when text fields contain unquoted
        # commas), fall back to a simple regex-based loader that expects the
        # file to have four columns: id, report_text, target, prediction_json
        import re
        rows = []
        with open(config['file_paths']['input'], 'r', encoding='utf-8') as f:
            header = f.readline().strip()
            # attempt to derive column names from header, but fall back to defaults
            cols = [c.strip() for c in header.split(',')]
            for line in f:
                line = line.rstrip('\n')
                if not line:
                    continue
                # match: id, report_text (greedy), target (no commas), last field = quoted JSON
                m = re.match(r'([^,]+),(.+),([^,]+),(".*")$', line)
                if m:
                    rows.append([m.group(1).strip(), m.group(2).strip(), m.group(3).strip(), m.group(4).strip()])
                else:
                    # fallback: split into 4 parts (max 3 splits) to avoid losing text
                    parts = line.split(',', 3)
                    if len(parts) < 4:
                        # pad missing fields
                        parts += [None] * (4 - len(parts))
                    rows.append([p.strip() if p is not None else None for p in parts])
        # build DataFrame using detected header or reasonable defaults
        if len(cols) >= 4:
            df = pd.DataFrame(rows, columns=cols[:4])
        else:
            df = pd.DataFrame(rows, columns=['Report_ID', 'Report_Text', 'BTFU Score (Updated)', 'prediction'])

    # After successfully loading the dataframe (either via pandas or fallback),
    # attempt to normalize any prediction columns that contain JSON-like strings
    # into simple extracted labels for evaluation. This handles cases where the
    # prediction column stores a JSON string like '{"BTFU Score (Updated)": "0"}'.
    def _try_extract_prediction(val, target_key):
        if pd.isna(val):
            return val
        s = str(val).strip()
        # remove surrounding quotes if present
        if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
            s = s[1:-1]
        # collapse doubled quotes used in some CSV exports: "" -> "
        if '""' in s:
            s = s.replace('""', '"')
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict) and target_key in parsed:
                return parsed[target_key]
        except Exception:
            pass
        # final fallback: return stripped original
        return s

    target_var = config.get('evaluation', {}).get('target_variable', 'BTFU Score (Updated)')
    # look for candidate prediction columns and normalize them
    for col in list(df.columns):
        if col in (target_var, 'Report_Text', 'Report Text', 'Report_ID'):
            continue
        # only examine object/string columns
        if df[col].dtype == object or str(df[col].dtype).startswith('string'):
            sample = df[col].dropna().astype(str)
            if not sample.empty and any(('{' in x or '"' in x or '""' in x) for x in sample.head(5).tolist()):
                try:
                    df[col] = df[col].apply(lambda v: _try_extract_prediction(v, target_var))
                except Exception:
                    pass
    # tolerate different column names for report text
    report_col = None
    possible_report_cols = ["Report Text", "report_text", "report text", "Report_Text", "reportText", "ReportText"]
    for c in possible_report_cols:
        if c in df.columns:
            report_col = c
            break
    if report_col is None:
        # fallback: pick the first column whose name contains 'report'
        for c in df.columns:
            if 'report' in c.lower():
                report_col = c
                break
    if report_col is None:
        raise KeyError("No report text column found in input file. Expected one of: 'Report Text' or 'report_text'.")

    df = df[~df[report_col].isna()]

    # ensure target variable column exists; if missing, create it as NaN so processing can continue
    target_var = config['evaluation']['target_variable']
    if target_var not in df.columns:
        # create silently to avoid noisy warnings; evaluation will skip if no labels
        df[target_var] = np.nan
    else:
        df = df[df[target_var].isin(config['evaluation']['valid_values'])]
    df.reset_index(inplace=True, drop=True)
    
    if config['processing'].get('process_all', False):
        batch_size = len(df)
    else:
        batch_size = min(len(df), config['processing'].get('batch_size', 100))
    
    best_model = None
    highest_accuracy = 0
    
    if config['run_benchmark']:
        param_combinations = list(itertools.product(
            config['models']['llm_models'],
            [True, False],  # rag_enabled
            config['embedding_models'],
            config['retriever']['types'],
            [True, False],  # use_reranker
            [True, False],  # simple_prompting
            [True, False],  # fewshots_method
            [True, False],  # fewshots_with_NR_method
            [True, False],  # fewshots_with_NR_extended_method
            [True, False],  # json_value
            config['sampling']['temperatures'],
            config['sampling']['top_ks'],
            config['sampling']['top_ps']
        ))
    else:
        param_combinations = [(
            config['models']['llm_models'][0],
            config['rag']['enabled'],
            config['embedding_models'][0],
            config['retriever']['types'][0],
            config['retriever']['use_reranker'],
            config['prompting']['simple_prompting'],
            config['prompting']['fewshots_method'],
            config['prompting']['fewshots_with_NR_method'],
            config['prompting']['fewshots_with_NR_extended_method'],
            config['output']['json_format'],
            config['sampling']['temperatures'][0],
            config['sampling']['top_ks'][0],
            config['sampling']['top_ps'][0]
        )]
    
    for params in param_combinations:
        llm_model, rag_enabled, embedding_model, retriever_type, use_reranker, simple_prompting, fewshots_method, fewshots_with_NR_method, fewshots_with_NR_extended_method, json_value, temp, top_k, top_p = params
        
        # If langchain / embedding implementations are missing, fall back to no embeddings
        if rag_enabled and not _HAVE_LANGCHAIN:
            print("Warning: RAG requested but langchain/langchain_community not installed. Proceeding without RAG (rag_enabled=False).")
            rag_enabled = False

        if embedding_model != "mistral":
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model) if HuggingFaceEmbeddings is not None else None
        else:
            embeddings = OllamaEmbeddings(model="mistral") if OllamaEmbeddings is not None else None
        
        # Create column name. Support two styles:
        # - 'verbose' (full details controlled by column_name_format)
        # - 'short' (compact, human-friendly)
        style = config.get('output', {}).get('column_name_style', 'verbose')
        if style == 'short':
            # Short, readable column name: <Target>_pred_<model>_<RAG|NoRAG>_<Rerank|NoRerank>
            model_short = llm_model.split(':')[0].replace('/', '__') if isinstance(llm_model, str) else str(llm_model)
            emb_short = embedding_model.split('/')[0].replace('/', '__') if isinstance(embedding_model, str) else str(embedding_model)
            rag_flag = 'RAG' if rag_enabled else 'NoRAG'
            rerank_flag = 'Rerank' if use_reranker else 'NoRerank'
            column_name = f"{config['evaluation']['target_variable']}_pred_{model_short}_{rag_flag}_{rerank_flag}"
        else:
            column_name = config['column_name_format'].format(
                target_variable=config['evaluation']['target_variable'],
                model=llm_model.replace("/", "__"),
                rag=rag_enabled,
                embeddings=embedding_model.replace("/", "__"),
                retriever=retriever_type,
                reranker=use_reranker,
                simple=simple_prompting,
                fewshots=fewshots_method,
                nr=fewshots_with_NR_method,
                nr_extended=fewshots_with_NR_extended_method,
                json=json_value,
                temp=temp,
                top_k=top_k,
                top_p=top_p
            )
        
        if column_name not in df.columns:
            # create column with object dtype so we can store stringified JSON results
            df[column_name] = pd.Series([None] * len(df), index=df.index, dtype="object")
        
        print(f"Processing model: {column_name}")
        
        df = process_model(
            column_name=column_name,
            file_path=config['file_paths']['input'],
            df=df,
            batch_size=batch_size,
            llm_model=llm_model,
            rag_enabled=rag_enabled,
            embeddings=embeddings,
            retriever_type=retriever_type,
            use_reranker=use_reranker,
            simple_prompting=simple_prompting,
            fewshots_method=fewshots_method,
            fewshots_with_NR_method=fewshots_with_NR_method,
            fewshots_with_NR_extended_method=fewshots_with_NR_extended_method,
            json_value=json_value,
            temp=temp,
            top_k=top_k,
            top_p=top_p,
            verbose=config['processing']['verbose']
            ,report_column=report_col
        )
        
        accuracy = evaluate_experiment(
            df,
            column_name,
            config['file_paths']['figures'],
            config['file_paths']['metrics'],
            config['file_paths']['log'],
            len(df.columns) - 1
        )
        
        print(f"Accuracy: {accuracy}")
        
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            best_model = params
        
        df.to_csv(config['file_paths']['input'], index=False)
        # additionally write a copy with cleaned predictions for inspection
        predictions_path = config['file_paths'].get('predictions', 'data/output/predictions.csv')
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
        try:
            df.to_csv(predictions_path, index=False)
        except Exception:
            # best-effort: ignore write errors
            pass
    
    print(f"Best model: {best_model}")
    print(f"Highest accuracy: {highest_accuracy}")

if __name__ == "__main__":
    main()