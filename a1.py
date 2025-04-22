import streamlit as st
import requests
import json
import os
import sys
import logging
import pandas as pd
import plotly.express as px
from pathlib import Path
import time
import concurrent.futures
from transformers import AutoModel, AutoTokenizer
import huggingface_hub
import requests.exceptions
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import socket
import urllib.request
import threading
import queue
from tqdm.auto import tqdm
import re
import random
import numpy as np


# Define response generation functions
def generate_llm_response(input_text, model_name):
    """Generate simulated responses mimicking a large language model (LLM).

    Args:
        input_text (str): The input text to generate a response for.
        model_name (str): The name of the model being simulated.

    Returns:
        str: A simulated response based on the input text. Handles math expressions,
             time/date queries, greetings, and general knowledge questions.
    """
    # Check for math expressions
    if re.match(r'^\s*[\d\+\-\*\/\(\)\^\.\s]+\s*$', input_text):
        try:
            # Evaluate basic math expressions
            result = eval(input_text)
            return f"The answer is {result}"
        except:
            return f"Sorry, I couldn't evaluate that math expression. Please check the syntax."
    
    # Check for common questions
    input_lower = input_text.lower()
    
    # Time/date questions
    if "time" in input_lower and "what" in input_lower:
        return f"It's currently {time.strftime('%H:%M:%S')}."
    
    if "date" in input_lower and "what" in input_lower:
        return f"Today is {time.strftime('%A, %B %d, %Y')}."
    
    # General knowledge
    if "who are you" in input_lower:
        return f"I am a simulation of the {model_name} language model, designed to assist with various tasks."
    
    if "hello" in input_lower or "hi" in input_lower:
        greetings = ["Hello!", "Hi there!", "Greetings!", "Hey! How can I help?"]
        return random.choice(greetings)
    
    # Generate a more elaborate response for other queries
    responses = [
        f"Based on my training, I understand '{input_text}' relates to {random.choice(['science', 'technology', 'arts', 'history', 'culture'])}. Let me elaborate on that...",
        f"Interesting question about '{input_text}'. From my knowledge, this involves several aspects worth discussing...",
        f"When considering '{input_text}', I would approach this from multiple perspectives...",
        f"'{input_text}' is a fascinating topic that connects to several key concepts..."
    ]
    return random.choice(responses)

def generate_transformer_response(input_text, model_name):
    """Generate simulated responses mimicking a BERT/transformer model.

    Args:
        input_text (str): The input text to analyze.
        model_name (str): The name of the model being simulated.

    Returns:
        str: A simulated analysis result including sentiment analysis, classification,
             or named entity recognition based on the input text.
    """
    # Sentiment analysis simulation
    if "sentiment" in input_text.lower() or any(word in input_text.lower() for word in ["good", "bad", "happy", "sad", "angry", "like", "love", "hate"]):
        sentiments = ["positive", "negative", "neutral"]
        scores = [round(random.random(), 2) for _ in range(3)]
        total = sum(scores)
        normalized = [round(s/total, 2) for s in scores]
        
        result = "Sentiment Analysis:\n"
        for sentiment, score in zip(sentiments, normalized):
            result += f"- {sentiment.capitalize()}: {score}\n"
        return result
    
    # Classification simulation
    elif "classify" in input_text.lower() or "category" in input_text.lower():
        categories = ["Technology", "Science", "Business", "Arts", "Sports"]
        scores = [round(random.random(), 2) for _ in range(len(categories))]
        total = sum(scores)
        normalized = [round(s/total, 2) for s in scores]
        
        result = "Classification Results:\n"
        for category, score in zip(categories, normalized):
            result += f"- {category}: {score}\n"
        return result
    
    # Named entity recognition simulation
    elif "entity" in input_text.lower() or "identify" in input_text.lower():
        words = input_text.split()
        if len(words) < 3:
            return "Please provide a longer text for entity recognition."
            
        entities = []
        for i, word in enumerate(words):
            if word[0].isupper() and random.random() > 0.7:
                entity_type = random.choice(["PERSON", "ORG", "LOC", "DATE"])
                entities.append((word, entity_type))
                
        if entities:
            result = "Entities Recognized:\n"
            for word, entity_type in entities:
                result += f"- {word}: {entity_type}\n"
            return result
        else:
            return "No entities detected in the provided text."
    
    # Default response
    return f"Analysis complete. {model_name} has processed your input: '{input_text}'."

def generate_general_response(input_text, model_name, model_format):
    """Generate generic responses for other model types.

    Args:
        input_text (str): The input text to process.
        model_name (str): The name of the model being simulated.
        model_format (str): The format of the model (e.g., 'pytorch', 'onnx').

    Returns:
        str: A response including math evaluations, text transformations, or word counts.
    """
    # Check for math expressions first
    if re.match(r'^\s*[\d\+\-\*\/\(\)\^\.\s]+\s*$', input_text):
        try:
            # Evaluate basic math expressions
            result = eval(input_text)
            return f"{input_text} = {result}"
        except:
            return f"Could not evaluate expression: {input_text}"
    
    # Generate a response based on the input length
    words = input_text.split()
    input_length = len(words)
    
    if input_length < 3:
        return f"Input: {input_text}\nOutput: {input_text.upper() if random.random() > 0.5 else input_text.lower()}"
    
    if "?" in input_text:
        answers = ["Yes", "No", "Maybe", "Probably", "Unlikely"]
        return f"Question: {input_text}\nAnswer: {random.choice(answers)}"
    
    # Simple text transformation
    if random.random() > 0.5:
        # Reverse some words
        for _ in range(min(3, len(words))):
            idx = random.randint(0, len(words)-1)
            words[idx] = words[idx][::-1]
        return f"Transformed: {' '.join(words)}"
    else:
        # Generate word counts
        unique_words = set(word.lower() for word in words)
        return f"Analysis: {len(words)} words, {len(unique_words)} unique words in the input."

# Import core functionality from the original script
# Assuming the original script is saved as slm_deployment.py
from slm_deployment import (
    load_config, save_config, download_model, deploy_model, 
    get_inference, convert_to_onnx, DEFAULT_MODELS_DIR,
    SUPPORTED_FORMATS, ORT_OPTIMIZATION_LEVELS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("slm_streamlit.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("slm_streamlit")

# Page configuration
st.set_page_config(
    page_title="SLM Deployment Tool",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .subsection-header {
        font-size: 1.3rem;
        margin-top: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .success-box {
        background-color: #d1e7dd;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .progress-container {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .multi-item {
        background-color: #e9ecef;
        border-radius: 0.3rem;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to get model status color
def get_status_color(status):
    """Return a color associated with a model status.

    Args:
        status (str): The status of the model (e.g., 'deployed', 'downloaded').

    Returns:
        str: A color name (e.g., 'green', 'red') for UI display.
    """
    colors = {
        "deployed": "green",
        "downloaded": "blue",
        "error": "red",
        "unknown": "gray",
        "in_progress": "orange"
    }
    return colors.get(status, "gray")

# Helper function to format bytes
def format_bytes(size):
    """Convert bytes to a human-readable format (KB, MB, GB, etc.).

    Args:
        size (int): The size in bytes.

    Returns:
        str: Formatted string with appropriate unit (e.g., '1.23 MB').
    """
    power = 2**10
    n = 0
    power_labels = {0 : 'B', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
    while size > power:
        size /= power
        n += 1
    return f"{size:.2f} {power_labels[n]}"

# Helper function to get directory size
def get_dir_size(path):
    """Calculate the total size of a directory and its contents recursively.

    Args:
        path (str): The path to the directory.

    Returns:
        int: Total size in bytes.
    """
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total

# Helper function to load system info
def get_system_info():
    """Gather system information including CPU, memory, and platform details.

    Returns:
        dict: A dictionary containing system metrics like CPU count, memory usage, etc.
    """
    import psutil
    return {
        "cpu_count": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(logical=True),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "platform": sys.platform,
        "python_version": sys.version
    }

# Helper function to check internet connection with retries
def check_internet_connection(max_retries=3, retry_delay=2):
    """Check internet connectivity by testing multiple endpoints with retries.

    Args:
        max_retries (int): Maximum number of retry attempts.
        retry_delay (int): Delay between retries in seconds.

    Returns:
        bool: True if internet connection is available, False otherwise.
    """
    for attempt in range(max_retries):
        try:
            # Try multiple endpoints
            endpoints = [
                ("https://huggingface.co", "Hugging Face"),
                ("https://google.com", "Google"),
                ("https://api.github.com", "GitHub")
            ]
            
            for url, name in endpoints:
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        return True
                except requests.exceptions.RequestException:
                    st.warning(f"Could not connect to {name}, trying alternative...")
                    continue
            
            if attempt < max_retries - 1:
                st.info(f"Retrying connection in {retry_delay} seconds...")
                time.sleep(retry_delay)
            
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"Connection attempt {attempt + 1} failed: {str(e)}")
                time.sleep(retry_delay)
            
    return False

# Create a streamlit progress bar that can be updated from a separate thread
class ThreadSafeProgressBar:
    """A thread-safe progress bar for tracking downloads/deployments across threads.

    Attributes:
        total_steps (int): Total steps to completion.
        description (str): Description of the progress bar.
        progress (int): Current progress.
        lock (threading.Lock): Lock for thread-safe updates.
        queue (queue.Queue): Queue for passing updates to the main thread.
    """

    def __init__(self, total_steps, description="Progress"):
        self.total_steps = total_steps
        self.description = description
        self.progress = 0
        self.lock = threading.Lock()
        # Create a queue for passing updates to the main thread
        self.queue = queue.Queue()
        
    def update(self, increment=1):
        with self.lock:
            self.progress += increment
            # Put the new progress in the queue
            self.queue.put(self.progress)
            
    def reset(self):
        with self.lock:
            self.progress = 0
            self.queue.put(self.progress)

# Improved function to check internet connectivity
def check_internet_connectivity():
    """Test internet connection and Hugging Face accessibility"""
    try:
        # First check basic internet connectivity
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        
        # Then check Hugging Face specifically
        response = requests.get("https://huggingface.co", timeout=5)
        return True
    except (socket.error, requests.RequestException):
        return False

# Setup requests session with retries
def setup_requests_session():
    """Create a requests session with retry strategy"""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[408, 429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# Improved download function with progress tracking
def download_model_parallel(model_id, progress_queue=None):
    """Download a model from Hugging Face with progress tracking and error handling.

    Args:
        model_id (str): The model identifier (e.g., 'gpt2').
        progress_queue (queue.Queue, optional): Queue for progress updates.

    Returns:
        tuple: (model, tokenizer, error) where error is None if successful.
    """
    try:
        # Configure proxy if present
        proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY')
        
        # Create directory for the model
        if '/' in model_id:
            model_dir_name = model_id.split('/')[-1]
        else:
            model_dir_name = model_id
            
        model_dir = os.path.join(DEFAULT_MODELS_DIR, model_dir_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Report progress: Starting download
        if progress_queue:
            progress_queue.put(("status", model_id, "Downloading tokenizer..."))
        
        try:
            # Download tokenizer first as it's smaller
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=False,
                use_auth_token=False,
                proxies={'https': proxy} if proxy else None,
                local_files_only=False,
                resume_download=True
            )
            
            # Report progress: Tokenizer downloaded
            if progress_queue:
                progress_queue.put(("status", model_id, "Downloading model..."))
                progress_queue.put(("progress", model_id, 25))
            
            # Download model
            model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=False,
                use_auth_token=False,
                proxies={'https': proxy} if proxy else None,
                local_files_only=False,
                resume_download=True
            )
            
            # Report progress: Model downloaded
            if progress_queue:
                progress_queue.put(("status", model_id, "Saving model..."))
                progress_queue.put(("progress", model_id, 75))
            
            # Save model and tokenizer
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            
            # Update configuration
            config = load_config()
            config["models"][model_id] = {
                "path": model_dir,
                "format": "pytorch",
                "status": "downloaded",
                "tokenizer_path": os.path.join(model_dir, "tokenizer")
            }
            save_config(config)
            
            # Report progress: Download completed
            if progress_queue:
                progress_queue.put(("status", model_id, "Download complete"))
                progress_queue.put(("progress", model_id, 100))
                progress_queue.put(("result", model_id, "success"))
            
            return model, tokenizer, None
            
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Network error while downloading {model_id}: {str(e)}"
            logger.error(error_msg)
            if progress_queue:
                progress_queue.put(("status", model_id, "Error: Network issue"))
                progress_queue.put(("result", model_id, "error", error_msg))
            return None, None, error_msg
            
        except Exception as e:
            error_msg = f"Error downloading model {model_id}: {str(e)}"
            logger.error(error_msg)
            if progress_queue:
                progress_queue.put(("status", model_id, "Error: Download failed"))
                progress_queue.put(("result", model_id, "error", error_msg))
            return None, None, error_msg
            
    except Exception as e:
        error_msg = f"Unexpected error with {model_id}: {str(e)}"
        logger.error(error_msg)
        if progress_queue:
            progress_queue.put(("status", model_id, "Error: Unexpected issue"))
            progress_queue.put(("result", model_id, "error", error_msg))
        return None, None, error_msg

# Function to download multiple models in parallel
def download_multiple_models(model_ids, max_workers=3):
    """Download multiple models in parallel with progress tracking.

    Args:
        model_ids (list): List of model identifiers to download.
        max_workers (int): Maximum parallel downloads.

    Returns:
        dict: A dictionary mapping model IDs to success status (True/False).
    """
    # Create progress tracking queue
    progress_queue = queue.Queue()
    
    # Create a container for progress information in session state
    if "download_progress" not in st.session_state:
        st.session_state.download_progress = {}
    
    # Initialize progress tracking for each model
    for model_id in model_ids:
        st.session_state.download_progress[model_id] = {
            "progress": 0,
            "status": "Preparing...",
            "result": "pending"
        }
    
    # Create and start the download thread
    def download_thread():
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            futures = {
                executor.submit(download_model_parallel, model_id, progress_queue): model_id 
                for model_id in model_ids
            }
            
            # Wait for all tasks to complete
            for future in concurrent.futures.as_completed(futures):
                model_id = futures[future]
                try:
                    model, tokenizer, error = future.result()
                except Exception as e:
                    logger.error(f"Exception for {model_id}: {str(e)}")
    
    # Start the download thread
    thread = threading.Thread(target=download_thread)
    thread.daemon = True
    thread.start()
    
    # Create a placeholder for progress bars
    progress_containers = {}
    for model_id in model_ids:
        progress_containers[model_id] = st.empty()
    
    # Create a placeholder for the completion message
    completion_placeholder = st.empty()
    
    # Monitor the progress queue and update the UI
    update_interval = 0.1  # seconds
    all_completed = False
    
    while not all_completed:
        # Process all available queue items
        try:
            while True:
                msg_type, *msg_data = progress_queue.get(block=False)
                
                if msg_type == "progress":
                    model_id, progress = msg_data
                    st.session_state.download_progress[model_id]["progress"] = progress
                
                elif msg_type == "status":
                    model_id, status = msg_data
                    st.session_state.download_progress[model_id]["status"] = status
                
                elif msg_type == "result":
                    if len(msg_data) == 2:
                        model_id, result = msg_data
                        st.session_state.download_progress[model_id]["result"] = result
                    else:
                        model_id, result, error_msg = msg_data
                        st.session_state.download_progress[model_id]["result"] = result
                        st.session_state.download_progress[model_id]["error"] = error_msg
                
                progress_queue.task_done()
        
        except queue.Empty:
            pass
        
        # Update all progress bars
        for model_id, container in progress_containers.items():
            progress_data = st.session_state.download_progress[model_id]
            
            # Render the progress bar and status
            with container.container():
                st.markdown(f"<div class='multi-item'>", unsafe_allow_html=True)
                st.write(f"**Model:** {model_id}")
                st.write(f"**Status:** {progress_data['status']}")
                st.progress(progress_data["progress"] / 100)
                
                if progress_data["result"] == "error" and "error" in progress_data:
                    st.error(progress_data["error"])
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Check if all downloads are completed
        all_completed = all(
            progress_data["result"] in ["success", "error"] 
            for progress_data in st.session_state.download_progress.values()
        )
        
        if not all_completed:
            time.sleep(update_interval)
    
    # Show completion message
    success_count = sum(1 for progress_data in st.session_state.download_progress.values() 
                      if progress_data["result"] == "success")
    
    if success_count == len(model_ids):
        completion_placeholder.success(f"✅ All {len(model_ids)} models downloaded successfully!")
    else:
        completion_placeholder.warning(f"⚠️ {success_count} of {len(model_ids)} models downloaded successfully. Check errors above.")
    
    # Return success status for each model
    return {
        model_id: progress_data["result"] == "success"
        for model_id, progress_data in st.session_state.download_progress.items()
    }

# Function to deploy a model with progress reporting
def deploy_model_with_progress(model_id, format, progress_queue=None):
    """Deploy a model with progress tracking and ONNX conversion if needed.

    Args:
        model_id (str): The model identifier.
        format (str): Deployment format ('pytorch' or 'onnx').
        progress_queue (queue.Queue, optional): Queue for progress updates.

    Returns:
        tuple: (model_info, error) where error is None if successful.
    """

    try:
        # If ONNX is selected but not yet converted, convert first
        config = load_config()
        
        if format == "onnx" and not config["models"][model_id].get("onnx_path"):
            if progress_queue:
                progress_queue.put(("status", model_id, "Converting to ONNX..."))
                progress_queue.put(("progress", model_id, 25))
            
            optimization_level = config["settings"]["optimization_level"]
            convert_to_onnx(model_id, optimization_level)
            config = load_config()  # Reload config
            
            if progress_queue:
                progress_queue.put(("progress", model_id, 50))
        
        # Report progress: Starting deployment
        if progress_queue:
            progress_queue.put(("status", model_id, f"Deploying in {format} format..."))
            progress_queue.put(("progress", model_id, 75))
        
        # Deploy the model
        model_info = deploy_model(model_id, format)
        
        # Report progress: Deployment completed
        if progress_queue:
            progress_queue.put(("status", model_id, "Deployment complete"))
            progress_queue.put(("progress", model_id, 100))
            progress_queue.put(("result", model_id, "success"))
        
        return model_info, None
        
    except Exception as e:
        error_msg = f"Error deploying model {model_id}: {str(e)}"
        logger.error(error_msg)
        if progress_queue:
            progress_queue.put(("status", model_id, "Error: Deployment failed"))
            progress_queue.put(("result", model_id, "error", error_msg))
        return None, error_msg

# Function to deploy multiple models in parallel
def deploy_multiple_models(model_configs, max_workers=3):
    """Deploy multiple models in parallel with progress tracking.

    Args:
        model_configs (list): List of (model_id, format) tuples.
        max_workers (int): Maximum parallel deployments.

    Returns:
        dict: A dictionary mapping model IDs to success status (True/False).
    """
    # Create progress tracking queue
    progress_queue = queue.Queue()
    
    # Create a container for progress information in session state
    if "deployment_progress" not in st.session_state:
        st.session_state.deployment_progress = {}
    
    # Initialize progress tracking for each model
    for model_id, format in model_configs:
        st.session_state.deployment_progress[model_id] = {
            "progress": 0,
            "status": "Preparing...",
            "result": "pending"
        }
    
    # Create and start the deployment thread
    def deployment_thread():
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all deployment tasks
            futures = {
                executor.submit(deploy_model_with_progress, model_id, format, progress_queue): model_id 
                for model_id, format in model_configs
            }
            
            # Wait for all tasks to complete
            for future in concurrent.futures.as_completed(futures):
                model_id = futures[future]
                try:
                    model_info, error = future.result()
                except Exception as e:
                    logger.error(f"Exception for {model_id}: {str(e)}")
    
    # Start the deployment thread
    thread = threading.Thread(target=deployment_thread)
    thread.daemon = True
    thread.start()
    
    # Create a placeholder for progress bars
    progress_containers = {}
    for model_id, _ in model_configs:
        progress_containers[model_id] = st.empty()
    
    # Create a placeholder for the completion message
    completion_placeholder = st.empty()
    
    # Monitor the progress queue and update the UI
    update_interval = 0.1  # seconds
    all_completed = False
    
    while not all_completed:
        # Process all available queue items
        try:
            while True:
                msg_type, *msg_data = progress_queue.get(block=False)
                
                if msg_type == "progress":
                    model_id, progress = msg_data
                    st.session_state.deployment_progress[model_id]["progress"] = progress
                
                elif msg_type == "status":
                    model_id, status = msg_data
                    st.session_state.deployment_progress[model_id]["status"] = status
                
                elif msg_type == "result":
                    if len(msg_data) == 2:
                        model_id, result = msg_data
                        st.session_state.deployment_progress[model_id]["result"] = result
                    else:
                        model_id, result, error_msg = msg_data
                        st.session_state.deployment_progress[model_id]["result"] = result
                        st.session_state.deployment_progress[model_id]["error"] = error_msg
                
                progress_queue.task_done()
        
        except queue.Empty:
            pass
        
        # Update all progress bars
        for model_id, container in progress_containers.items():
            progress_data = st.session_state.deployment_progress[model_id]
            
            # Render the progress bar and status
            with container.container():
                st.markdown(f"<div class='multi-item'>", unsafe_allow_html=True)
                st.write(f"**Model:** {model_id}")
                st.write(f"**Status:** {progress_data['status']}")
                st.progress(progress_data["progress"] / 100)
                
                if progress_data["result"] == "error" and "error" in progress_data:
                    st.error(progress_data["error"])
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Check if all deployments are completed
        all_completed = all(
            progress_data["result"] in ["success", "error"] 
            for progress_data in st.session_state.deployment_progress.values()
        )
        
        if not all_completed:
            time.sleep(update_interval)
    
    # Show completion message
    success_count = sum(1 for progress_data in st.session_state.deployment_progress.values() 
                      if progress_data["result"] == "success")
    
    if success_count == len(model_configs):
        completion_placeholder.success(f"✅ All {len(model_configs)} models deployed successfully!")
    else:
        completion_placeholder.warning(f"⚠️ {success_count} of {len(model_configs)} models deployed successfully. Check errors above.")
    
    # Return success status for each model
    return {
        model_id: st.session_state.deployment_progress[model_id]["result"] == "success"
        for model_id, _ in model_configs
    }

# Add API configuration
API_CONFIG = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "endpoints": {
            "chat": "/chat/completions",
            "completion": "/completions"
        }
    },
    "huggingface": {
        "base_url": "https://api-inference.huggingface.co/models",
        "endpoints": {
            "inference": ""
        }
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "endpoints": {
            "messages": "/messages"
        }
    }
}

# Initialize session state variables
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.inference_history = []
    st.session_state.api_keys = {}
    st.session_state.download_progress = {}
    st.session_state.deployment_progress = {}
    st.session_state.error_log = []

def handle_api_error(error, context="API"):
    """Handle API errors by logging and displaying user-friendly messages.

    Args:
        error (Exception): The error encountered.
        context (str): Context for the error (e.g., 'OpenAI').

    Returns:
        None
    """
    error_msg = str(error)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    if isinstance(error, requests.exceptions.ConnectionError):
        user_msg = f"Connection error: Please check your internet connection"
    elif isinstance(error, requests.exceptions.Timeout):
        user_msg = f"Request timed out: The {context} server is taking too long to respond"
    elif isinstance(error, requests.exceptions.RequestException):
        user_msg = f"API error: {error_msg}"
    else:
        user_msg = f"Unexpected error: {error_msg}"
    
    # Log error
    st.session_state.error_log.append({
        "timestamp": timestamp,
        "context": context,
        "error": error_msg
    })
    
    # Display error to user
    st.error(user_msg)
    return None

def safe_api_call(api_func, *args, **kwargs):
    """Wrapper for API calls with error handling.

    Args:
        api_func (function): The API function to call.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        The result of api_func or None if an error occurs.
    """
    try:
        return api_func(*args, **kwargs)
    except Exception as e:
        return handle_api_error(e)

# Update API integration functions with better error handling
def call_openai_api(input_text, model_name, api_key):
    """Call the OpenAI API for text generation.

    Args:
        input_text (str): The input prompt.
        model_name (str): The OpenAI model to use.
        api_key (str): The API key for authentication.

    Returns:
        str: The generated text or an error message.
    """

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": input_text}],
        "temperature": 0.7,
        "max_tokens": 150
    }
    
    try:
        with st.spinner("Calling OpenAI API..."):
            response = requests.post(
                f"{API_CONFIG['openai']['base_url']}{API_CONFIG['openai']['endpoints']['chat']}",
                headers=headers,
                json=data,
                timeout=(5, 30)
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return handle_api_error(e, "OpenAI")

def call_huggingface_api(input_text, model_name, api_key):
    """Call the Hugging Face API for model inference.

    Args:
        input_text (str): The input text.
        model_name (str): The Hugging Face model ID.
        api_key (str): The API key for authentication.

    Returns:
        dict: The API response or an error message.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "inputs": input_text
    }
    
    try:
        response = requests.post(
            f"{API_CONFIG['huggingface']['base_url']}/{model_name}",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return f"API Error: {str(e)}"

def call_anthropic_api(input_text, model_name, api_key):
    """Call the Anthropic API for text generation.

    Args:
        input_text (str): The input prompt.
        model_name (str): The Anthropic model to use.
        api_key (str): The API key for authentication.

    Returns:
        str: The generated text or an error message.
    """

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": input_text}],
        "max_tokens": 150
    }
    
    try:
        response = requests.post(
            f"{API_CONFIG['anthropic']['base_url']}{API_CONFIG['anthropic']['endpoints']['messages']}",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]
    except Exception as e:
        return f"API Error: {str(e)}"
    

""""This part of the code focuses on setting up the streamlit interface for users. 
This is the concrete front end part
""" 

# Sidebar for navigation
st.sidebar.markdown("# 🤖 SLM Deployment")
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Download Models", "Deploy Models", "Inference", "Settings", "System Info"]
)

# Load current configuration
config = load_config()

# Dashboard page
if page == "Dashboard":
    st.markdown("<h1 class='main-header'>SLM Deployment Dashboard</h1>", unsafe_allow_html=True)
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Models", len(config["models"]))
    with col2:
        deployed_count = sum(1 for model in config["models"].values() if model.get("status") == "deployed")
        st.metric("Deployed Models", deployed_count)
    with col3:
        sys_info = get_system_info()
        st.metric("CPU Cores", sys_info["cpu_count"])
    with col4:
        st.metric("RAM Available", f"{sys_info['memory_available_gb']} GB")
    
    # Models table
    st.markdown("<h2 class='section-header'>Models</h2>", unsafe_allow_html=True)
    
    if not config["models"]:
        st.markdown("<div class='info-box'>No models available. Use the 'Download Models' page to get started.</div>", unsafe_allow_html=True)
    else:
        # Create DataFrame for models
        models_data = []
        for model_id, model_info in config["models"].items():
            model_path = model_info.get("path", "")
            model_size = get_dir_size(model_path) if os.path.exists(model_path) else 0
            
            models_data.append({
                "Model ID": model_id,
                "Status": model_info.get("status", "unknown"),
                "Format": model_info.get("format", "unknown"),
                "ONNX Available": "Yes" if model_info.get("onnx_path") else "No",
                "Size": format_bytes(model_size)
            })
        
        models_df = pd.DataFrame(models_data)
        
        # Display as styled table
        st.dataframe(
            models_df.style.applymap(
                lambda x: f"color: {get_status_color(x)}", 
                subset=["Status"]
            ),
            use_container_width=True
        )
    
    # Storage usage chart
    st.markdown("<h2 class='section-header'>Storage Usage</h2>", unsafe_allow_html=True)
    
    if config["models"]:
        storage_data = []
        for model_id, model_info in config["models"].items():
            model_path = model_info.get("path", "")
            if os.path.exists(model_path):
                model_size = get_dir_size(model_path)
                storage_data.append({"Model": model_id, "Size (GB)": model_size / (1024**3)})
        
        if storage_data:
            storage_df = pd.DataFrame(storage_data)
            fig = px.pie(storage_df, names="Model", values="Size (GB)", title="Model Storage Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No storage data available")
    else:
        st.info("No models downloaded yet")

# Download Models page
elif page == "Download Models":
    st.markdown("<h1 class='main-header'>Download Models</h1>", unsafe_allow_html=True)
    
    # Add proxy configuration
    with st.expander("Network Settings"):
        proxy = st.text_input(
            "Proxy URL (optional)", 
            placeholder="http://your-proxy:port",
            help="If behind a corporate firewall, enter your proxy URL"
        )
        if proxy:
            os.environ['HTTPS_PROXY'] = proxy
            os.environ['HTTP_PROXY'] = proxy
    
    # Example models section
    st.markdown("### Recommended Models")
    example_models = [
        "gpt2",  # Very small model
        "distilbert-base-uncased",
        "bert-base-uncased",
        "facebook/opt-125m",
        "microsoft/DialoGPT-small"
    ]
    
    # Multi-download section
    st.markdown("<h2 class='section-header'>Multiple Model Download</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        Select multiple models to download in parallel to save time.
    </div>
    """, unsafe_allow_html=True)
    
    # Add selection mechanism for multiple models
    selected_models = st.multiselect(
        "Select models to download", 
        example_models,
        help="Select up to 3 models to download simultaneously"
    )
    
    # Option to add custom models
    add_custom = st.checkbox("Add custom model")
    
    custom_models = []
    if add_custom:
        custom_model = st.text_input(
            "Enter custom model ID",
            placeholder="e.g., organization/model-name"
        )
        if custom_model:
            custom_models.append(custom_model)
            
        # Option to add another custom model
        add_another = st.checkbox("Add another custom model")
        if add_another:
            custom_model2 = st.text_input(
                "Enter second custom model ID",
                placeholder="e.g., organization/model-name",
                key="custom_model2"
            )
            if custom_model2:
                custom_models.append(custom_model2)
    
    # Combine selected and custom models
    download_models = selected_models + custom_models
    
    # Controls for parallel downloads
    col1, col2 = st.columns(2)
    with col1:
        max_workers = st.slider(
            "Max parallel downloads", 
            1, 5, 3,
            help="Increasing this may speed up downloads but use more bandwidth and memory"
        )
    
    with col2:
        st.info(f"Selected {len(download_models)} models for download")
    
    # Download button
    if st.button("Download Selected Models", disabled=len(download_models) == 0):
        if download_models:
            st.markdown("<h3 class='subsection-header'>Download Progress</h3>", unsafe_allow_html=True)
            # Download models in parallel
            results = download_multiple_models(download_models, max_workers=max_workers)
            
            # Refresh config after downloads
            config = load_config()

# Deploy Models page
elif page == "Deploy Models":
    st.markdown("<h1 class='main-header'>Deploy Models</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        Deploy downloaded models for inference. Select multiple models to deploy in parallel.
    </div>
    """, unsafe_allow_html=True)
    
    # Check if there are any models to deploy
    if not config["models"]:
        st.warning("No models available for deployment. Please download a model first from the Download Models page.")
    else:
        # Multi-deploy section
        st.markdown("<h2 class='section-header'>Multi-Model Deployment</h2>", unsafe_allow_html=True)
        
        # Get list of downloaded models
        downloaded_models = {
            model_id: model_info 
            for model_id, model_info in config["models"].items() 
            if model_info.get("status") == "downloaded"
        }
        
        if not downloaded_models:
            st.warning("No models available for deployment. All models are already deployed or need to be downloaded first.")
        else:
            # Create form for multi-deployment
            with st.form("multi_deploy_form"):
                # Get default format from settings
                default_format = config["settings"]["default_format"]
                
                # Select multiple models
                selected_models = st.multiselect(
                    "Select models to deploy", 
                    list(downloaded_models.keys()),
                    help="Select multiple models to deploy in parallel"
                )
                
                # Format selection (applies to all selected models)
                format = st.selectbox(
                    "Deployment Format for All Selected Models", 
                    SUPPORTED_FORMATS,
                    index=SUPPORTED_FORMATS.index(default_format) if default_format in SUPPORTED_FORMATS else 0
                )
                
                # ONNX specific options
                show_onnx_options = format == "onnx"
                if show_onnx_options:
                    st.markdown("### ONNX Optimization Options")
                    
                    # Define optimization level descriptions
                    optimization_descriptions = [
                        "Level 0: Basic optimizations",
                        "Level 1: Standard optimizations",
                        "Level 2: Extended optimizations",
                        "Level 3: Maximum optimizations"
                    ]
                    
                    onnx_optimization_level = st.select_slider(
                        "Optimization Level",
                        options=list(range(len(optimization_descriptions))),
                        value=config["settings"].get("optimization_level", 0),
                        format_func=lambda x: optimization_descriptions[x]
                    )
                
                # Submit button for the form
                submit_button = st.form_submit_button("Deploy Selected Models")
                
                # Process form submission
                if submit_button and selected_models:
                    st.markdown("<h3 class='subsection-header'>Deployment Progress</h3>", unsafe_allow_html=True)
                    # Create list of (model_id, format) tuples for deployment
                    deploy_configs = [(model_id, format) for model_id in selected_models]
                    # Deploy models in parallel
                    results = deploy_multiple_models(deploy_configs)
                    # Refresh config after deployments
                    config = load_config()

# Inference page
elif page == "Inference":
    st.markdown("<h1 class='main-header'>Model Inference</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        Test deployed models with your own inputs or use external APIs.
    </div>
    """, unsafe_allow_html=True)
    
    # Add API configuration section
    with st.expander("API Configuration"):
        api_provider = st.selectbox(
            "Select API Provider",
            ["None", "OpenAI", "Hugging Face", "Anthropic"]
        )
        
        if api_provider != "None":
            api_key = st.text_input(
                f"{api_provider} API Key",
                type="password",
                help="Enter your API key for the selected provider"
            )
            
            if api_provider == "OpenAI":
                model_name = st.selectbox(
                    "OpenAI Model",
                    ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]
                )
            elif api_provider == "Hugging Face":
                model_name = st.text_input(
                    "Hugging Face Model ID",
                    placeholder="e.g., meta-llama/Llama-2-7b-chat-hf"
                )
            elif api_provider == "Anthropic":
                model_name = st.selectbox(
                    "Anthropic Model",
                    ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-2.1"]
                )
    
    # Get deployed models
    deployed_models = {
        model_id: model_info 
        for model_id, model_info in config["models"].items() 
        if model_info.get("status") == "deployed"
    }
    
    if not deployed_models and api_provider == "None":
        st.warning("No deployed models available. Please deploy a model first from the Deploy Models page or configure an API provider.")
    else:
        # Create columns for input and output
        input_col, output_col = st.columns([1, 1])
        
        with input_col:
            st.markdown("<h2 class='section-header'>Input</h2>", unsafe_allow_html=True)
            
            # Model selection (only if not using API)
            if api_provider == "None":
                selected_model = st.selectbox(
                    "Select model for inference",
                    list(deployed_models.keys())
                )
                model_format = deployed_models[selected_model].get("format", "pytorch")
                st.info(f"Model format: {model_format}")
            
            # Input text
            input_text = st.text_area(
                "Input text",
                "Hello, how are you?",
                height=150
            )
            
            # Parameters
            with st.expander("Inference Parameters"):
                max_tokens = st.slider("Max Output Tokens", 1, 100, 20)
                temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
            
            # Run inference button
            run_button = st.button("Run Inference", use_container_width=True, type="primary")
        
        # Output section
        with output_col:
            st.markdown("<h2 class='section-header'>Output</h2>", unsafe_allow_html=True)
            
            # Create a placeholder for results
            result_placeholder = st.empty()
            
            # Store API keys in session state
            if api_provider != "None" and api_key:
                st.session_state.api_keys[api_provider] = api_key
            
            # Run inference with proper error handling
            if run_button:
                try:
                    with st.spinner("Running inference..."):
                        if api_provider != "None":
                            api_key = st.session_state.api_keys.get(api_provider)
                            if not api_key:
                                st.error(f"Please provide an API key for {api_provider}")
                            else:
                                if api_provider == "OpenAI":
                                    output_text = safe_api_call(call_openai_api, input_text, model_name, api_key)
                                elif api_provider == "Hugging Face":
                                    output_text = safe_api_call(call_huggingface_api, input_text, model_name, api_key)
                                elif api_provider == "Anthropic":
                                    output_text = safe_api_call(call_anthropic_api, input_text, model_name, api_key)
                        else:
                            output_text = safe_api_call(get_inference, selected_model, input_text, max_tokens, temperature)
                        
                        if output_text:
                            # Update inference history
                            st.session_state.inference_history.append({
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "model": model_name if api_provider != "None" else selected_model,
                                "provider": api_provider if api_provider != "None" else "local",
                                "input": input_text,
                                "output": output_text
                            })
                            
                            # Display result
                            with result_placeholder.container():
                                st.success("Inference completed successfully!")
                                st.markdown("### Generated Text")
                                st.write(output_text)
                                
                except Exception as e:
                    handle_api_error(e, "Inference")
        
        # History section
        st.markdown("<h2 class='section-header'>Inference History</h2>", unsafe_allow_html=True)
        
        if "inference_history" in st.session_state and st.session_state.inference_history:
            history = st.session_state.inference_history
            
            # Display the last 5 inferences
            for i, item in enumerate(reversed(history[-5:])):
                with st.expander(f"Inference {len(history)-i} - {item['timestamp']}"):
                    st.markdown(f"**Provider:** {item['provider']}")
                    st.markdown(f"**Model:** {item['model']}")
                    st.markdown("**Input:**")
                    st.markdown(f"<div class='info-box'>{item['input']}</div>", unsafe_allow_html=True)
                    st.markdown("**Output:**")
                    st.markdown(f"<div class='success-box'>{item['output']}</div>", unsafe_allow_html=True)
            
            # Button to clear history
            if st.button("Clear History"):
                st.session_state.inference_history = []
                st.experimental_rerun()
        else:
            st.info("No inference history yet. Run some inferences to see them here.")

# Settings page
elif page == "Settings":
    st.markdown("<h1 class='main-header'>Settings</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        Configure application settings and defaults.
    </div>
    """, unsafe_allow_html=True)
    
    # Create form for settings
    with st.form("settings_form"):
        st.markdown("<h2 class='section-header'>General Settings</h2>", unsafe_allow_html=True)
        
        # Default format
        default_format = st.selectbox(
            "Default Deployment Format",
            SUPPORTED_FORMATS,
            index=SUPPORTED_FORMATS.index(config["settings"].get("default_format", "pytorch")) 
            if "default_format" in config["settings"] and config["settings"]["default_format"] in SUPPORTED_FORMATS 
            else 0
        )
        
        # Models directory
        models_dir = st.text_input(
            "Models Directory",
            config["settings"].get("models_dir", DEFAULT_MODELS_DIR)
        )
        
        # Optimization level for ONNX
        st.markdown("<h2 class='section-header'>ONNX Settings</h2>", unsafe_allow_html=True)
        
        # Define optimization level descriptions
        optimization_descriptions = [
            "Level 0: Basic optimizations",
            "Level 1: Standard optimizations",
            "Level 2: Extended optimizations",
            "Level 3: Maximum optimizations"
        ]
        
        optimization_level = st.select_slider(
            "ONNX Optimization Level",
            options=list(range(len(optimization_descriptions))),
            value=config["settings"].get("optimization_level", 0),
            format_func=lambda x: optimization_descriptions[x]
        )
        
        # Parallel processing settings
        st.markdown("<h2 class='section-header'>Performance Settings</h2>", unsafe_allow_html=True)
        max_workers = st.slider(
            "Max Parallel Workers",
            1, 8, config["settings"].get("max_workers", 3),
            help="Maximum number of parallel workers for downloads and deployments"
        )
        
        # Cache settings
        cache_enabled = st.checkbox(
            "Enable Result Caching",
            config["settings"].get("cache_enabled", True),
            help="Cache inference results to improve performance"
        )
        
        # Submit button
        submitted = st.form_submit_button("Save Settings")
        
        if submitted:
            # Update settings
            config["settings"]["default_format"] = default_format
            config["settings"]["models_dir"] = models_dir
            config["settings"]["optimization_level"] = optimization_level
            config["settings"]["max_workers"] = max_workers
            config["settings"]["cache_enabled"] = cache_enabled
            
            # Save settings
            save_config(config)
            st.success("Settings saved successfully!")

# System Info page
elif page == "System Info":
    st.markdown("<h1 class='main-header'>System Information</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        View system information and application metrics.
    </div>
    """, unsafe_allow_html=True)
    
    # Get system info
    sys_info = get_system_info()
    
    # System specifications
    st.markdown("<h2 class='section-header'>System Specifications</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='multi-item'>", unsafe_allow_html=True)
        st.markdown("### CPU")
        st.write(f"**Physical cores:** {sys_info['cpu_count']}")
        st.write(f"**Logical processors:** {sys_info['logical_cpus']}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='multi-item'>", unsafe_allow_html=True)
        st.markdown("### Platform")
        st.write(f"**OS:** {sys_info['platform']}")
        st.write(f"**Python version:** {sys_info['python_version']}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='multi-item'>", unsafe_allow_html=True)
        st.markdown("### Memory")
        st.write(f"**Total:** {sys_info['memory_total_gb']} GB")
        st.write(f"**Available:** {sys_info['memory_available_gb']} GB")
        st.write(f"**Usage:** {(1 - sys_info['memory_available_gb']/sys_info['memory_total_gb'])*100:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Network check
    st.markdown("<h2 class='section-header'>Network Status</h2>", unsafe_allow_html=True)
    
    if st.button("Check Internet Connection"):
        with st.spinner("Checking connection..."):
            is_connected = check_internet_connection()
            
        if is_connected:
            st.success("✅ Connected to the internet. HuggingFace is accessible.")
        else:
            st.error("❌ Internet connection issues detected. HuggingFace might not be accessible.")
    
    # Models storage info
    st.markdown("<h2 class='section-header'>Models Storage</h2>", unsafe_allow_html=True)
    
    # Calculate total and used storage
    models_dir = config["settings"].get("models_dir", DEFAULT_MODELS_DIR)
    if os.path.exists(models_dir):
        total_size = get_dir_size(models_dir)
        
        # Display storage info
        st.write(f"**Models directory:** {models_dir}")
        st.write(f"**Total storage used:** {format_bytes(total_size)}")
        
        # Per model breakdown
        if config["models"]:
            st.markdown("### Storage per Model")
            
            # Create data for bar chart
            models_data = []
            for model_id, model_info in config["models"].items():
                model_path = model_info.get("path", "")
                if os.path.exists(model_path):
                    size = get_dir_size(model_path)
                    models_data.append({"Model": model_id, "Size (MB)": size / (1024**2)})
            
            if models_data:
                models_df = pd.DataFrame(models_data)
                fig = px.bar(models_df, x="Model", y="Size (MB)", title="Model Size Comparison")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Models directory {models_dir} does not exist")
    
    # Application logs
    st.markdown("<h2 class='section-header'>Application Logs</h2>", unsafe_allow_html=True)
    
    log_file = "slm_streamlit.log"
    if os.path.exists(log_file):
        # Show last few lines of log
        with open(log_file, "r") as f:
            log_lines = f.readlines()
            # Display last 10 lines or fewer if not enough
            last_logs = log_lines[-10:] if len(log_lines) > 10 else log_lines
            log_text = "".join(last_logs)
        
        st.code(log_text, language="text")
        
        if st.button("View Full Log"):
            st.code("".join(log_lines), language="text")
    else:
        st.info(f"Log file {log_file} not found")

    # Error Log section
    st.markdown("<h2 class='section-header'>Error Log</h2>", unsafe_allow_html=True)
    
    if st.session_state.error_log:
        with st.expander("View Error Log"):
            for error in reversed(st.session_state.error_log):
                st.markdown(f"""
                **Time:** {error['timestamp']}  
                **Context:** {error['context']}  
                **Error:** {error['error']}  
                ---
                """)
        
        if st.button("Clear Error Log"):
            st.session_state.error_log = []
            st.experimental_rerun()
    else:
        st.info("No errors logged")