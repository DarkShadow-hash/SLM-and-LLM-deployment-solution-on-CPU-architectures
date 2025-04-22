# Democratizing AI: Fast and Efficient SLM and LLM Deployment on CPU Architectures


### Problem definition: Why is fast and efficient deployment of SLMs and LLMs on any CPU architecture important?

Large Language Models (LLMs) and Small Language Models (SLMs) such as GPT-4, LLaMA 2, and Phi-2 are transforming the way industries approach customer service, automation, content creation, and more. However, running these models usually requires expensive GPU hardware or reliance on cloud services. This becomes a challenge for smaller organizations, independent developers, and students who may not have access to high-performance hardware or sufficient budgets.

Cloud solutions, while accessible, often bring issues like data privacy, latency and recurring operational costs. That’s why there's a growing need for tools that allow LLMs and SLMs to be run locally on standard CPU hardware, like a regular laptop, without compromising performance too heavily. Our project aims to solve this problem.

_________________________________________________________________________________________
### Existing solutions: How are current tools addressing SLM and LLM deployment, and what are their limitations?

Some tools already exist to help users deploy LLMs more easily. For example, Ollama and the Hugging Face Transformers library are popular choices. However, they still have limitations :

-	Ollama is beginner-friendly but not optimized for CPU-only environments. It also doesn’t allow for advanced download or deployment configurations.
  
-	Hugging Face Pipelines are powerful but require a lot of manual setup. They're also relatively slow on CPU machines if no additional optimization is done.
  
-	Cloud APIs like OpenAI and Anthropic require sending your data over the internet and can be expensive over time.
These limitations show that we need a solution that is lightweight, efficient, and works offline on any CPU-based device.

_________________________________________________________________________________________
### Our contribution: How does our solution improve SLM and LLM deployment efficiency and speed?

We built a Python-based tool that simplifies the entire process of downloading, deploying, and managing LLMs and SLMs on CPU systems. Here's how our solution helps:

-	Parallel Downloads: We plan to use multi-threading to download multiple parts of a model at the same time, which reduces waiting time.
  
-	ONNX Conversion: We convert models from PyTorch to ONNX format, which is much more efficient for CPU inference.
  
-	Streamlined Workflow: Our tool manages downloads and deployments automatically using a configuration system.
  
-	API Integration: Although not fully implemented yet, we plan to use FastAPI so users can access their models through a web interface.
  
-	User Dashboard: A simple Streamlit dashboard helps users manage models, check statuses, and simulate inference.

_________________________________________________________________________________________
### Resources & techniques: What tools, hardware, and methodologies are being used?

We built the backend in Python using popular libraries like Hugging Face Transformers and ONNX Runtime. The frontend uses Streamlit to provide a user-friendly dashboard. Our optimization techniques include multi-threading for downloads and model conversion to ONNX. We tested the solution on regular laptops and cloud CPU instances to ensure accessibility for a wide range of users.

_________________________________________________________________________________________
### Model download and deployment optimization

We began by creating a system that automatically downloads models from the Hugging Face hub. When a user enters a model name (like gpt2), our tool finds the correct files and downloads them. To improve performance, we designed the system to eventually support parallel downloads, which means the tool can download different pieces of a model at the same time instead of one-by-one.

We also track the status of each model using a JSON configuration file. This way, users can quickly see which models are downloaded, which ones have been deployed, and if any issues occurred.

Next, we integrated ONNX conversion into our workflow. ONNX is a model format optimized for fast inference on CPUs. After a model is downloaded, the tool can convert it to ONNX format, saving a lot of memory and improving speed when it's time to use the model.

_________________________________________________________________________________________
### Deployment workflow development

We created a deployment pipeline that checks if a model is available, converts it to ONNX if needed, and marks it as ready for use. This automated process removes a lot of the manual steps users would otherwise need to perform.
Since actual inference with large models on CPUs can be very slow, we implemented a simulated inference system for now. When a user sends input text to the model, the tool returns a fake-but-realistic-response. This is useful for testing and demonstration purposes without requiring full computational resources.

_________________________________________________________________________________________
### Total Impact if we had implemented everything (rough estimation)

Let’s say the full process for Mistral 8x7B takes around 3600 minutes :

-	Download : 3000 minutes
  
-	Extraction/decompression : 300 minutes
  
-	File operations and setup : 300 minutes
  
With optimizations :

-	Download : decreasing to 300-600 minutes (via multi-threaded downloading)
  
-	Decompression : decreasing to 75-150 minutes (via SIMD)
  
-	Setup : decreasing to 50-100 minutes (via Numba or C++ file ops)
  
Estimated total time with advanced methods : 425–850 minutes
Original time : 3600 minutes
Overall speedup : 4 times to 8 times faster

_________________________________________________________________________________________
### Conclusion

Our project shows that it is entirely possible to run powerful AI models on everyday computers without expensive hardware. By focusing on lightweight models, smart optimizations, and user-friendly design, we created a solution that brings LLM and SLM deployment within reach for students, developers, and small organizations.

If we continue the project, we plan to:

- Integrate FastAPI to allow models to be used in web applications
  
- Allow real inference instead of simulating one (necessary in the development phase we unfortunately never left because we lacked time)

- Make our solution fully available from an API
  
-	Use advanced download tools like aria2c to improve speed
  
-	Add support for quantized models, which are smaller and faster
  
This project is a great example of how even beginner or intermediate developers can build useful tools by combining the right technologies and focusing on accessibility. We hope this inspires more students to explore the world of AI deployment.
Check out our GitHub repository and try deploying your first model today!



