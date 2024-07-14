

# Openvino_LLM_Chatbot

The OpenVINO LLM Chatbot is an advanced conversational AI application designed to leverage the power of large language models (LLMs) optimized for Intel® CPUs using the Intel® Distribution of OpenVINO™ toolkit. This chatbot aims to provide efficient and scalable natural language processing (NLP) capabilities on CPU-based systems, reducing dependency on expensive GPU infrastructure.


## Components

| Component | Description |
|-----------|-------------|
| LLM Integration and model for inference | Selecting TinyLlama model as a LLM model and also used for inference |
| OpenVINO Optimization | Using Intel's OpenVINO toolkit for optimized model inference |
| generating text in a causal language modeling manner | Uses distilgpt2 model for generating text in a causal language modeling manner|
| User Interface | Implements a Gradio interface for easy interaction |

## Programs / Files

### 1. Prerequisites

Install all the required dependencies:
If openvino is already install then no need to run it.


```bash
  Prerequisites.ipynb
```

### 2. Huggingface login

For models requiring license agreements, users need to log in to the Hugging Face Hub and accept the terms to access the pretrained models. Users can log in using provided code snippets to obtain access tokens.


```bash
  Huggingface_login.ipynb
```

### 3. model_selection

This tutorial provides an overview of various open-source large language models (LLMs) and their specifications, aimed at comparing their quality and usability. Here is a summary of the selected model:
1. tiny-llama-1b-chat:
   - A compact, chat-oriented model with 1.1 billion parameters.
   - Based on the Llama 2 architecture, fine-tuned for chat applications.
   - Suitable for use cases with restricted computational resources.

```bash
  model_selection.ipynb
```

### 4. converting_model_using_optimum_CLI

Optimum Intel serves as an interface between the  Transformers and Diffusers libraries and OpenVINO, enhancing end-to-end pipelines on Intel architectures. It offers a user-friendly CLI for exporting models to OpenVINO Intermediate Representation (IR) format. The Weights Compression algorithm optimizes the footprint and performance of large models, such as LLMs, by compressing their weights. INT4 compression further boosts performance compared to INT8, with only a minor reduction in prediction quality.


```bash
  converting_model_using_optimum_CLI.ipynb
```

### 5. select device for inference



```bash
  select_device_for_inference.ipynb
```

### 6. Instantiate Model using Optimum Intel
Optimum Intel allows loading optimized models from the Hugging Face Hub and creating pipelines for inference with OpenVINO Runtime using Hugging Face APIs. The Optimum Inference models are API compatible with Hugging Face Transformers models, requiring only a switch from AutoModelForXxx to OVModelForXxx. To initialize the model, use the from_pretrained method with export=True if needed. The converted model can be saved with the save_pretrained method. The Tokenizer class and pipelines API are compatible with Optimum models. More details are available in the OpenVINO LLM inference guide.

•	Here we use distilgpt2 model.



```bash
  Instantiate_Model_using_Optimum_Intel.ipynb
```

### 7. Openvino Chatbot Full Code

Complete Code in .py format

```bash
  OpenvinoChatbotFullCode.py
```

## How to Run
1. Open the notebook in Google Colab.
2. Run all cells in order and add one extra cell for user interface.
3. Ask any questions using the Gradio interface.

# Future Improvements

- Can use more powerful language models
- Support for generating answer with image
- Support for generating conversation history


## Team Members and their contributions

This project was developed by a team of 4 members as part of the Intel Unnati Industrial Training 2024:

- Subhadip Ghosh & Jitu Pradhan: Project lead-(Subhadip Ghosh), LLM integration , model evaluation , OpenVINO optimization, instantiate model.
- Akash Ghosh:  Architecture design and Documentation
- Sumita Das: User interface development and integration.

We would like to thank  Intel Unnati program for their guidance and support.
