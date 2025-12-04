# medical_model.py
import os
import torch
import logging
from typing import Dict, Any, Tuple, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline,
    Pipeline
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    "biogpt": {
        "path": "microsoft/biogpt-large-pubmedqa",
        "type": "causal",
        "max_length": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "min_length": 50,
    },
    "flan-t5": {
        "path": "google/flan-t5-base",
        "type": "seq2seq",
        "max_length": 512,
        "temperature": 0.3,
        "min_length": 30,
    },
    "flan-t5-base": {
        "path": "google/flan-t5-base",
        "type": "seq2seq",
        "max_length": 512,
        "temperature": 0.5,
        "min_length": 30,
    }
}

# Global model cache
_model_cache: Dict[str, Any] = {}

def get_device() -> int:
    """Get the appropriate device for model inference."""
    return 0 if torch.cuda.is_available() else -1

def load_medical_model(model_name: str = "flan-t5-base") -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Load the medical model with error handling and fallback.
    
    Args:
        model_name: Name of the model configuration to use
        
    Returns:
        Tuple of (pipeline, config)
    """
    # Validate model name
    if model_name not in MODEL_CONFIGS:
        logger.warning(f"Model {model_name} not found, using flan-t5-base")
        model_name = "flan-t5-base"
    
    # Return from cache if available
    if model_name in _model_cache:
        return _model_cache[model_name]["pipeline"], MODEL_CONFIGS[model_name]
    
    config = MODEL_CONFIGS[model_name]
    model_path = config["path"]
    
    try:
        logger.info(f"Loading medical model: {model_path}")
        
        # Clear CUDA cache to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        device_id = get_device()
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if config["type"] == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device_id
            )
        else:  # causal
            model = AutoModelForCausalLM.from_pretrained(model_path)
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device_id
            )
        
        # Update cache
        _model_cache[model_name] = {
            "pipeline": pipe,
            "tokenizer": tokenizer,
            "model": model
        }
        
        logger.info(f"Successfully loaded model: {model_path}")
        return pipe, config
        
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        if model_name != "flan-t5-base":
            logger.info("Falling back to default model")
            return load_medical_model("default")
        raise RuntimeError("Failed to load any medical model. Please check the logs for details.")

def generate_medical_answer(
    question: str,
    max_length: Optional[int] = None,
    model_name: str = "default"
) -> str:
    """
    Generate a medical answer for the given question.
    
    Args:
        question: The medical question to answer
        max_length: Maximum length of the generated response
        model_name: Name of the model configuration to use
        
    Returns:
        Generated medical answer as a string
    """
    try:
        # Load the model pipeline
        pipe, config = load_medical_model(model_name)
        
        # Prepare the prompt based on model type
        if config["type"] == "causal":
            prompt = (
                "You are a helpful, knowledgeable, and precise medical assistant. "
                "Provide accurate, evidence-based medical information in a clear and concise manner.\n\n"
                f"Question: {question}\n\n"
                "Answer: [Provide a detailed, evidence-based response. If uncertain, clearly state that. "
                "If this is a medical emergency, advise seeking immediate professional help.]"
            )
        else:  # seq2seq
            prompt = (
                "You are a helpful medical assistant. Please provide a clear, accurate, "
                f"and concise answer to the following medical question.\n\n"
                f"Question: {question}\n\n"
                "Answer:"
            )
        
        # Set generation parameters
        generation_params = {
            "max_length": min(max_length or config.get("max_length", 512), 1024),
            "temperature": config.get("temperature", 0.7),
            "top_p": config.get("top_p", 0.9),
            "do_sample": True,
            "num_return_sequences": 1,
            "min_length": config.get("min_length", 30),
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
        }
        
        # Add model-specific parameters
        if config["type"] == "causal":
            generation_params["pad_token_id"] = _model_cache[model_name]["tokenizer"].eos_token_id
            
        # Generate response
        response = pipe(prompt, **generation_params)
        
        # Extract the generated text
        if config["type"] == "causal":
            answer = response[0]["generated_text"][len(prompt):].strip()
        else:  # seq2seq
            answer = response[0]["generated_text"].strip()
        
        # Post-process the answer
        if not answer:
            return "I'm sorry, I couldn't generate a response. Please try rephrasing your question."

        # Add a simple disclaimer to reduce weird translations of bracketed text
        disclaimer = (
            "\n\nThis information is for educational purposes only and is not a "
            "substitute for professional medical advice, diagnosis, or treatment."
        )
        if len(answer.split()) < 10 or not any(p in answer[-3:] for p in ".!?"):
            answer += disclaimer
        elif "educational purposes only" not in answer.lower():
            answer += disclaimer

        return answer
        
    except Exception as e:
        logger.error(f"Error generating medical answer: {str(e)}")
        return (
            "I'm sorry, I encountered an error while processing your request. "
            "This might be due to technical limitations or an issue with the input. "
            "Please try again later or rephrase your question."
        )