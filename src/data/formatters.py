"""
Data Formatters Module
======================

Functions for formatting data with chat templates and preprocessing.
"""

from typing import Dict, Any, List
from datasets import Dataset


class ChatTemplateFormatter:
    """Formatter for chat templates."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def format_chat_template(self, example: Dict[str, Any]) -> Dict[str, str]:
        """Format example using chat template."""
        if hasattr(self.tokenizer, 'apply_chat_template'):
            # Extract messages from the text field if it contains chat format
            text = example.get("text", "")
            return {"text": text}
        return example


def format_chat_template(dataset: Dataset, tokenizer) -> Dataset:
    """
    Format dataset using chat template.
    
    Args:
        dataset: Input dataset
        tokenizer: Tokenizer with chat template
    
    Returns:
        Formatted dataset
    """
    formatter = ChatTemplateFormatter(tokenizer)
    return dataset.map(formatter.format_chat_template, batched=False)


def alpaca_prompt_format(instruction: str, input_text: str = "", output: str = "") -> str:
    """
    Format text using Alpaca prompt template.
    
    Args:
        instruction: The instruction/task description
        input_text: Input text (optional)
        output: Expected output (for training)
    
    Returns:
        Formatted prompt string
    """
    if input_text:
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    else:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""
    
    return prompt


def format_example_for_task(example: Dict[str, Any], task_type: str) -> Dict[str, str]:
    """
    Format example based on task type.
    
    Args:
        example: Raw example from dataset
        task_type: Type of task ("summarization", "translation", "classification")
    
    Returns:
        Formatted example with "text" field
    """
    if task_type == "summarization":
        return {
            "text": f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that summarizes news articles. Provide a concise and accurate summary of the given article.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Please summarize the following article:

{example.get('article', '')}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example.get('highlights', '')}<|eot_id|>"""
        }
    
    elif task_type == "translation":
        source_lang = example.get('source_lang', 'de')
        target_lang = example.get('target_lang', 'en')
        source_text = example.get('source_text', '')
        target_text = example.get('target_text', '')
        
        lang_names = {"de": "German", "en": "English", "fr": "French", "es": "Spanish"}
        source_name = lang_names.get(source_lang, source_lang)
        target_name = lang_names.get(target_lang, target_lang)
        
        return {
            "text": f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a professional translator. Translate the given {source_name} text to {target_name} accurately and naturally.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Translate the following {source_name} text to {target_name}:

{source_text}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{target_text}<|eot_id|>"""
        }
    
    elif task_type == "classification":
        text = example.get('sentence', example.get('text', ''))
        label = example.get('label', '')
        
        # Convert numeric labels to text if needed
        if isinstance(label, (int, float)):
            label_map = {0: "negative", 1: "positive"}
            label = label_map.get(int(label), str(label))
        
        return {
            "text": f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a sentiment analysis expert. Analyze the sentiment of the given text and classify it as either "positive" or "negative".

<|eot_id|><|start_header_id|>user<|end_header_id|>

Analyze the sentiment of the following text:

{text}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{label}<|eot_id|>"""
        }
    
    else:
        # Default formatting
        return {"text": str(example)}


def create_formatting_function(task_type: str):
    """
    Create a formatting function for a specific task type.
    
    Args:
        task_type: Type of task
    
    Returns:
        Formatting function
    """
    def formatting_func(example):
        return format_example_for_task(example, task_type)
    
    return formatting_func 