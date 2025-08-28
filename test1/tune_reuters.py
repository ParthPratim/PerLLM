"""
This script will do the following : 
1. Load a 70B Llama model
2. Check using ICL if it is able to identify the topics
3. If yes, then we can move forward with an ICL based technique. We are interested in 0-shot, due to unavailibity of supervision
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets 

def load_model(model_name=""):
