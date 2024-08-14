import os
from dotenv import load_dotenv, find_dotenv

def load_api_keys():
    load_dotenv(r'C:\Users\user\Desktop\LangChain\Daily\all.env')
    
    api_keys = {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'LANGCHAIN_API_KEY': os.getenv('LANGCHAIN_API_KEY'),
        'FIRE_API_KEY': os.getenv('FIRE_API_KEY')
    }
    
    # Langsmith Tracing (optional)
    os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2', 'false')
    os.environ['LANGCHAIN_ENDPOINT'] = os.getenv('LANGCHAIN_ENDPOINT', '')
    os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT', '')
    
    return api_keys

def set_api_keys(api_keys):
    for key, value in api_keys.items():
        os.environ[key] = value

        