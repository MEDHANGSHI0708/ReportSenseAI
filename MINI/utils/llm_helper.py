import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

def initialize_genai():
    """Initialize the Google Generative AI client."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set in the .env file.")
    genai.configure(api_key=api_key)

def refine_text_with_llm(raw_text):
    """Refine text using Google's Generative AI."""
    initialize_genai()
    try:
        # Get the model
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Generate content
        response = model.generate_content(
            f"Refine the following text to make it more clear and professional:\n\n{raw_text}",
            generation_config={
                'temperature': 0.7,
                'max_output_tokens': 1024,
            }
        )
        
        return response.text if response.text else "Unable to refine the text."
    except Exception as e:
        return f"An error occurred while refining text: {str(e)}"

def query_llm_with_context(context, user_query):
    """Query the LLM with context using Google's Generative AI."""
    initialize_genai()
    try:
        # Get the model
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Construct the prompt with context
        prompt = f"""Context: {context}
        
Question: {user_query}

Please provide a detailed answer based on the context above."""
        
        # Generate response
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.7,
                'max_output_tokens': 1024,
            }
        )
        
        return response.text if response.text else "Unable to generate a response."
    except Exception as e:
        return f"An error occurred while generating a response: {str(e)}"

def main():
    """Example usage of the functions."""
    # Refine raw text
    raw_text = "This is a sample text that needs refinement."
    refined_text = refine_text_with_llm(raw_text)
    print("Refined Text:", refined_text)
    
    # Chat with context
    context = "The patient has a history of diabetes and hypertension."
    user_query = "What precautions should they take during a flu season?"
    response = query_llm_with_context(context, user_query)
    print("Response to Query:", response)

if __name__ == "__main__":
    main()