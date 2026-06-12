import requests

def explain_recommendations(api_key, model_name, user_history, recommendations):
    """
    Calls OpenRouter API to explain the recommendations based on user history.
    """
    if not api_key:
        return "Please provide an OpenRouter API key to get an AI explanation."
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Format the input
    history_str = "\n".join([f"- {movie} (Rated {rating}/5)" for movie, rating in user_history])
    rec_str = "\n".join([f"- {movie}" for movie in recommendations])
    
    prompt = f"""
The user has highly rated the following movies:
{history_str}

Based on this history, we are recommending the following movies:
{rec_str}

Please explain in a short, friendly paragraph (max 3 sentences) why these recommendations make sense for the user. Focus on genres, themes, or similar vibes.
"""
    
    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful, expert movie recommendation assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=15)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error from OpenRouter API: {response.status_code} - {response.text}"
    except Exception as e:
        return f"An error occurred while connecting to OpenRouter: {str(e)}"
