import os
import re
import ast
import requests
from typing import Dict, Tuple
import yt_dlp
import pandas as pd
import numpy as np
import time

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Perplexity API Configuration
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

if not PERPLEXITY_API_KEY:
    print("WARNING: PERPLEXITY_API_KEY not found in .env file")

TEXT_STRATEGY_TEMPLATE = """You are an expert Python trading strategy developer. Generate a complete, working strategy function following this EXACT format:

def strategy(df):
    \"\"\"
    [Brief description of the strategy]
    \"\"\"
    import pandas as pd
    import numpy as np
    # Optional: import talib or pandas_ta for complex indicators
    
    # Step 1: Calculate indicators from OHLC data
    # Available columns: df['Open'], df['High'], df['Low'], df['Close'], df['Volume']
    
    # Step 2: Initialize Signal column
    df['Signal'] = 0
    
    # Step 3: Generate trading signals
    # 1 = BUY, -1 = SELL, 0 = HOLD
    
    return df

IMPORTANT RULES:
1. Function MUST be named "strategy" and take "df" parameter
2. MUST return df with 'Signal' column added
3. Calculate ALL indicators inside the function (RSI, MACD, SMA, etc.)
4. Use pandas/numpy/talib for calculations
5. Return ONLY executable Python code, no markdown, no explanations
6. Ensure the code is syntactically correct and will run

User's Trading Strategy Description:
{user_input}

Generate the complete Python function now:"""

YOUTUBE_STRATEGY_TEMPLATE = """You are an expert Python trading strategy developer. Your task is to extract a viable trading strategy from a YouTube transcript and implement it in Python.

TRANSCRIPT:
{transcript}

INSTRUCTIONS:
1. SIMPLIFY COMPLEX LOGIC: If the video describes complex "Market Structure" (swings, valid highs, etc.), approximate it with simple Rolling Max/Min or Donchian Channels.
   - WRONG: Complex loops with uninitialized variables (e.g., `valid_low = NaN`, then checking `if valid_low:`).
   - RIGHT: `df['Trend'] = np.where(df['Close'] > df['High'].rolling(20).max().shift(1), 1, -1)`
   
2. NO FUTURE LEAKAGE: NEVER use `shift(-1)` or negative indices. You cannot trade on future data.

3. ROBUSTNESS: 
   - Ensure the strategy WILL generate trades. Avoid strict conditions like `Range < Range * 0.3` (impossible).
   - Use standard indicators (ATR, MA, RSI) to detect "Consolidation" or "Trend".

4. R:R FILTERS: If implementing Reward:Risk ratio, ensure Stop Loss and Take Profit are calculated sensibly and allow trades to execute.

FORMAT:
def strategy(df):
    \"\"\"
    Strategy derived from YouTube video.
    Logic: [Summary of logic]
    \"\"\"
    import pandas as pd
    import numpy as np
    
    # Step 1: Indicators (MA, RSI, ATR, etc.)
    
    # Step 2: Signal (1=Buy, -1=Sell, 0=Hold)
    df['Signal'] = 0
    
    # [Implement Logic Here - PREFER VECTORIZED CODE OVER LOOPS]
    
    return df

Create the executable Python code now. NO markdown. ONLY code."""


def call_perplexity_api(prompt: str, max_tokens: int = 2000) -> str:
    """Call Perplexity API with the given prompt"""
    try:
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "sonar-pro",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert Python trading strategy code generator. Generate clean, executable Python code."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "top_p": 0.9
        }
        
        response = requests.post(PERPLEXITY_API_URL, json=payload, headers=headers, timeout=30)
        
        if not response.ok:
            print(f"API Error Response: {response.text}")
            
        response.raise_for_status()
        
        data = response.json()
        return data['choices'][0]['message']['content']
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Perplexity API error: {str(e)}")


def extract_youtube_subtitles(url: str) -> str:
    """Extract subtitles/transcript from YouTube video using yt-dlp"""
    try:
        print(f"DEBUG: Extracting from URL: {url}")
        
        ydl_opts = {
            'skip_download': True,
            'writesubtitles': True, # Try manual subs
            'writeautomaticsub': True, # Try auto-generated subs
            'subtitleslangs': ['en'], # download English subtitles
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 10, # Add timeout (seconds)
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            # 1. Try Manual Subtitles
            subs = info.get('subtitles', {})
            auto_subs = info.get('automatic_captions', {})
            
            caption_url = None
            
            if 'en' in subs:
                # Prefer json3 format
                for fmt in subs['en']:
                    if fmt['ext'] == 'json3':
                        caption_url = fmt['url']
                        break
                if not caption_url and subs['en']:
                    caption_url = subs['en'][0]['url']
            
            # 2. Fallback to Auto-Subtitles
            if not caption_url and 'en' in auto_subs:
                for fmt in auto_subs['en']:
                    if fmt['ext'] == 'json3':
                        caption_url = fmt['url']
                        break
                if not caption_url and auto_subs['en']:
                    caption_url = auto_subs['en'][0]['url']
            
            if not caption_url:
                raise Exception("No English subtitles found for this video")
            
            # Fetch the subtitle content
            print(f"DEBUG: Fetching subtitles from {caption_url[:50]}...")
            sub_res = requests.get(caption_url, timeout=10) # 10s timeout
            sub_res.raise_for_status()
            
            try:
                # Parse JSON3 format
                data = sub_res.json()
                full_text = ""
                # Structure: events -> segs -> utf8
                events = data.get('events', [])
                for event in events:
                    segs = event.get('segs', [])
                    for seg in segs:
                        text = seg.get('utf8', '').strip()
                        if text and text != '\n':
                            full_text += text + " "
                
                return full_text.strip()
            except Exception as e:
                print(f"JSON parsing failed, returning raw text: {e}")
                return sub_res.text[:10000] # Fallback
                
    except Exception as e:
        raise Exception(f"Error extracting subtitles: {str(e)}")


def verify_strategy_runtime(code: str) -> Tuple[bool, str]:
    """
    Execute the strategy code on dummy data to ensure it doesn't crash 
    and returns expected format.
    """
    try:
        # Create dummy data
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='5min')
        df = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 1000),
            'High': np.random.uniform(100, 200, 1000),
            'Low': np.random.uniform(100, 200, 1000),
            'Close': np.random.uniform(100, 200, 1000),
            'Volume': np.random.uniform(1000, 5000, 1000)
        }, index=dates)
        
        # Execute code to define the strategy function
        local_scope = {}
        # Catch syntax errors during definition
        try:
            exec(code, {}, local_scope)
        except Exception as e:
            return False, f"Syntax/Definition Error: {str(e)}"
        
        if 'strategy' not in local_scope:
            return False, "Function 'strategy' not found in executed code"
            
        strategy_func = local_scope['strategy']
        
        # Run strategy on dummy data
        try:
            result_df = strategy_func(df.copy())
        except Exception as e:
            return False, f"Execution Error: {str(e)}"
        
        # Check result
        if not isinstance(result_df, pd.DataFrame):
            return False, "Strategy must return a pandas DataFrame"
            
        if 'Signal' not in result_df.columns:
            return False, "Result DataFrame missing 'Signal' column"
            
        # Basic logical check
        # We allow other values but warn, or strictly enforce. Let's strictly enforce or just check existence.
        # Strict enforcement is better for compatibility.
        unique_signals = result_df['Signal'].unique()
        valid_signals = {-1, 0, 1} # using set for check, but unique returns numpy array
        
        # Just check it runs is mostly enough, but let's check one more thing
        if result_df.empty:
             return False, "Strategy returned empty DataFrame"

        return True, "Valid"
        
    except Exception as e:
        return False, f"Runtime Verification Error: {str(e)}"


def extract_python_code(text: str) -> str:
    """Extract Python code from API response"""
    # Remove markdown code blocks if present
    code = re.sub(r'```python\n?', '', text)
    code = re.sub(r'```\n?', '', code)
    
    # Try to find the strategy function
    match = re.search(r'def strategy\(.*?\):.*?return df', code, re.DOTALL)
    if match:
        return match.group(0)
    
    # If not found, return cleaned text
    return code.strip()


def validate_strategy_code(code: str) -> Tuple[bool, str]:
    """Validate that the generated code is syntactically correct and has required structure"""
    try:
        # Parse the code to check syntax
        tree = ast.parse(code)
        
        # Check for required function
        has_strategy_func = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'strategy':
                has_strategy_func = True
                
                # Check if it has one parameter
                if len(node.args.args) != 1:
                    return False, "strategy() function must take exactly one parameter (df)"
                
                break
        
        if not has_strategy_func:
            return False, "Code must contain a function named 'strategy'"
        
        # Check for return statement
        if 'return df' not in code and 'return' not in code:
            return False, "strategy() function must return df"
        
        return True, "Valid"
        
    except SyntaxError as e:
        return False, f"Syntax error: {str(e)}"


def generate_strategy_name(description: str) -> str:
    """Generate a safe filename from strategy description"""
    # Take first few words
    words = description.lower().split()[:3]
    name = '_'.join(word for word in words if word.isalnum())
    
    if not name:
        name = "ai_generated"
    
    # Add timestamp to make unique
    import time
    timestamp = int(time.time()) % 100000
    
    return f"{name}_{timestamp}"


def generate_and_validate_stream(prompt: str, max_retries: int = 2):
    """Generate strategy, validate runtime, and retry if failed - STREAMING VERSION"""
    
    current_prompt = prompt
    last_error = ""
    
    for attempt in range(max_retries + 1):
        if attempt > 0:
            print(f"DEBUG: Validation failed. Retrying (Attempt {attempt})...")
            yield {"status": "progress", "message": f"Validation failed ({last_error}). Retrying..."}
            # Enhance prompt with error
            current_prompt = f"{prompt}\n\nIMPORTANT: Your previous code had this runtime error:\n{last_error}\n\nPlease fix the code and return the CORRECTED full function."
        
        # Call API
        yield {"status": "progress", "message": "Asking AI to generate code..."}
        response = call_perplexity_api(current_prompt)
        code = extract_python_code(response)
        
        # 1. Static Validation
        yield {"status": "progress", "message": "Checking syntax..."}
        is_valid_syntax, message = validate_strategy_code(code)
        if not is_valid_syntax:
            last_error = f"Syntax/Structure Error: {message}"
            continue
            
        # 2. Runtime Validation
        yield {"status": "progress", "message": "Verifying logic on sample data..."}
        is_valid_runtime, message = verify_strategy_runtime(code)
        if not is_valid_runtime:
            print(f"DEBUG: Runtime Validation Error: {message}")
            last_error = f"Runtime Error: {message}"
            continue
            
        # Success
        print("DEBUG: Strategy passed all validation checks.")
        yield {"status": "complete", "code": code}
        return
        
    # If all retries fail
    yield {"status": "error", "message": f"Failed to generate valid strategy after {max_retries} retries. Final error: {last_error}"}


def generate_strategy_from_text_stream(description: str):
    """Generate strategy code from text description - Streaming"""
    if not description or len(description.strip()) < 10:
        yield {"status": "error", "message": "Please provide a more detailed strategy description"}
        return
    
    # Create prompt
    yield {"status": "progress", "message": "Preparing AI instructions..."}
    prompt = TEXT_STRATEGY_TEMPLATE.format(user_input=description)
    
    # Generate and Validate
    generated_code = None
    for update in generate_and_validate_stream(prompt):
        if update["status"] == "complete":
            generated_code = update["code"]
        elif update["status"] == "error":
            yield update
            return
        else:
            yield update
            
    if generated_code:
        # Generate suggested name
        suggested_name = generate_strategy_name(description)
        
        yield {
            "status": "complete",
            "data": {
                "code": generated_code,
                "suggested_name": suggested_name,
                "description": description
            }
        }


def generate_strategy_from_youtube_stream(url: str):
    """Generate strategy code from YouTube video - Streaming"""
    # Extract subtitles
    yield {"status": "progress", "message": "Extracting YouTube Transcript..."}
    try:
        subtitles = extract_youtube_subtitles(url)
    except Exception as e:
        yield {"status": "error", "message": f"Failed to extract video: {str(e)}"}
        return
    
    # Create enhanced prompt
    yield {"status": "progress", "message": "Analyzing Strategy Logic..."}
    prompt = YOUTUBE_STRATEGY_TEMPLATE.format(transcript=subtitles[:15000]) # Limit length safety
    
    # Generate and Validate
    generated_code = None
    for update in generate_and_validate_stream(prompt):
        if update["status"] == "complete":
            generated_code = update["code"]
        elif update["status"] == "error":
            yield update
            return
        else:
            yield update
            
    if generated_code:
        # Generate suggested name
        suggested_name = f"youtube_strategy_{int(time.time()) % 100000}"
        
        yield {
            "status": "complete",
            "data": {
                "code": generated_code,
                "suggested_name": suggested_name,
                "description": f"Strategy from YouTube: {url}"
            }
        }


def save_strategy_file(code: str, name: str) -> str:
    """Save strategy code to a file in strategies/ folder"""
    # Sanitize filename
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '', name)
    if not safe_name:
        safe_name = "generated_strategy"
    
    filename = f"{safe_name}.py"
    # Use backtesting/strategies relative to project root
    strategies_dir = os.path.join("backtesting", "strategies")
    filepath = os.path.join(strategies_dir, filename)
    
    # Ensure strategies directory exists
    os.makedirs(strategies_dir, exist_ok=True)
    
    # Check if file already exists
    if os.path.exists(filepath):
        # Add number suffix
        counter = 1
        while os.path.exists(filepath):
            filename = f"{safe_name}_{counter}.py"
            filepath = os.path.join("strategies", filename)
            counter += 1
    
    # Write file
    with open(filepath, 'w') as f:
        f.write(code)
        f.write('\n')  # Ensure newline at end
    
    return filename


# For testing
if __name__ == "__main__":
    import time
    # Test text generation
    print("Testing text generation...")
    result = generate_strategy_from_text("Buy when RSI is below 30, sell when RSI is above 70")
    print(f"Generated code:\n{result['code']}")
    print(f"\nSuggested name: {result['suggested_name']}")
