import re

class PineScriptParser:
    def __init__(self, pine_code):
        self.pine_code = pine_code

    def parse(self):
        """
        Parses the Pine Script code and returns a Python function (as a closure or source code string)
        that implements the strategy.
        For simplicity in this version, we will return a function that takes a DataFrame.
        """
        
        # 1. basic cleanup
        lines = self.pine_code.split('\n')
        
        # We will construct a python string and then exec it to get the function
        # This is compliant with the "dynamic" nature requested, though has security implications in prod
        
        python_code = [
            "def strategy(df):",
            "    import pandas as pd",
            "    df = df.copy()",
            # "    df.columns = [c.lower() for c in df.columns]", # standardizing - REMOVED to keep compatibility with backtester
        ]
        
        # 2. Parse variables and indicators
        # Example: fast = ta.sma(close, 14)
        # Regex to find assignments: var = ta.something(source, args)
        
        for line in lines:
            line = line.strip()
            if line.startswith("//") or not line:
                continue
            
            # Skip strategy configuration line
            if line.startswith("strategy("):
                continue
            
            # Skip entry/close commands (we rely on variable detection at the end for now)
            if line.startswith("strategy."):
                continue
                
            # Skip 'if' lines for this simple prototype (we assume vectorized logic)
            if line.startswith("if"):
                continue

            # Common Pine variables - Map to Capitalized columns as per backtester
            line = re.sub(r'\bclose\b', 'df["Close"]', line)
            line = re.sub(r'\bopen\b', 'df["Open"]', line)
            line = re.sub(r'\bhigh\b', 'df["High"]', line)
            line = re.sub(r'\blow\b', 'df["Low"]', line)
            line = re.sub(r'\bvolume\b', 'df["Volume"]', line)
            
            # ta.sma(source, length)
            match_sma = re.search(r'ta\.sma\((.*?), \s*(\d+)\)', line)
            if match_sma:
                src = match_sma.group(1)
                length = match_sma.group(2)
                line = line.replace(match_sma.group(0), f"{src}.rolling({length}).mean()")
            
            # ta.crossover(a, b)
            match_cross = re.search(r'ta\.crossover\((.*?), (.*?)\)', line)
            if match_cross:
                a = match_cross.group(1)
                b = match_cross.group(2)
                # Ensure we handle the comparison robustly
                replacement = f"(({a} > {b}) & ({a}.shift(1) <= {b}.shift(1)))"
                line = line.replace(match_cross.group(0), replacement)

            if "=" in line:
                 python_code.append(f"    {line}")
        
        # 3. Final Signal generation logic (Hardcoded helper for the prototype)
        # We assume the user creates boolean Series named 'longCondition' and 'shortCondition'
        # This is a constraint we must inform the user about or try to parse
        
        python_code.append("    df['Signal'] = 0")
        
        # Check if longCondition exists in the parsed lines (heuristic)
        python_code.append("    if 'longCondition' in locals():")
        python_code.append("        df.loc[longCondition, 'Signal'] = 1")
        
        python_code.append("    if 'shortCondition' in locals():")
        python_code.append("        df.loc[shortCondition, 'Signal'] = -1")
            
        python_code.append("    return df")
        
        full_code = "\n".join(python_code)
        
        # Compile and return function
        # WARNING: unsafe, but standard for this kind of "playground" tool
        local_scope = {}
        exec(full_code, {}, local_scope)
        return local_scope['strategy']

# Note: This is an extremely basic parser. 
# Robust pine script parsing requires a full lexer/parser/transpiler.
