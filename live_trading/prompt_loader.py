import yaml
import os
from jinja2 import Environment, BaseLoader, TemplateNotFound

class DictLoader(BaseLoader):
    def __init__(self, templates):
        self.templates = templates
        
    def get_source(self, environment, template):
        if template in self.templates:
            return self.templates[template], template, lambda: True
        raise TemplateNotFound(template)

class PromptLoader:
    """
    A generic YAML-based prompt loader for AI trading prompts.
    Provides backwards compatibility for existing system prompts via `create_system_prompt()`.
    """
    def __init__(self, prompts_dir="prompts"):
        self.directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), prompts_dir)
        self.prompts = self._load_all_yaml()
        
        # Flatten system/user templates for Jinja
        jinja_templates = {}
        for prompt_name, prompt_config in self.prompts.items():
            if isinstance(prompt_config, dict):
                if "system_prompt" in prompt_config:
                    jinja_templates[f"{prompt_name}_system"] = prompt_config["system_prompt"]
                if "user_prompt" in prompt_config:
                    jinja_templates[f"{prompt_name}_user"] = prompt_config["user_prompt"]
                
        self.env = Environment(loader=DictLoader(jinja_templates))

    def _load_all_yaml(self):
        prompts = {}
        if not os.path.exists(self.directory):
            print(f"Warning: Directory {self.directory} not found.")
            return prompts
            
        for filename in os.listdir(self.directory):
            if filename.endswith(".yaml") or filename.endswith(".yml"):
                filepath = os.path.join(self.directory, filename)
                name = os.path.splitext(filename)[0]
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        file_content = yaml.safe_load(f) or {}
                        # If the YAML content has a 'default' or similar top-level key, use that.
                        # But since we're using separate files now, we can expect the file itself 
                        # to contain 'system_prompt' and 'user_prompt', or it might contain sub-strategies.
                        # Let's support both: if it has 'system_prompt', wrap it. Otherwise merge.
                        if 'system_prompt' in file_content:
                            prompts[name] = file_content
                        else:
                            # If it has sub-keys (like 'default:' inside), merge them
                            for sub_name, sub_content in file_content.items():
                                if sub_name == 'default' and name != 'default':
                                    prompts[name] = sub_content
                                else:
                                    prompts[f"{name}_{sub_name}" if sub_name != 'default' else name] = sub_content
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
        return prompts

    def get_prompt(self, prompt_name: str, **kwargs) -> tuple:
        """
        Returns (system_prompt, user_prompt) strings formatted with Jinja2 using provided kwargs.
        """
        if prompt_name not in self.prompts:
            # Fallback to default if not found
            if 'default' in self.prompts:
                prompt_name = 'default'
            else:
                return "", ""
                
        system_template = self.env.get_template(f"{prompt_name}_system") if f"{prompt_name}_system" in self.env.loader.templates else None
        user_template = self.env.get_template(f"{prompt_name}_user") if f"{prompt_name}_user" in self.env.loader.templates else None
        
        system_prompt = system_template.render(**kwargs) if system_template else ""
        user_prompt = user_template.render(**kwargs) if user_template else ""
        
        return system_prompt, user_prompt


# ---------- Adapter to keep analyst.py seamlessly backwards-compatible ---------- #
_global_loader = None

def create_system_prompt(
    minutes_elapsed: int,
    indicators: dict,
    account_info: dict,
    symbol: str,
    interval: str = "1m",
    ema_period: int = 20,
    rsi_period: int = 14,
    leverage: float = 1.0,
    currency_symbol: str = "$",
    **kwargs
) -> str:
    """
    Backwards-compatible wrapper that parses trading state into a template dictionary 
    and generates the prompt using Jinja2 templates stored in prompts.yaml.
    """
    global _global_loader
    if _global_loader is None:
        _global_loader = PromptLoader()
        
    # Python-side computed logic for complex fields
    market_structure = []
    if indicators.get('above_ema20'): market_structure.append("Above EMA20")
    if indicators.get('above_ema50'): market_structure.append("Above EMA50")
    if indicators.get('above_ema200'): market_structure.append("Above EMA200")
    market_structure_summary = " | ".join(market_structure) if market_structure else "Below key EMAs"
    
    # Pre-format arrays for simplicity so Jinja doesn't have to loop overly complex dictionaries
    price_series_fmt = str([f"{currency_symbol}{x:.1f}" for x in indicators.get('price_series', [])[-10:]]).replace("'", "")
    ema_series_fmt = str([f"{currency_symbol}{x:.1f}" for x in indicators.get('ema_series', [])[-10:]]).replace("'", "")
    macd_series_fmt = str([f"{x:.4f}" for x in indicators.get('macd_series', [])[-10:]]).replace("'", "")
    rsi_series_fmt = str([f"{x:.2f}" for x in indicators.get('rsi_series', [])[-10:]]).replace("'", "")
    
    # Context injected into Jinja template
    ctx = {
        'minutes_elapsed': minutes_elapsed,
        'symbol': symbol,
        'interval': interval,
        'ema_period': ema_period,
        'rsi_period': rsi_period,
        'leverage': leverage,
        'currency_symbol': currency_symbol,
        'market_structure_summary': market_structure_summary,
        
        # Top level indicators
        'current_price': indicators.get('current_price', 0),
        'rsi_level': indicators.get('rsi_level', 'NEUTRAL'),
        'macd_trend': indicators.get('macd_trend', 'NEUTRAL'),
        'ema_20': indicators.get('ema_20', 0),
        'ema_50': indicators.get('ema_50', 0),
        'ema_200': indicators.get('ema_200', 0),
        'macd': indicators.get('macd', 0),
        'macd_signal': indicators.get('macd_signal', 0),
        'rsi': indicators.get('rsi', 0),
        'bb_position': indicators.get('bb_position', 0),
        'oi_latest': indicators.get('oi_latest', 0),
        'oi_average': indicators.get('oi_average', 0),
        'funding_rate': indicators.get('funding_rate', 0),
        
        # String pre-formatted series
        'price_series_fmt': price_series_fmt,
        'ema_series_fmt': ema_series_fmt,
        'macd_series_fmt': macd_series_fmt,
        'rsi_series_fmt': rsi_series_fmt,
        
        # Account info Flattened
        'total_return_percent': account_info.get('total_return_percent', 0),
        'available_cash': account_info.get('available_cash', 0),
        'account_value': account_info.get('current_value', account_info.get('account_value', 0)),
        'sharpe_ratio': account_info.get('sharpe_ratio', 0.0),
        
        # Position Data
        'position': account_info.get('position', {}),
        'has_position': account_info.get('position', {}).get('quantity', 0) > 0,
        'position_type': account_info.get('position', {}).get('position_type', 'NONE'),
        'position_quantity': account_info.get('position', {}).get('quantity', 0.0),
        'entry_price': account_info.get('position', {}).get('entry_price', 0.0),
        'position_current_price': account_info.get('position', {}).get('current_price', 0.0),
        'unrealized_pnl': account_info.get('position', {}).get('unrealized_pnl', 0.0),
        'pnl_percent': account_info.get('position', {}).get('pnl_percent', 0.0),
        'risk_usd': account_info.get('position', {}).get('risk_usd', 0.0),
    }
    
    # Pull any additional kwargs explicitly
    ctx.update(kwargs)
    
    # 'prompt_name' handles logic routing for new prompt testing
    prompt_name = kwargs.get('prompt_name', 'default')
    
    system_str, user_str = _global_loader.get_prompt(prompt_name, **ctx)
    
    # Provide system prompt + optionally appended user parameters
    final_prompt = system_str
    if user_str and user_str.strip():
        final_prompt += "\n\n" + user_str.strip()
        
    return final_prompt
