import json
from typing import Any, Optional
from langchain_core.output_parsers import JsonOutputParser

from ..base.exceptions import ParserError
from .base import BaseParser


class JSONParser(BaseParser, JsonOutputParser):
    """JSON output parser with error handling"""
    
    def parse(self, text: str) -> Any:
        """Parse JSON with error handling"""
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # Try to extract JSON from text
            import re
            json_match = re.search(r'\{.*\}|\[.*\]', text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            raise ParserError(f"Failed to parse JSON: {e}") from e