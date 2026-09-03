from dataclasses import dataclass
from dotenv import load_dotenv

from doc_explainer.config.backend import BackendConfig
from doc_explainer.config.llm import LLMConfig

# Load .env variables at the top of the module
load_dotenv()

@dataclass
@dataclass
class OrchestratorConfig:
    llm: LLMConfig
    backend: BackendConfig
    default_user_id: str = "default_user"