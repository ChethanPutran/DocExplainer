
from langchain_core.output_parsers import PydanticOutputParser
from src.core.agent.models import ExplanationPydantic





# Initialize the parser
explanation_output_parser = PydanticOutputParser(pydantic_object=ExplanationPydantic)



# explanation_output_parser = PydanticOutputParser(
#     pydantic_object=Explanation
# )