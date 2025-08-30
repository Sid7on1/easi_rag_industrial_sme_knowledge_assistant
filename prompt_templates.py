import logging
from typing import Dict, List, Optional
from jinja2 import Template, FileSystemLoader, Environment
from pydantic import BaseModel, ValidationError
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE_LOADER = FileSystemLoader(str(TEMPLATE_DIR))
TEMPLATE_ENV = Environment(loader=TEMPLATE_LOADER)

# Define exception classes
class PromptTemplateError(Exception):
    """Base exception class for prompt template errors"""
    pass

class PromptTemplateNotFoundError(PromptTemplateError):
    """Raised when a prompt template is not found"""
    pass

class PromptTemplateValidationError(PromptTemplateError):
    """Raised when a prompt template is invalid"""
    pass

# Define data structures/models
class PromptTemplate(BaseModel):
    """Represents a prompt template"""
    name: str
    template: str

class PromptContext(BaseModel):
    """Represents a prompt context"""
    query: str
    context: str

# Define validation functions
def validate_prompt_template(template: Dict) -> None:
    """Validates a prompt template"""
    try:
        PromptTemplate(**template)
    except ValidationError as e:
        raise PromptTemplateValidationError(f"Invalid prompt template: {e}")

# Define utility methods
def load_template(template_name: str) -> Template:
    """Loads a template from the template directory"""
    try:
        return TEMPLATE_ENV.get_template(template_name)
    except Exception as e:
        raise PromptTemplateNotFoundError(f"Template not found: {template_name}") from e

def render_template(template: Template, context: Dict) -> str:
    """Renders a template with the given context"""
    try:
        return template.render(**context)
    except Exception as e:
        raise PromptTemplateError(f"Error rendering template: {e}") from e

# Define main class
class PromptTemplateManager:
    """Manages prompt engineering and templates for different query types"""
    def __init__(self, template_dir: Optional[str] = None):
        """Initializes the prompt template manager"""
        if template_dir:
            global TEMPLATE_DIR
            TEMPLATE_DIR = Path(template_dir)
            global TEMPLATE_LOADER
            TEMPLATE_LOADER = FileSystemLoader(str(TEMPLATE_DIR))
            global TEMPLATE_ENV
            TEMPLATE_ENV = Environment(loader=TEMPLATE_LOADER)

    def load_prompt_template(self, template_name: str) -> PromptTemplate:
        """Loads a prompt template by name"""
        template = load_template(template_name)
        return PromptTemplate(name=template_name, template=str(template))

    def format_context_prompt(self, prompt_context: PromptContext) -> str:
        """Formats a context prompt"""
        template = self.load_prompt_template("context_prompt.jinja2")
        context = {"query": prompt_context.query, "context": prompt_context.context}
        return render_template(Template(template.template), context)

    def create_system_prompt(self, query: str, context: str) -> str:
        """Creates a system prompt"""
        template = self.load_prompt_template("system_prompt.jinja2")
        context = {"query": query, "context": context}
        return render_template(Template(template.template), context)

    def validate_prompt(self, prompt: str) -> None:
        """Validates a prompt"""
        try:
            # Implement prompt validation logic here
            pass
        except Exception as e:
            raise PromptTemplateError(f"Invalid prompt: {e}") from e

# Define integration interfaces
class PromptTemplateService:
    """Provides prompt template services"""
    def __init__(self, prompt_template_manager: PromptTemplateManager):
        """Initializes the prompt template service"""
        self.prompt_template_manager = prompt_template_manager

    def get_prompt_template(self, template_name: str) -> PromptTemplate:
        """Gets a prompt template by name"""
        return self.prompt_template_manager.load_prompt_template(template_name)

    def format_prompt(self, prompt_context: PromptContext) -> str:
        """Formats a prompt"""
        return self.prompt_template_manager.format_context_prompt(prompt_context)

# Example usage
if __name__ == "__main__":
    prompt_template_manager = PromptTemplateManager()
    prompt_context = PromptContext(query="What is the meaning of life?", context="The meaning of life is a philosophical question.")
    formatted_prompt = prompt_template_manager.format_context_prompt(prompt_context)
    print(formatted_prompt)