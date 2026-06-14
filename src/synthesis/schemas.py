from pydantic import BaseModel, Field

class GeneratedModule(BaseModel):
    module_name: str = Field(description="The functional title abstracting the core semantic nodes, formatted strictly as a snake_case string (e.g., 'core_database_architecture').")
    source_code: str = Field(description="The raw Python source code containing the generated Pydantic BaseModels and Enums, including all necessary import statements.")
