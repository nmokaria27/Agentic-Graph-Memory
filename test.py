
from strands.models.ollama import OllamaModel
from strands import Agent

# %%time
ollama_agent = OllamaModel(
    host="http://localhost:11434",
    #host="http://gpu01.mind.cs.umd.edu:11434",
    model_id='qwen3:8b',
    options={
        "temperature": 0.1,
        "max_tokens": 500,
        "seed": 5151
    }
)

agent = Agent(model=ollama_agent)
remote_response = agent("What is the color of the sky? Give me a one word answer only.")
print(remote_response)