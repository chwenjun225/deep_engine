from langchain_core.prompts import PromptTemplate 
from langchain_ollama import ChatOllama



from pydantic import BaseModel



class AnswerWithJustification(BaseModel):
	"""An answer to the user's question along with justification for the answer."""
	answer: str
	justification: str



tool_desc = PromptTemplate.from_template(
	"""{name_for_model}: Call this tool to interact with the {name_for_human} API. 
What is the {name_for_human} API useful for? 
{description_for_model}.
Parameters: {parameters}"""
)



prompt_react = PromptTemplate.from_template("""You are an AI assistant that follows the ReAct reasoning framework. 
You have access to the following APIs:

{tools_desc}

Use the following strict format:

### Input Format:

Question: [The input question]
Thought: [Think logically about the next step]
Action: [Select from available tools: {tools_name}]
Action Input: [Provide the required input]
Observation: [Record the output from the action]
... (Repeat the Thought/Action/Observation loop as needed)
Thought: I now know the final answer
Final Answer: [Provide the final answer]

Begin!

Question: {query}""")



STOP_WORDS = ["Observation:", "Observation:\n"]

model = ChatOllama(
	model="llama3.2:1b-instruct-fp16", 
	temperature=0.1, 
	num_predict="1024"
)



model_structured_outptu = model.with_structured_output(AnswerWithJustification)

resp = model_structured_outptu.invoke("""What weighs more, a pound of bricks or a pound of feathers""")

print(resp)

