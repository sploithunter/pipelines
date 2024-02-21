from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import OpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOpenAI
from langchain_core.runnables import chain
import json

from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
mistral_api_key = os.getenv('MISTRAL_API_KEY')



from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)





def num_tokens_from_string(string):
    num_tokens = len(string.split())
    return num_tokens

def calculate_llm_input_cost(model_type, inputs):

    word_count = num_tokens_from_string(inputs)

    model_input_cost_mapping = {
        'gpt35': 3/1000,
        'gpt4T': 120/1000,
        'mistral-tiny': 0.4/1000,
        'mistral-small': 1.5/1000,
        'mistral-medium': 6/1000
    }
    llmInCost = model_input_cost_mapping.get(model_type, 0)
    return word_count * llmInCost

def calculate_llm_output_cost(model_type, output):
    
    word_count = num_tokens_from_string(output)

    model_output_cost_mapping = {
        'gpt35': 4/1000,
        'gpt4T': 240/1000,
        'mistral-tiny': 1.2/1000,
        'mistral-small': 4.4/1000,
        'mistral-medium': 18/1000
    }
    llmOutCost = model_output_cost_mapping.get(model_type, 0)
    return word_count * llmOutCost



def get_generic_runnable(memory,model_type="mistral-medium", api_key = '', temperature=0.3, promptSysInternal = "you are a helpful assistant" ):
    promptInternal = ChatPromptTemplate.from_messages(
    [
        ("system", promptSysInternal),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
    )

    promptInternalNoMem = ChatPromptTemplate.from_messages(
    [
        ("system", promptSysInternal),
        ("human", "{input}"),
    ]
    )
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    mistral_api_key = os.getenv('MISTRAL_API_KEY')
    #if model is mistral type
    if "mistral" in model_type:
        if api_key:
            mistral_api_key = api_key
        modelInternal = ChatMistralAI(mistral_api_key=mistral_api_key, model=model_type,temperature=temperature )
    elif "gpt" in model_type:
        if api_key:
            openai_api_key = api_key

        if model_type == "gpt35":
            modelInternal = ChatOpenAI(temperature=temperature,  openai_api_key=openai_api_key, model="gpt-3.5-turbo")
        elif model_type == "gpt4T":
            modelInternal = ChatOpenAI(temperature=temperature,  openai_api_key=openai_api_key, model="gpt-4-turbo-preview")
        else:
            modelInternal = ChatOpenAI(temperature=temperature,  openai_api_key=openai_api_key, model=model_type)
            if "gpt-4" in model_type:
                model_type = "gpt4T"
            else:
                model_type = "gpt35"


    @chain 
    def generic_chain(inputDict):
        runnable1 = (
            RunnablePassthrough.assign(
                history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
            )
            | promptInternal
            | modelInternal
            | StrOutputParser()
        )
        inputs = {"input": inputDict["input"]}
       # print(f' This is the input: {inputs}')
        fullresponse = ""
        for chunk in runnable1.stream(inputs):
            fullresponse += chunk
            yield {"stream":chunk}
        
        cost = inputDict["cost"]
        cost += calculate_llm_input_cost(model_type, inputDict["input"])
        cost += calculate_llm_input_cost(model_type, str(memory.load_memory_variables({}))) * 2
        cost += calculate_llm_output_cost(model_type, fullresponse)
        memory.save_context(inputs, {"output": fullresponse})
        yield {"output": fullresponse, "cost": cost}

    @chain 
    def generic_chain_no_memory(inputDict):
        runnable1 = (
             promptInternalNoMem
            | modelInternal
            | StrOutputParser()
        )
        inputs = {"input": inputDict["input"]}
       # print(f' This is the input: {inputs}')
        fullresponse = ""
        for chunk in runnable1.stream(inputs):
            fullresponse += chunk
            yield {"stream":chunk}
        
        cost = inputDict["cost"]
        cost += calculate_llm_input_cost(model_type, inputDict["input"])
        #cost += calculate_llm_input_cost(model_type, str(memory.load_memory_variables({}))) * 2
        cost += calculate_llm_output_cost(model_type, fullresponse)
        #memory.save_context(inputs, {"output": fullresponse})
        yield {"output": fullresponse, "cost": cost} 
    if memory:
        return generic_chain
    else:
        return generic_chain_no_memory
    

memmodelTest = ChatOpenAI(temperature=.1,  openai_api_key=openai_api_key)
memoryTest = ConversationSummaryBufferMemory(llm=memmodelTest,memory_key="history", max_token_limit=4000, return_messages=True)
schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Task Breakdown",
  "description": "A schema for representing tasks, their subtasks, dependencies, and outputs",
  "type": "object",
  "properties": {
    "taskId": {
      "type": "string",
      "description": "Unique identifier for the task"
    },
    "taskName": {
      "type": "string",
      "description": "Name of the task"
    },
    "description": {
      "type": "string",
      "description": "A brief description of the task"
    },
     "reasoning": {
      "type": "string",
      "description": "The reasoning behing the assignment of this task.  This is a verbose description of why the task is needed"
    },
    "status": {
      "type": "string",
      "description": "Current status of the task (e.g., 'pending', 'in progress', 'completed')",
      "enum": ["pending", "in progress", "completed"]
    },
    "dependencies": {
      "type": "array",
      "description": "List of taskIds that this task depends on",
      "items": {
        "type": "string"
      }
    },
    "subtasks": {
      "type": "array",
      "description": "List of subtasks, each a nested object following the same schema",
      "items": {
        "$ref": "#"
      }
    },
    "output": {
      "type": ["object", "null"],
      "description": "Output of the task. Null if the task has subtasks and is not a terminal task",
      "properties": {
        "finalAnswer": {
          "type": ["string", "null"],
          "description": "The final answer or result of the task. Null if the task is not terminal"
        },
        "intermediateOutput": {
          "type": ["string", "null"],
          "description": "Intermediate output of the task. Useful for tasks that are completed but are part of a larger task"
        }
      },
      "required": ["finalAnswer", "intermediateOutput"]
    }
  },
  "required": ["taskId", "taskName", "status", "subtasks", "output"]
}

def newJSONfromString(inputStr, cost):
    s = str(inputStr)
    try:
        if "```json" in s:
            start = s.find('```json') + len('```json')
            end = s.find('```', start)
            s = s[start:end].strip()
        print("going to load json here:  " + str(s))
        newInput = json.loads(s)
    except:
        system_prompt =f"You are a helpful AI assistant.   Output in JSON."
        runnable = get_generic_runnable(None,model_type="mistral-medium", promptSysInternal=system_prompt)
        payload = {"input": "Consider this JSON\n{s}\n If it is not valid JSON, the assistant will try to convert it to JSON.", "cost": cost}
        newInputDict = runnable.invoke(payload)
        newInput = newInputDict["output"]
        cost = newInputDict["cost"]
    return newInput, cost
 
def execute_building_chain(newRequest, runningJSON, runningcost):
    if not runningJSON:
        runningRequest = ""
    else:
        runningJSONstr = json.dumps(runningJSON)
        print(f"runningJSONstr is: {runningJSONstr}")
        runningRequest = f"This is what we have so far: \n {runningJSONstr}\n You are working on the following:\n"
    schemaDump = json.dumps(schema)
    system_prompt =f"You are a helpful AI assistant.   Output in JSON."
    runnable = get_generic_runnable(None,model_type="gpt35", promptSysInternal=system_prompt)
    import sys
    country = sys.argv[1]
    print(f"country is: {country}")
   
        
    promptKind =  f"""Think step by step.
        Your output should be a first level task breakdown.  If there are complext tasks,
        you just list them as subtasks and take no other action. If the task is simple enough to be completed,
        you can output the final answer.  taskId should be in the form of 1, 1.1, 1.1.1, etc. 
        If the taskId is more than 3 digits long, just provide your best final answer.
        If you are not creating at least one subtask, there must be a final answer!
        As we are testing, if the task is to research and you have the answer, answer it.
        If the task is not simple enough, you can output intermediate results.
        Output JSON using the schema defined here: \n{schemaDump}\n ONLY OUTPUT JSON!!!  
        Follow the schema exactly.  JSON data must always be enclosed by the ```json``` tag.  

        """
    if "taskId" in newRequest:
        
        tempJson = newRequest
        taskId = tempJson["taskId"]
        levels = taskId.split('.')
        if len(levels) > 3:
            print("Working on final answer!!!!!!!!!!!!!!!!!!!")
            promptKind = f"""
                Given the json
                Your output should be the 'finalAnswer' to the task.  
                Output JSON using the schema defined here: \n{schemaDump}\n 
                THERE SHOULD BE NO SUBTASKS!!!
                THE OUTPUT SHOULD BE IN THE FIELD 'finalAnswer' of the JSON.
                ONLY OUTPUT JSON!!!
                Follow the schema exactly.  JSON data must always be enclosed by the ```json``` tag. 
            """
    payload = {
        "role": "user",
        "input": f"""
        {runningRequest}

        {newRequest} 
        {promptKind},
        """,
        "cost":0
        #"schema": schema
    }

    for chunk in runnable.stream(payload):
        if "stream" in chunk:
            print(chunk["stream"], end="")
        if "output" in chunk:
            print("Output: " + chunk["output"])    

    #response = runnable.invoke({"input": "my name is jason", "cost":0})
    #print(response["output"])
    s = str(chunk["output"])
    newInput, runningcost = newJSONfromString(s,runningcost)
    if "subtasks" in newInput:
        for i in range(len(newInput["subtasks"])):
            task = newInput["subtasks"][i]
            #taskString = json.dumps(task)
            #print(f"Task: {taskString}")
            taskID = task["taskId"]
            newInput["subtasks"][i], runningcost = execute_building_chain(task, newInput, runningcost)
    
    return newInput, runningcost

newRequest = "Write a 5 paragraph paper on climate change" 
newInput, newcost = execute_building_chain(newRequest, {}, 0)







print(f"\nnewInput was: " + str(newInput))
print("\ncost was: " + str(newcost))


