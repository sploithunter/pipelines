from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import OpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain_core.pydantic_v1 import BaseModel, Field, validator

from langchain.schema import (
    HumanMessage,
    SystemMessage,
    AIMessage,
)
from langchain_community.chat_models.huggingface import ChatHuggingFace

from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.tools.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)
import json
from langchain_core.runnables import chain

from langchain_core.messages import HumanMessage
from langchain_mistralai.chat_models import ChatMistralAI
 
from colorama import Fore, Back, Style
import os

chatbotTeam_properties = {
     "AdaptiveSystem2": {
        "team_name": "Adaptive AI Team",
        "grading_criteria": {
            "relevance": "Add 1 point if the response is relevant and provides some information related to the user’s inquiry, even if it is incomplete or contains some irrelevant content.",
            "substantial_address": "Add 1 point if the response addresses a substantial portion of the user’s question, but does not completely resolve the query or provide a direct answer.",
            "useful_answer": "Add 1 point if the response answers the basic elements of the user’s question in a useful way, regardless of the writing style.",
            "comprehensive_summary": "Add 1 point if there is NO PLACEHOLDER CONTENT. There is no partial credit for this.  It is complete or it is not.  If the task involved producing code, the code is 100%, fully functional, ready to execute and ACHIEVE ALL REQUESTED FUNCTIONALITY!!!",
            "tailored_response": "Add 1 point for a response that is impeccably tailored to the user’s request, representing the formatting distinctly and could be reproduced given a semantically equivalent input."
        },  
        "agents":{
            "Coordinator": {
                "name": "Adaptive",
                "role": "Coordinator",
                "prompt": "You are the user interface agent.  You have memory capabilities.  Your name is Adaptive.",
                "voice_provider": "elevenlabs",
                "voice": "TdHp3azgcXeQym4dVNqa",
                "AI_model_provider":"openai",
                "model": "gpt4T",
            },
            "Generator": {
                "name": "Peter",
                "role": "Generator",
                "prompt": "You are the response generator.  You have memory capabilities.  You do your best to generate answers and refine them. Your name is Peter.",
                "voice_provider": "elevenlabs",
                "voice": "ZUFs7iYbxrbr0qT34NQK",
                "AI_model_provider":"openai",
                "model": "gpt4T",
            },
            "Critic": {
                "name": "Val",
                "role": "Critic",
                "prompt": "You are a fair but thurough critic.  You do your best to provide crticism for every request/response presented to you. Your name is Val.",
                "voice_provider": "elevenlabs",
                "voice": "fv33WwbYff1fuqd2aMiI",
                "AI_model_provider":"openai",
                "model": "gpt4T",
            }
        }
    }

}
def format_grading_criteria(grading_criteria):
    criteria_strings = []
    for criterion, description in grading_criteria.items():
        criteria_strings.append(f" {description}\n")

    return "\n".join(criteria_strings)


def get_generic_runnable(memory,model_type="mistral-medium", temperature=0.3, promptSysInternal = "you are a helpful assistant" ):
    promptInternal = ChatPromptTemplate.from_messages(
    [
        ("system", promptSysInternal),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
    )

    promptInternalNoMem = ChatPromptTemplate.from_messages(
    [
        ("system", promptSysInternal),
        ("human", "{input}"),
    ]
    )

    #if model is mistral type
    if "mistral" in model_type:

        modelInternal = ChatMistralAI( model=model_type,temperature=temperature )
    elif "gpt" in model_type:


        if model_type == "gpt35":
            modelInternal = ChatOpenAI(temperature=temperature,   model="gpt-3.5-turbo")
        elif model_type == "gpt4T":
            modelInternal = ChatOpenAI(temperature=temperature,   model="gpt-4-turbo-preview")
        else:
            modelInternal = ChatOpenAI(temperature=temperature,   model=model_type)
            if "gpt-4" in model_type:
                model_type = "gpt4T"
            else:
                model_type = "gpt35"


    @chain 
    def generic_chain(inputDict):
        runnable1 = (
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history")
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
        
        memory.save_context(inputs, {"output": fullresponse})
        yield {"output": fullresponse}

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
        
        yield {"output": fullresponse} 
    if memory:
        return generic_chain
    else:
        return generic_chain_no_memory
def create_a_team_simple(team_id):

    # Initialize chatbot teams
    chatbotTeams= {}


    # Process each agent in the team
    team_agents = {}
    if team_id == "AdaptiveSystem2":
        team_properties = chatbotTeam_properties
    else:
        #need to add created teams here, right now, just adaptivesystem2
        team_properties = chatbotTeam_properties
    print(f"Team Properties: {team_properties}")
    for agent_role, agent_properties in team_properties[team_id]["agents"].items():
        # Fetch and unpickle memory for each agent
        print(f"Agent Properties: {agent_properties}")
        model = ChatOpenAI(temperature=.1)
        if agent_role == "Generator":
            memory = ConversationSummaryBufferMemory(llm=model,memory_key="chat_history", max_token_limit=4000, return_messages=True) 

            # Create a runnable instance for the agent
            runnable = get_generic_runnable(memory,model_type=agent_properties["model"])
        else:
            memory = None
            # Create a runnable instance for the agent
            runnable = get_generic_runnable(memory,model_type=agent_properties["model"])
        # Store information for each agent
        team_agents[agent_role] = {
            'runnable': runnable,
            'memory': memory,
            'properties': agent_properties
        }
        print(f"runnable {agent_role} complete")
    return team_agents, team_properties

def fireteam(bot_id, message, audio=False):

    team_agents, team_properties = create_a_team_simple("AdaptiveSystem2")
    

    text_chunk = message
    sentence = ""
    genfullresponse = ""
    critfullresponse = ''
    coordfullresponse = ''
    message_chunk = ""
    iterations = 3


    #get gradingCriteria from cteam_properties['grading_criteria'] and convert to string
    gradingCriteria = format_grading_criteria(team_properties[bot_id]['grading_criteria'])

    # send the user message to everyone in the team
    generatorInput = {"input" : text_chunk +  " \n " + "You will be graded on the following \n " + gradingCriteria}
    
    # Ask the generation from the generator
    output_length = 0
    genfullresponse = ""
    team_member = "Generator"
    if team_agents[team_member]['properties']['model'] == 'gpt35':
        encoding_name = 'gpt-3.5-turbo'
    elif team_agents[team_member]['properties']['model'] == 'gpt4T':
        encoding_name = 'gpt-4'
    else:
        encoding_name = team_agents[team_member]['properties']['model'] 
    chatbot_color = Fore.BLUE
    for generatorchunk in team_agents[team_member]['runnable'].stream(generatorInput):
        if 'stream' in generatorchunk:
                genfullresponse += generatorchunk['stream']
                message_chunk += generatorchunk['stream']
                sentence += generatorchunk['stream']
                
            # print(message_chunk)
                # Check for code block start or end
                if '`' in generatorchunk['stream']:
                # print("continuing")
                    continue

                #print("emitting message chunk: " + message_chunk)
                print(chatbot_color + message_chunk, end='')
                message_chunk = ""


        if audio and any(char in sentence for char in ".!?") and voice_api_key and voice_id != '':
            print(Fore.WHITE + f"sentence: {sentence}")             
            #audio here

            sentence = ""
    print("\n")
    team_agents[team_member]['memory'].save_context(generatorInput, {"outputs": genfullresponse})
    # Catch the response from the generator via genfullresponse
        
    # Evaluate if iteration is complete
    #todo
        
    

    # start the loop for 2 rounds
    while True:
        # Take the generator's response, pass it to the coordinator.  This is the init before the loop
        output_length = 0
        sentence = ""
        coordfullresponse = ""
        team_member = "Critic"
        chatbot_color = Fore.RED
        CriticInput = {"input" : "Consider the following request: \"\n" + text_chunk  + "\"\n and the following response: \"\n" + genfullresponse + "\"\n" + "Grade the answer based on the following rubric: \n " + gradingCriteria + "\nPay particular attention for placeholder comments by outputing every comment step by step, then examining the comments to determine if they may be placeholders. EXAMINE EVERY COMMENT CAREFULLY. Look for words in comments that indicate that the task was not completed fully.  Words in comments like 'Simplified', 'Placeholder', are likely giveaways that something was not completed\n"}    
        
        if team_agents[team_member]['properties']['model'] == 'gpt35':
            encoding_name = 'gpt-3.5-turbo'
        elif team_agents[team_member]['properties']['model'] == 'gpt4T':
            encoding_name = 'gpt-4'
        else:
            encoding_name = team_agents[team_member]['properties']['model'] 

        for criticchunk in team_agents['Critic']['runnable'].stream(CriticInput):  
            if 'stream' in criticchunk:
                    critfullresponse += criticchunk['stream']
                    message_chunk += criticchunk['stream']
                    sentence += criticchunk['stream']
                    
                # print(message_chunk)
                    # Check for code block start or end
                    if '`' in criticchunk['stream']:
                    # print("continuing")
                        continue

                    #print("emitting message chunk: " + message_chunk)
                    print(chatbot_color + message_chunk, end='')
                    message_chunk = ""

            if audio and any(char in sentence for char in ".!?") and voice_api_key and voice_id != '':
                print(Fore.WHITE + f"sentence: {sentence}")         
                # audio here
                
                
                sentence = ""
        print("\n")
        
        # The coordinator evaluates if score is perfect and if so, sends the response to the user
        output_length = 0
        sentence = ""
        coordfullresponse = ""
        team_member = "Coordinator"
        chatbot_color = Fore.GREEN
        
        CoordInput = {"input" : "Consider the following response: \"\n" + genfullresponse  + "\"\n and the following grade: \"\n" + critfullresponse + "\"\n" + "If the grade is perfect, respond with the single word \"PERFECT\" \n" + "Otherwise, respond with total score out of the possible score and the single word \"AGAIN\""}
        if team_agents[team_member]['properties']['model'] == 'gpt35':
            encoding_name = 'gpt-3.5-turbo'
        elif team_agents[team_member]['properties']['model'] == 'gpt4T':
            encoding_name = 'gpt-4'
        else:
            encoding_name = team_agents[team_member]['properties']['model'] 

        for coordchunk in team_agents[team_member]['runnable'].stream(CoordInput):
            if 'stream' in  coordchunk:
                coordfullresponse += coordchunk['stream']
                message_chunk += coordchunk['stream']
                sentence += coordchunk['stream']
                
                # print(message_chunk)
                # Check for code block start or end
                if '`' in coordchunk['stream']:
                # print("continuing")
                    continue

                #print("emitting message chunk: " + message_chunk)
                print(chatbot_color + message_chunk, end='')
                message_chunk = ""

            if audio and any(char in sentence for char in ".!?") and voice_api_key and voice_id != '':
                print(Fore.WHITE + f"sentence: {sentence}")       
                #audio here
                
                
                sentence = ""
        print("\n")
        # If the coordinator says perfect, send the response to the user ie, break the loop

        if 'PERFECT' in coordfullresponse or iterations <= 0:
            message_chunk += "\nFinal Answer: \n" + genfullresponse
            #print("emitting message chunk: " + message_chunk)

            print(chatbot_color + message_chunk)
            message_chunk = ""
            break
        iterations -= 1



        #if not generator again
        generatorInput = {"input" : text_chunk +  " \n " + "Your grade was as follows: \n " + critfullresponse + " \n Please try again"}
    
        # Ask the generation from the generator
        genfullresponse = ""
        critfullresponse = ""
        coordfullresponse = ""
        sentence = ""
        output_length = 0
        team_member = "Generator"
        chatbot_color = Fore.BLUE
        if team_agents[team_member]['properties']['model'] == 'gpt35':
            encoding_name = 'gpt-3.5-turbo'
        elif team_agents[team_member]['properties']['model'] == 'gpt4T':
            encoding_name = 'gpt-4'
        else:
            encoding_name = team_agents[team_member]['properties']['model'] 
        
        for generatorchunk in team_agents[team_member]['runnable'].stream(generatorInput):
            if 'stream' in  generatorchunk:
                genfullresponse += generatorchunk['stream']
                message_chunk += generatorchunk['stream']
                sentence += generatorchunk['stream']
                
                # print(message_chunk)
                # Check for code block start or end
                if '`' in generatorchunk['stream']:
                # print("continuing")
                    continue

                #print("emitting message chunk: " + message_chunk)
                print(chatbot_color + message_chunk, end='')
                message_chunk = ""

            if audio and any(char in sentence for char in ".!?") and voice_api_key and voice_id != '':
                print(Fore.WHITE + f"sentence: {sentence}")         
                #audio Here
                
                
                sentence = ""
        print("\n")
    print(Fore.WHITE + "complete")   
    
import os


# Make sure this is at the very end of your file
script_path = os.path.realpath(__file__)
with open(script_path, "r") as file:
    selfcode = file.read()

prompt = f"Consider: \n ```python\n {selfcode}\n ```\n Refactor this code.  DO NOT LEAVE ANY PLACEHOLDER COMMENTS!"

fireteam("AdaptiveSystem2", prompt, audio = False)