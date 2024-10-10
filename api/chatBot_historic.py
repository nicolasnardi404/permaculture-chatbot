import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAI, OpenAIEmbeddings, ChatOpenAI
from supabase import create_client
from langchain.schema.output_parser import StrOutputParser
from transformers import pipeline
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from image_generator import generate_image, save_image
from datetime import datetime
import random

from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain import LLMChain
from typing import List
import re
import torch
import base64
import requests
from fastapi import HTTPException
from langchain.schema import Document

# Load environment variables from the .env file
load_dotenv(override=True)

# Fetch the environment variables
openAIApiKey = os.getenv("OPENAI_API_KEY")
hfApiKey = os.getenv("HUGGINGFACE_API_KEY")
sbApiKey = os.getenv("SUPABASE_API_KEY")
sbUrl = os.getenv("SUPABASE_URL_LC_CHATBOT")

# Debugging: Check that the keys are being loaded properly
print("OPENAI_API_KEY is set:", bool(openAIApiKey))
print("HUGGINGFACE_API_KEY is set:", bool(hfApiKey))
print("SUPABASE_API_KEY is set:", bool(sbApiKey))
print("SUPABASE_URL_LC_CHATBOT is set:", bool(sbUrl))

client = create_client(sbUrl, sbApiKey)
print("Supabase client created.")

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings()
print("Embeddings initialized.")

vectorStore = SupabaseVectorStore(
    client=client,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents_permaculture",
)
print("Vector store created.")

# Initialize the language model with the API key
llm = ChatOpenAI(api_key=openAIApiKey)
print("Language model initialized.")

# Create the retriever
retriever = vectorStore.as_retriever()
print("Retriever created.")

# Initialize vector stores for each database
vector_store_ecofeminism = SupabaseVectorStore(
    client=client,
    embedding=embeddings,
    table_name="documents_ecofeminism",
    query_name="match_documents_ecofeminism",
)

vector_store_permaculture = SupabaseVectorStore(
    client=client,
    embedding=embeddings,
    table_name="documents_permaculture",
    query_name="match_documents_permaculture",
)

vector_store_mushrooms = SupabaseVectorStore(
    client=client,
    embedding=embeddings,
    table_name="documents_mushrooms",
    query_name="match_documents_mushrooms",
)

# Define the prompt templates
standaloneQuestionTemplate = "Given a question, convert it to a standalone question. question: {question} standalone question:"

documentProcessingTemplate = """
Process the information on: {documents}
You are a chatbot helping people with permaculture and mushrooms. Answer the question: {question}"""

# Create prompt objects
standaloneQuestionPrompt = PromptTemplate.from_template(standaloneQuestionTemplate)
documentProcessingPrompt = PromptTemplate.from_template(documentProcessingTemplate)

# Create the standalone question chain
standaloneQuestionChain = (
    standaloneQuestionPrompt.pipe(llm).pipe(StrOutputParser()).pipe(retriever)
)

# Create the document processing chain
documentProcessingChain = documentProcessingPrompt.pipe(llm).pipe(StrOutputParser())

# Combine the chains
combinedChain = standaloneQuestionChain | documentProcessingChain

# Initialize the zero-shot classification pipeline
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    tokenizer_kwargs={"clean_up_tokenization_spaces": True},
    device=device,
)

# === Agent Integration Starts Here ===


# Define Tools
def get_documents(query: str, category: str):
    """Retrieve documents from the specified category."""
    if category == "ecofeminism":
        function_name = "public.match_documents_ecofeminism"
    elif category == "permaculture":
        function_name = "public.match_documents_permaculture"
    elif category == "mushrooms":
        function_name = "public.match_documents_mushrooms"
    else:
        return "I don't have information on that topic."

    # Replace 'call_function' with the actual function to retrieve documents
    documents = vectorStore.get_relevant_documents(
        query_embedding=query,  # Assuming 'query' is the embedding
        filter=None,  # Adjust as needed
        match_count=10,  # Adjust as needed
    )
    combined_docs = "\n\n".join([doc.page_content for doc in documents])
    return combined_docs


# Define the tools
knowledge_tool = Tool(
    name="KnowledgeBase",
    func=lambda query, category: get_documents(query, category),
    description="Useful for answering questions about ecofeminism, permaculture, and mushrooms.",
)

general_chat_tool = Tool(
    name="GeneralChat",
    func=lambda query: general_chat(query),
    description="Useful for general conversation and questions outside the knowledge base.",
)

tools = [knowledge_tool, general_chat_tool]


# Define the general chat function
def general_chat(query: str) -> str:
    """Handle general chat using the OpenAI model."""
    general_llm = OpenAI(api_key=openAIApiKey, temperature=0.7)
    response = general_llm(query)
    return response


# Initialize the language model for the agent
agent_llm = OpenAI(api_key=openAIApiKey, temperature=0)

# Define the prompt template for the agent
agent_prompt = PromptTemplate.from_template(
    """You are an AI assistant specializing in permaculture, ecofeminism, and mushrooms.

    **Behavior Guidelines:**
    - **Greetings & Small Talk:** When the user's input is a simple greeting or involves small talk, respond appropriately *without* using any tools. **Do not** include `Action`, `Action Input`, or `Observation` fields. Provide only the `Final Answer`.
    - **Specific Questions:** When the user asks a specific question or requests information, utilize the provided tools to generate a comprehensive response. If the question is not about agriculture, mushrooms or permaculture, respond with "I'm sorry, I'm not trained on this topic. Try another question."

    **Available Tools:**
    {tools}

    **Response Format:**
    - **For Greetings & Small Talk:**
        ```
        Final Answer: [Your appropriate response]
        ```
    - **For Specific Questions:**
        ```
        Question: [User's input]
        Thought: [Your reasoning]
        Action: [One of {tool_names}]
        Action Input: [Input for the action]
        Observation: [Result of the action]
        ... (This Thought/Action/Action Input/Observation sequence can repeat N times)
        Thought: I now know the final answer
        Final Answer: [Your final answer to the user's question]
        ```

    **Begin!**

    Question: {input}
    Thought: {agent_scratchpad}
    """
)

# Create the agent
agent = create_react_agent(llm=agent_llm, tools=tools, prompt=agent_prompt)

# Create the agent executor with handle_parsing_errors=True
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True  # Add this line
)

# === Agent Integration Ends Here ===


# Initialize conversation history
conversation_history = {}

str_number = str(random.randint(1000, 9999))


# Function to handle chat interactions
async def chat_interaction(input_text: str, session_id: str = str_number) -> str:
    """
    Handle the chat interaction by processing the input, maintaining history,
    and generating a response using Supabase and OpenAI's API.
    """
    print(f"Received input: {input_text}")

    # Retrieve the conversation history for the session
    history = conversation_history.get(session_id, [])
    print(f"Current history for session {session_id}: {history}")

    # Add the new user message to the history
    history.append({"role": "user", "content": input_text})
    print(f"Updated history after adding user input: {history}")

    try:
        # Prepare messages for the OpenAI API
        messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ] + history

        # Call the OpenAI API with the conversation history using the invoke method
        response = llm.invoke(messages)

        # Extract the assistant's response
        assistant_response = response.content
        print(f"Assistant response: {assistant_response}")

        # Add the assistant's response to the history
        history.append({"role": "assistant", "content": assistant_response})
        print(f"Updated history after adding assistant response: {history}")

        # Save the updated history back to the session
        conversation_history[session_id] = history

        return assistant_response

    except Exception as e:
        print(f"An error occurred during chat interaction: {str(e)}")
        return "I apologize, but I encountered an error while processing your request. Could you please try again?"


# Modified main function to create a chatbot
async def main():
    print("Welcome to the Permaculture Chatbot!")
    print("Type 'exit' to end the conversation.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Thank you for using the Permaculture Chatbot. Goodbye!")
            break

        response = await chat_interaction(user_input)
        print("\nChatbot:", response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Define request model
class ChatInput(BaseModel):
    message: str


# Replace Flask route with FastAPI route
@app.post("/chat")
async def chat_endpoint(chat_input: ChatInput):
    try:
        response = await chat_interaction(chat_input.message)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}


from huggingface_hub import InferenceClient

client_hf = InferenceClient(api_key=hfApiKey)


# New endpoint for generating images
@app.post("/generate-image")
async def generate_image_endpoint(chat_input: ChatInput):
    try:
        print("Starting image generation process")

        print("Initializing text generation")
        print(f"Input message: {chat_input.message}")

        generated_text = client_hf.text_generation(
            model="meta-llama/Llama-3.2-1B-Instruct",
            prompt=f"Based on this conversation, create a short description for an image about permaculture: {chat_input.message}",
            max_new_tokens=50,
            temperature=0.7,
            return_full_text=False,
        )

        print(f"Generated text: {generated_text}")

        prompt = f"Create an image about permaculture based on this description: {generated_text}"

        print("Generating image")
        image_bytes = generate_image(prompt)

        if image_bytes:
            print("Image generated successfully")
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            return {"image_data": f"data:image/jpeg;base64,{base64_image}"}
        else:
            print("Failed to generate image")
            return {"error": "Failed to generate the image"}
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return {"error": str(e)}


# Instructions to run with uvicorn:
# uvicorn chatBot:app --reload
