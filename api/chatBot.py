import os
from dotenv import load_dotenv
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from supabase import create_client
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import requests
import base64
from huggingface_hub import InferenceClient
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from typing import List
from langchain.tools import Tool

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
    query_name="match_documents",
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
standaloneQuestionTemplate = "Given a question, convert it to a standalone question. if the question is another language convert it to english before, convert to standalone question: {question} standalone question:"

documentProcessingTemplate = """
Process the information on: {documents}
Respect the language the user start the conversation and answer in that language.
Also be aware of the historic of this conversation: {chat_history}
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


logging.basicConfig(level=logging.INFO)


def zero_shot_classify(text, candidate_labels):
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
    payload = {"inputs": text, "parameters": {"candidate_labels": candidate_labels}}
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        result = response.json()
        logging.info(f"Classification result: {result}")
        return result
    except requests.exceptions.RequestException as e:
        logging.error(f"Error in zero_shot_classify: {e}")
        return None


# Define Tools
def get_documents(query: str, category: str):
    """Retrieve documents from the specified category."""
    if category == "ecofeminism":
        vector_store = vector_store_ecofeminism
    elif category == "permaculture":
        vector_store = vector_store_permaculture
    elif category == "mushrooms":
        vector_store = vector_store_mushrooms
    else:
        return "I don't have information on that topic."
    retriever = vector_store.as_retriever()
    documents = retriever.get_relevant_documents(query)
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
def general_chat(messages: List[dict]) -> str:
    """Handle general chat using the OpenAI model with conversation history."""
    system_message = {
        "role": "system",
        "content": "You are a helpful assistant. Remember details from the conversation history, including names and context that users share. When asked about information that was previously shared in the conversation, refer back to it."
    }
    # Add system message at the start of the conversation
    full_messages = [system_message] + messages
    general_llm = ChatOpenAI(api_key=openAIApiKey, temperature=0.7)
    response = general_llm(full_messages)
    return response.content


# Initialize the language model for the agent
agent_llm = ChatOpenAI(api_key=openAIApiKey, temperature=0)

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
from langchain.agents import create_react_agent

agent = create_react_agent(llm=agent_llm, tools=tools, prompt=agent_prompt)

# Create the agent executor with handle_parsing_errors=True
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

# === Agent Integration Ends Here ===


# New function to handle chat interactions with agent
import time
from langchain.callbacks import get_openai_callback

chat_histories = {}

import random

str_number = str(random.randint(1000, 9999))


# Define this function before chat_interaction
def get_documents_for_category(query: str, category: str):
    """Helper function to get documents for a specific category"""
    print(f"Getting documents for category: {category}")
    try:
        if category == "ecofeminism":
            retriever = vector_store_ecofeminism.as_retriever()
        elif category == "permaculture":
            retriever = vector_store_permaculture.as_retriever()
        elif category == "mushrooms":
            retriever = vector_store_mushrooms.as_retriever()
        else:
            print(f"Unknown category: {category}")
            return []
        
        documents = retriever.get_relevant_documents(query)
        print(f"Retrieved {len(documents)} documents for {category}")
        return documents
    except Exception as e:
        print(f"Error retrieving documents for {category}: {e}")
        return []

async def chat_interaction(
    input_text: str,
    session_id: str,
    client_id: str | None = None
) -> str:
    try:
        print(f"\n=== Starting New Chat Interaction ===")
        print(f"Input text: {input_text}")
        
        # Get existing history from the chat_histories dictionary
        history = chat_histories.get(session_id, [])
        print(f"\n=== Current Chat History ===")
        print(f"History: {history}")
        
        # Update history with new user input
        history.append({"role": "user", "content": input_text})
        chat_histories[session_id] = history
        
        # Define candidate labels for classification
        candidate_labels = ["ecofeminism", "permaculture", "mushrooms"]

        # Perform zero-shot classification
        print(f"Performing zero-shot classification for input: {input_text}")
        classification_result = zero_shot_classify(input_text, candidate_labels)
        logging.info(f"First classification result: {classification_result}")

        if classification_result is None:
            print("Classification failed, using general chat")
            assistant_response = general_chat(messages=history)
            history.append({"role": "assistant", "content": assistant_response})
            return assistant_response
        
        # Check if this is likely a general conversation
        highest_score = classification_result["scores"][0]
        print(f"\n=== Classification Check ===")
        print(f"Highest classification score: {highest_score}")
        
        # If all scores are very low, treat as general conversation
        if highest_score < 0.35:  # Threshold for general chat
            print("Scores too low, switching to general chat")
            assistant_response = general_chat(messages=history)
            history.append({"role": "assistant", "content": assistant_response})
            return assistant_response
            
        # Otherwise, proceed with RAG process
        all_documents = []
        
        # First Classification
        print("\n=== First Classification Process ===")
        first_label = classification_result["labels"][0]
        first_score = classification_result["scores"][0]
        print(f"Primary topic: {first_label} with score: {first_score}")
        
        if first_score >= 0.35:
            print(f"\n=== Retrieving Documents for {first_label} ===")
            first_documents = get_documents_for_category(input_text, first_label)
            if first_documents:
                all_documents.extend(first_documents)
                print(f"Retrieved {len(first_documents)} documents from {first_label}")
        
        # Second Classification
        remaining_labels = [label for label in ["ecofeminism", "permaculture", "mushrooms"] 
                          if label != first_label]
        second_classification = zero_shot_classify(input_text, remaining_labels)
        
        if second_classification:
            print("\n=== Second Classification Process ===")
            second_label = second_classification["labels"][0]
            second_score = second_classification["scores"][0]
            print(f"Secondary topic: {second_label} with score: {second_score}")
            
            if second_score >= 0.5:
                print(f"\n=== Retrieving Documents for {second_label} ===")
                second_documents = get_documents_for_category(input_text, second_label)
                if second_documents:
                    all_documents.extend(second_documents)
                    print(f"Retrieved {len(second_documents)} documents from {second_label}")

        # Process documents or fall back to general chat
        if all_documents:
            print(f"\n=== Processing Combined Documents ===")
            print(f"Total documents retrieved: {len(all_documents)}")
            
            combined_docs = "\n\n".join([doc.page_content for doc in all_documents])
            print(f"Total combined document length: {len(combined_docs)} characters")

            print("\n=== Generating Response ===")
            response = documentProcessingChain.invoke({
                "documents": combined_docs,
                "question": input_text,
                "chat_history": history,
            })
            return response if isinstance(response, str) else response.content
        else:
            print("\nNo relevant documents found, using general chat")
            return general_chat(messages=history)

    except Exception as e:
        print(f"Error in chat_interaction: {e}")
        logging.error(f"Error in chat_interaction: {e}")
        return f"I apologize, but I encountered an error while processing your request. Please try again."


# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Define request model
class ChatInput(BaseModel):
    message: str
    session_id: str | None = None
    client_id: str | None = None


# Replace Flask route with FastAPI route
@app.post("/chat")
async def chat_endpoint(chat_input: ChatInput):
    try:
        # Use the provided session_id or generate a new one
        session_id = chat_input.session_id or str(random.randint(1000, 9999))
        
        response = await chat_interaction(
            input_text=chat_input.message,
            session_id=session_id,
            client_id=chat_input.client_id
        )
        
        return {
            "response": response,
            "session_id": session_id  # Return the session_id to the client
        }
    except Exception as e:
        logging.error(f"Error in chat endpoint: {e}")
        return {"error": str(e)}


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
            prompt=f"Based on this conversation, create a short description for an image: {chat_input.message}",
            max_new_tokens=50,
            temperature=0.7,
            return_full_text=False,
        )

        print(f"Generated text: {generated_text}")

        prompt = f"Create an image on biodiversity and nature based on this description: {generated_text}"

        from image_generator import generate_image

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


# Add a cleanup function to prevent memory leaks
from datetime import datetime, timedelta

def cleanup_old_conversations():
    """Remove conversation histories older than 24 hours"""
    current_time = datetime.utcnow()
    for session_id in list(chat_histories.keys()):
        history = chat_histories[session_id]
        if history:
            last_message = history[-1]
            if "timestamp" in last_message:
                last_time = datetime.fromisoformat(last_message["timestamp"])
                if current_time - last_time > timedelta(hours=24):
                    del chat_histories[session_id]
                    logging.info(f"Cleaned up conversation history for session {session_id}")

# Add periodic cleanup
import asyncio

async def periodic_cleanup():
    while True:
        cleanup_old_conversations()
        await asyncio.sleep(3600)  # Run every hour

# Instructions to run with uvicorn:
# uvicorn chatBot:app --reload

if __name__ == "__main__":
    import uvicorn
    
    # Start the cleanup task
    asyncio.create_task(periodic_cleanup())
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("chatBot:app", host="0.0.0.0", port=port, reload=False)
