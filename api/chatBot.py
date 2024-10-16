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
standaloneQuestionTemplate = "Given a question, convert it to a standalone question. question: {question} standalone question:"

documentProcessingTemplate = """
Process the information on: {documents}
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
    general_llm = ChatOpenAI(api_key=openAIApiKey, temperature=0.7)
    response = general_llm(messages)
    return response.content


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


# New function to handle chat interactions with agent
import time
from langchain.callbacks import get_openai_callback

conversation_history = {}

str_number = str(random.randint(1000, 9999))


async def chat_interaction(input_text: str, session_id: str = str_number) -> str:
    print(f"Received input: {input_text}")

    # Retrieve the conversation history for the session
    history = conversation_history.get(session_id, [])
    print(f"Current history for session {session_id}: {history}")

    # Add the new user message to the history
    history.append({"role": "user", "content": input_text})
    print(f"Updated history after adding user input: {history}")

    # Prepare the messages for the model, including the history
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant specializing in permaculture, ecofeminism, and mushrooms. Use the conversation history to provide context-aware responses.",
        }
    ] + history

    # Define candidate labels for classification
    candidate_labels = ["ecofeminism", "permaculture", "mushrooms"]

    # Perform zero-shot classification
    classification_result = classifier(input_text, candidate_labels)

    assistant_response = ""

    # If the classification score is low or the input is not about the specialized topics
    if classification_result["scores"][0] < 0.6:
        # Use the general chat function with the full conversation history
        assistant_response = general_chat(messages)
    else:
        # Check if the highest classification score is above the threshold
        if classification_result["scores"][0] >= 0.6:
            # Extract the most voted label
            most_voted_label = classification_result["labels"][0]
            print("Most voted label:", most_voted_label)

            # Run a second classification using the remaining labels
            candidate_labels_without_most_voted = [
                label for label in candidate_labels if label != most_voted_label
            ]
            second_classification_result = classifier(
                input_text, candidate_labels_without_most_voted
            )
            print("Second classification result:", second_classification_result)

            # Determine which databases to query
            databases_to_query = [most_voted_label]
            if second_classification_result["scores"][0] >= 0.7:
                second_most_voted_label = second_classification_result["labels"][0]
                databases_to_query.append(second_most_voted_label)
                print("Second most voted label:", second_most_voted_label)

            # Query the appropriate databases and combine the results
            all_documents = []
            for db_label in databases_to_query:
                if db_label == "ecofeminism":
                    vector_store = vector_store_ecofeminism
                elif db_label == "permaculture":
                    vector_store = vector_store_permaculture
                elif db_label == "mushrooms":
                    vector_store = vector_store_mushrooms

                print(f"Querying {db_label} database.")
                retriever = vector_store.as_retriever()
                documents = retriever.get_relevant_documents(input_text)
                all_documents.extend(documents)

            # Combine all retrieved documents
            combined_docs = "\n\n".join([doc.page_content for doc in all_documents])

            if combined_docs:
                try:
                    response = documentProcessingChain.invoke(
                        {
                            "documents": combined_docs,
                            "question": input_text,
                            "chat_history": messages,
                        }
                    )
                    assistant_response = (
                        response if isinstance(response, str) else response.content
                    )
                    print(f"Assistant response: {assistant_response}")
                    print(
                        f"Combined response from {', '.join(databases_to_query)} databases:",
                        response,
                    )
                except Exception as e:
                    print(f"An error occurred during document processing: {str(e)}")
                    assistant_response = "I apologize, but I encountered an error while processing the documents. Could you please try again?"
            else:
                # If no documents found, use the general chat tool with history
                print("No relevant documents found. Using general chat with history.")
                assistant_response = general_chat(messages)
        else:
            # If classification score is low, use the agent to handle the query
            print("Low classification confidence. Using agent to handle the query.")
            try:
                with get_openai_callback() as cb:
                    start_time = time.time()
                    agent_response = agent_executor.invoke(
                        {"input": input_text},
                        return_only_outputs=True,
                        timeout=30,  # Set a 30-second timeout
                    )
                    end_time = time.time()

                print(f"Agent execution time: {end_time - start_time:.2f} seconds")
                print(f"Total tokens used: {cb.total_tokens}")
                print(f"Prompt tokens: {cb.prompt_tokens}")
                print(f"Completion tokens: {cb.completion_tokens}")
                print(f"Total cost: ${cb.total_cost:.4f}")

                if isinstance(agent_response, dict) and "output" in agent_response:
                    assistant_response = agent_response["output"]
                else:
                    print(f"Unexpected agent response format: {agent_response}")
                    assistant_response = "I'm sorry, I couldn't generate a proper response. Could you please rephrase your question?"

            except Exception as e:
                print(f"An error occurred during agent execution: {str(e)}")
                assistant_response = "I apologize, but I encountered an error while processing your request. Could you please try again?"

    # Add the assistant's response to the history
    history.append({"role": "assistant", "content": assistant_response})
    print(f"Updated history after adding assistant response: {history}")

    # Update the conversation history for the session
    conversation_history[session_id] = history

    return assistant_response


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
            prompt=f"Based on this conversation, create a short description for an image: {chat_input.message}",
            max_new_tokens=50,
            temperature=0.7,
            return_full_text=False,
        )

        print(f"Generated text: {generated_text}")

        prompt = f"Create an image based on this description: {generated_text}"

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

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("chatBot:app", host="0.0.0.0", port=port, reload=False)
