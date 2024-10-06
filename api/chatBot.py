import os
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from supabase import create_client
from langchain.schema.output_parser import StrOutputParser
from transformers import pipeline
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio

# Load environment variables from .env file, allowing overrides
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv(override=True)

# Fetch the environment variables
openAIApiKey = os.getenv("OPENAI_API_KEY")
hfApiKey = os.getenv("HUGGINGFACE_API_KEY")
sbApiKey = os.getenv("SUPABASE_API_KEY")
sbUrl = os.getenv("SUPABASE_URL_LC_CHATBOT")

# Debugging: Check that the keys are being loaded properly (make sure to remove print statements before deployment)
print("OPENAI_API_KEY:", openAIApiKey)
print("HUGGINGFACE_API_KEY:", hfApiKey)
print("SUPABASE_API_KEY:", sbApiKey)
print("SUPABASE_URL_LC_CHATBOT:", sbUrl)


client = create_client(sbUrl, sbApiKey)
print("Supabase client created.")

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings()  # Make sure embeddings are set correctly
print("Embeddings initialized.")

vectorStore = SupabaseVectorStore(
    client=client,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents",
)
print("Vector store created.")

# Update the import statement for ChatOpenAI
from langchain_openai import ChatOpenAI

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
you are a chat bot helping people with permaculture and mushrooms. answer the question: {question}"""

# documentProcessingTemplate = """
# Process the information on: {documents}
# and you are professional political journalist. make sure to make the right citations and references to the sources. Don't make up any information. write and article on: {question}"""

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
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


# New function to handle chat interactions
async def chat_interaction(input_text):
    # Define candidate labels for classification
    candidate_labels = ["ecofeminism", "permaculture", "mushrooms"]

    # Perform zero-shot classification
    classification_result = classifier(input_text, candidate_labels)
    print("First classification result:", classification_result)

    # Check if the highest classification score is above the threshold
    if classification_result["scores"][0] >= 0.5:
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
        if second_classification_result["scores"][0] >= 0.5:
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

        # Create a new combined chain with the updated retriever
        response = documentProcessingChain.invoke(
            {"documents": combined_docs, "question": input_text}
        )
        print(
            f"Combined response from {', '.join(databases_to_query)} databases:",
            response,
        )
        return response
    else:
        return "The question is not related to our model."


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


# Remove the Flask run block
# if __name__ == "__main__":
#     app.run(debug=True, port=8000)

# Instead, you would run this with uvicorn:
# uvicorn chatBot:app --reload
