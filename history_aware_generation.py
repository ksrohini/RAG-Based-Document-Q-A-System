import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Connect to your document database
persistent_directory = "db/chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"}
)

# Set up AI model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",
    max_new_tokens=512,
    temperature=0.3,
    huggingfacehub_api_token=hf_token,
)
model = ChatHuggingFace(llm=llm)

# Store our conversation as messages
chat_history = []


def safe_invoke(messages, label="model call"):
    """Wrapper that prints the FULL error details if the API call fails."""
    try:
        return model.invoke(messages)
    except Exception as e:
        print(f"\n!!! ERROR during {label} !!!")
        print("Exception:", repr(e))
        response = getattr(e, "response", None)
        if response is not None:
            print("Response status:", getattr(response, "status_code", "N/A"))
            print("Response body:", getattr(response, "text", "N/A"))
        raise


def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    # Step 1: Make the question clear using conversation history
    if chat_history:
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        result = safe_invoke(messages, label="question rewriting")
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    # Step 2: Find relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"  Doc {i}: {preview}...")

    # Step 3: Create final prompt
    docs_text = "\n".join([f"- {doc.page_content}" for doc in docs])
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

Documents:
{docs_text}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

    # Step 4: Get the answer
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]

    result = safe_invoke(messages, label="answer generation")
    answer = result.content

    # Step 5: Remember this conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")
    return answer


def start_chat():
    print("Ask me questions! Type 'quit' to exit.")

    while True:
        question = input("\nYour question: ")

        if question.lower() == 'quit':
            print("Goodbye!")
            break

        ask_question(question)


if __name__ == "__main__":
    start_chat()