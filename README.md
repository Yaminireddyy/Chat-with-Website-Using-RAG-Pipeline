# Chat-with-Website-Using-RAG-Pipeline
The goal is to implement a Retrieval-Augmented Generation (RAG) pipeline that allows users to interact with structured and unstructured data extracted from websites
The system will crawl,scrape, and store website content, convert it into embeddings, and store it in a vector database.Users can query the system for information and receive accurate, context-rich responses generated by a selected LLM

Functional Requirements
1. Data Ingestion
    • Input: URLs or list of websites to crawl/scrape.
    • Process:
       o Crawl and scrape content from target websites.
       o Extract key data fields, metadata, and textual content.
       o Segment content into chunks for better granularity.
       o Convert chunks into vector embeddings using a pre-trained embedding model.
       o Store embeddings in a vector database with associated metadata for eFicient
          retrieval.
2. Query Handling
    • Input: User's natural language question.
    • Process:
        o Convert the user's query into vector embeddings using the same embedding
          model.
        o Perform a similarity search in the vector database to retrieve the most relevant
          chunks.
        o Pass the retrieved chunks to the LLM along with a prompt or agentic context to
          generate a detailed response.
3. Response Generation
   • Input: Relevant information retrieved from the vector database and the user query.
   • Process:
       o Use the LLM with retrieval-augmented prompts to produce responses with exact
          values and context.
       o Ensure factuality by incorporating retrieved data directly into the response.
