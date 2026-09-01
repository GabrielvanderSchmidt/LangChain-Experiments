import os
import re
import time
from pathlib import Path

import arxiv
import requests
from io import BytesIO
from pypdf import PdfReader
from typing import List, Dict, TypedDict
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Setup and ensure the download directory exists
downloads_dir = Path("./downloaded_papers")
downloads_dir.mkdir(parents=True, exist_ok=True)

# ==========================================
# 1. Define the Global State and Schemas
# ==========================================

class ResearchState(TypedDict):
    topic: str
    search_queries: List[str]
    source_materials: List[Dict[str, str]] # list of {"id": "...", "title": "...", "text": "..."}
    densified_notes: List[str]
    final_prospectus: str

class SearchQueries(BaseModel):
    queries: List[str] = Field(description="A list of 55555imized arXiv search queries.")

# ==========================================
# 2. Configure the Local vLLM Connection
# ==========================================

# By pointing ChatOpenAI to localhost:8000, we use local Llama 3 as if it were GPT-4.
llm = ChatOpenAI(
    model="qwen3:4b-q4_K_M",
    api_key="EMPTY", # vLLM doesn't require a key
    base_url="http://ollama:11434/v1",
    temperature=0.1
)

# ==========================================
# 3. Define the Graph Nodes (The Logic)
# ==========================================

def query_planner(state: ResearchState) -> dict:
    """Uses the LLM to generate arXiv-specific Boolean queries."""
    print("--- PLANNING QUERIES ---")
    
    # We use .with_structured_output to force the LLM to return our Pydantic schema
    planner_llm = llm.with_structured_output(SearchQueries)
    
    prompt = f"""
    You are an expert academic librarian. The user is researching: "{state['topic']}".
    Generate 5 distinct arXiv search queries. Use arXiv's syntax (e.g., ti:quantization AND all:LLM).
    """
    
    result = planner_llm.invoke(prompt)
    return {"search_queries": result.queries}

def arxiv_discovery(state: ResearchState) -> dict:
    """Executes the queries against the arXiv API."""
    print("--- DISCOVERING PAPERS ---")
    
    client = arxiv.Client()
    unique_papers = {}
    
    for query in state['search_queries']:
        search = arxiv.Search(
            query=query,
            max_results=10, # Keep low for the project scope
            sort_by=arxiv.SortCriterion.Relevance
        )
        for result in client.results(search):
            if result.entry_id not in unique_papers:
                unique_papers[result.entry_id] = {
                    "id": result.entry_id.split('/')[-1],
                    "title": result.title,
                    "pdf_url": result.pdf_url,
                    "text": "" # Will be filled in the next node
                }
                
    return {"source_materials": list(unique_papers.values())}

def fetch_and_scrape(state: ResearchState) -> dict:
    """Downloads PDFs and extracts text (First 5 pages to save context window)."""
    print("--- FETCHING & SCRAPING PDFs ---")
    
    materials = state["source_materials"]
    for paper in materials:
        print(f"Downloading: {paper['title']}")
        try:
            response = requests.get(paper["pdf_url"], timeout=10)
            safe_id = re.sub(r'[^\w\-_\.]', '_', paper["id"])
            file_path = downloads_dir / f"{safe_id}.pdf"

            # Save to disk only if it doesn't already exist
            if not file_path.exists():
                with open(file_path, "wb") as f:
                    f.write(response.content)

            # Extract only the first 5 pages (usually Abstract, Intro, Methodology)
            pdf_file = BytesIO(response.content)
            reader = PdfReader(pdf_file)
            extracted_text = ""
            for i in range(min(5, len(reader.pages))):
                extracted_text += reader.pages[i].extract_text() + "\n"
            
            paper["text"] = extracted_text

            print("Sleeping for 5 seconds...")
            time.sleep(5)
        except Exception as e:
            print(f"Failed to scrape {paper['id']}: {e}")
            paper["text"] = "Extraction failed."
            
    return {"source_materials": materials}

def precompile_notes(state: ResearchState) -> dict:
    """The Map-Reduce step. Compresses massive raw text into dense notes."""
    print("--- PRECOMPILING DATA ---")
    
    prompt = PromptTemplate.from_template("""
    Analyze this academic paper text regarding the topic: {topic}.
    Title: {title}
    
    Extract ONLY:
    1. The core methodology.
    2. Key metrics and quantitative results.
    3. Major conclusions.
    Ignore all references, boilerplate, and unrelated data.
    
    Raw Text:
    {text}
    """)
    
    notes = []
    # Note: In a true production app, you would run these llm.invoke() calls 
    # asynchronously using asyncio.gather to leverage vLLM's continuous batching.
    for paper in state["source_materials"]:
        if paper["text"] and paper["text"] != "Extraction failed.":
            print(f"Compressing: {paper['title']}")
            chain = prompt | llm
            summary = chain.invoke({
                "topic": state["topic"],
                "title": paper["title"],
                "text": paper["text"]
            })
            notes.append(f"Source: {paper['title']} ({paper['id']})\n{summary.content}")
            
    return {"densified_notes": notes}

def synthesize_prospectus(state: ResearchState) -> dict:
    """Writes the final literature review."""
    print("--- SYNTHESIZING FINAL REVIEW ---")
    
    prompt = PromptTemplate.from_template("""
    You are a Senior Deep Learning Researcher writing a literature prospectus on: {topic}.
    Using ONLY the precompiled notes below, write a cohesive, comprehensive state-of-the-art review.
    You MUST cite your claims using the Source IDs provided in the notes.
    
    Precompiled Notes:
    {notes}
    """)
    
    combined_notes = "\n\n---\n\n".join(state["densified_notes"])
    final_draft = llm.invoke(prompt.format(topic=state["topic"], notes=combined_notes))
    
    return {"final_prospectus": final_draft.content}

# ==========================================
# 4. Assemble and Compile the Graph
# ==========================================

workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("query_planner", query_planner)
workflow.add_node("arxiv_discovery", arxiv_discovery)
workflow.add_node("fetch_and_scrape", fetch_and_scrape)
workflow.add_node("precompile_notes", precompile_notes)
workflow.add_node("synthesize_prospectus", synthesize_prospectus)

# Wire the edges (Linear sequence for this DAG)
workflow.add_edge(START, "query_planner")
workflow.add_edge("query_planner", "arxiv_discovery")
workflow.add_edge("arxiv_discovery", "fetch_and_scrape")
workflow.add_edge("fetch_and_scrape", "precompile_notes")
workflow.add_edge("precompile_notes", "synthesize_prospectus")
workflow.add_edge("synthesize_prospectus", END)

# Compile the graph
app = workflow.compile()

# ==========================================
# 5. Execution
# ==========================================
if __name__ == "__main__":
    initial_state = {"topic": "Transformer optimization techniques to reduce training and inference cost"}
    
    # Run the graph
    final_state = app.invoke(initial_state)
    
    print("\n\n" + "="*50)
    print("FINAL LITERATURE PROSPECTUS")
    print("="*50)
    print(final_state["final_prospectus"])
