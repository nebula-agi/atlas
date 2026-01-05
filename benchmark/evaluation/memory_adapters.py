"""
Memory system adapters for benchmark evaluation.

Each adapter implements a common interface for:
1. Ingesting conversation sessions
2. Searching/retrieving relevant context for a query
3. Generating an answer using an LLM with retrieved context
"""

import os
import json
import hashlib
import time
import requests
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path

# Try importing optional dependencies
try:
    from mem0 import MemoryClient
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False

try:
    from supermemory import Supermemory
    SUPERMEMORY_AVAILABLE = True
except ImportError:
    SUPERMEMORY_AVAILABLE = False

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except (ImportError, RuntimeError):
    # RuntimeError catches SQLite version issues on Windows
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from google import genai
    from google.genai import types
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    from nebula import Nebula
    NEBULA_AVAILABLE = True
except ImportError:
    NEBULA_AVAILABLE = False


# ============================================================================
# LLM Client Configuration
# Priority: Google/Vertex (Gemini 2.5 Flash) > OpenRouter > OpenAI
# ============================================================================

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openai/gpt-4o"  # GPT-4o via OpenRouter
GEMINI_MODEL = "gemini-2.5-flash"  # Gemini 2.5 Flash


def get_google_client():
    """
    Create a Google GenAI client using API key or Application Default Credentials.
    Prioritizes API key if available, falls back to Vertex AI ADC.
    """
    if not GOOGLE_AVAILABLE:
        raise ImportError("google-genai not installed. Run: pip install google-genai")
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)
    else:
        # Use Application Default Credentials for Vertex AI
        return genai.Client(vertexai=True)


def call_google_llm(
    user_message: str, 
    system_message: str = None,
    max_tokens: int = 500, 
    temperature: float = 0.1
) -> str:
    """
    Call Google Gemini 2.5 Flash via API key or Vertex AI.
    
    Args:
        user_message: The user message to send
        system_message: Optional system message with context/instructions
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
    
    Returns:
        The generated text response
    """
    try:
        client = get_google_client()
        
        # Build contents with system instruction if provided
        if system_message:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_message,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    system_instruction=system_message,
                ),
            )
        else:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_message,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )
        
        # Extract text from response
        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            return part.text.strip()
        return ""
    except Exception as e:
        print(f"[Google] Error: {e}")
        return ""


def call_openrouter(
    user_message: str, 
    system_message: str = None,
    max_tokens: int = 500, 
    temperature: float = 0.1
) -> str:
    """
    Call OpenRouter API with GPT-4o (fallback).
    
    Args:
        user_message: The user message to send
        system_message: Optional system message with context/instructions
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
    
    Returns:
        The generated text response
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in environment")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    # Build messages with system message if provided
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})
    
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content.strip()
    
    except requests.exceptions.RequestException as e:
        print(f"[OpenRouter] Request error: {e}")
        return ""
    except (KeyError, IndexError) as e:
        print(f"[OpenRouter] Response parsing error: {e}")
        return ""


def get_llm_provider() -> str:
    """
    Determine which LLM provider to use based on available credentials.
    Priority: Google (API key or Vertex ADC) > OpenRouter > OpenAI
    """
    # Check for Google API key first
    if GOOGLE_AVAILABLE and os.environ.get("GOOGLE_API_KEY"):
        return "google"
    # Check for Vertex AI (ADC) - try to create client to verify
    elif GOOGLE_AVAILABLE:
        try:
            # Test if we can create a Vertex client (ADC available)
            _ = genai.Client(vertexai=True)
            return "google"
        except Exception:
            pass
    # Fallback to OpenRouter
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    elif OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
        return "openai"
    else:
        raise ValueError(
            "No LLM credentials found. Set one of: GOOGLE_API_KEY (or Vertex ADC), OPENROUTER_API_KEY, or OPENAI_API_KEY"
        )


class MemoryAdapter(ABC):
    """Abstract base class for memory system adapters."""
    
    def __init__(self, user_id: str = "benchmark_user"):
        self.user_id = user_id
    
    @abstractmethod
    def ingest_sessions(self, sessions: List[Dict[str, Any]]) -> None:
        """Ingest conversation sessions into the memory system."""
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[str]:
        """Search for relevant memories given a query."""
        pass
    
    @abstractmethod
    def generate_answer(self, question: str, context: List[str]) -> str:
        """Generate an answer to the question using retrieved context."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all memories for clean state between benchmarks."""
        pass
    
    def format_session_for_ingestion(self, session: Dict[str, Any]) -> str:
        """Convert a session to a text format for memory ingestion."""
        session_id = session.get("id", "unknown")
        timestamp = session.get("timestamp", 0)
        turns = session.get("turns", [])
        
        lines = [f"[Session {session_id}, Time: {timestamp}]"]
        for turn in turns:
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            lines.append(f"{speaker}: {text}")
        
        return "\n".join(lines)
    
    def format_sessions_as_messages(self, sessions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Convert sessions to message format for memory APIs that expect messages."""
        messages = []
        for session in sessions:
            for turn in session.get("turns", []):
                role = "user" if turn.get("speaker") == "user" else "assistant"
                messages.append({
                    "role": role,
                    "content": turn.get("text", "")
                })
        return messages


class Mem0Adapter(MemoryAdapter):
    """Adapter for Mem0 memory system."""
    
    def __init__(self, api_key: Optional[str] = None, user_id: str = "benchmark_user"):
        super().__init__(user_id)
        
        if not MEM0_AVAILABLE:
            raise ImportError("mem0 package not installed. Run: pip install mem0ai")
        
        self.api_key = api_key or os.environ.get("MEM0_API_KEY")
        if not self.api_key:
            raise ValueError("MEM0_API_KEY not provided or set in environment")
        
        self.client = MemoryClient(api_key=self.api_key)
        self.llm_provider = get_llm_provider()
        self.memory_count = 0
        self.stored_memories = []  # Will be populated after ingestion
    
    def ingest_sessions(self, sessions: List[Dict[str, Any]]) -> None:
        """Ingest sessions into Mem0 with fixed wait for processing."""
        print(f"[Mem0] Ingesting {len(sessions)} sessions for user_id: {self.user_id}")
        
        # Sort sessions by timestamp to maintain temporal ordering
        sorted_sessions = sorted(sessions, key=lambda s: s.get("timestamp", 0))
        
        total_turns = 0
        
        # Submit all sessions
        for session_idx, session in enumerate(sorted_sessions):
            session_id = session.get("id", f"session_{session_idx}")
            session_timestamp = session.get("timestamp", session_idx)
            turns = session.get("turns", [])
            total_turns += len(turns)
            
            # Create a rich conversation text that captures the full context
            conversation_text = f"=== Conversation Session {session_id} (Time: {session_timestamp}) ===\n\n"
            
            for turn in turns:
                speaker = turn.get("speaker", "user")
                text = turn.get("text", "")
                speaker_label = "User" if speaker == "user" else "Assistant"
                conversation_text += f"{speaker_label}: {text}\n\n"
            
            try:
                result = self.client.add(
                    messages=[{"role": "user", "content": conversation_text}],
                    user_id=self.user_id,
                    metadata={
                        "session_id": session_id,
                        "timestamp": session_timestamp,
                        "num_turns": len(turns),
                    }
                )
                print(f"[Mem0] Session {session_id}: {len(turns)} turns - submitted")
            except Exception as e:
                print(f"[Mem0] Error ingesting session {session_id}: {e}")
        
        print(f"[Mem0] Submitted {len(sessions)} sessions with {total_turns} total turns")
        
        # Fixed wait for Mem0 to process all memories
        # Mem0 has high async latency (~60-90s per session)
        wait_time = 360  # 6 minutes
        print(f"[Mem0] Waiting {wait_time}s for memory processing...")
        
        # Show progress
        for remaining in range(wait_time, 0, -30):
            print(f"[Mem0] ...{remaining}s remaining")
            time.sleep(min(30, remaining))
        
        print("[Mem0] Wait complete.")
        
        # Verify memories were stored and display them
        try:
            # API v2 requires explicit filters format
            filters = {"AND": [{"user_id": self.user_id}]}
            all_memories = self.client.get_all(filters=filters)
            if all_memories:
                mem_list = all_memories if isinstance(all_memories, list) else all_memories.get("results", [])
                print(f"[Mem0] Verified: {len(mem_list)} memories stored for user")
                
                # Display all extracted memories
                print("\n" + "=" * 60)
                print("EXTRACTED MEMORIES:")
                print("=" * 60)
                self.stored_memories = []  # Store for JSON output
                for i, mem in enumerate(mem_list):
                    if isinstance(mem, dict):
                        memory_text = mem.get("memory") or mem.get("text") or mem.get("content") or str(mem)
                        memory_id = mem.get("id", "")
                        created_at = mem.get("created_at", "")
                        self.stored_memories.append({
                            "id": memory_id,
                            "memory": memory_text,
                            "created_at": created_at,
                        })
                        print(f"  [{i+1}] {memory_text}")
                    else:
                        self.stored_memories.append({"memory": str(mem)})
                        print(f"  [{i+1}] {mem}")
                print("=" * 60 + "\n")
        except Exception as e:
            print(f"[Mem0] Could not verify memories: {e}")
            self.stored_memories = []
    
    def search(self, query: str, top_k: int = 5) -> List[str]:
        """Search Mem0 for relevant memories."""
        # API v2 requires explicit filters format with user_id
        filters = {"AND": [{"user_id": self.user_id}]}
        
        try:
            results = self.client.search(
                query=query,
                filters=filters
            )
            
            # Extract memory texts from results
            memories = self._extract_memories_from_results(results)
            return memories
            
        except Exception as e:
            print(f"[Mem0] Search error: {e}")
            # Try fallback: get all memories for this user
            try:
                all_memories = self.client.get_all(filters=filters)
                if all_memories:
                    memories = self._extract_memories_from_results(all_memories, limit=top_k)
                    if memories:
                        print(f"[Mem0] Using fallback: returning {len(memories)} memories")
                        return memories
            except Exception as e2:
                print(f"[Mem0] Fallback also failed: {e2}")
            return []
    
    def _extract_memories_from_results(self, results, limit: int = None) -> List[str]:
        """Extract memory text from Mem0 API results."""
        memories = []
        
        # Handle list response
        if isinstance(results, list):
            for r in results:
                if isinstance(r, dict):
                    memory_text = r.get("memory") or r.get("text") or r.get("content")
                    if memory_text:
                        memories.append(memory_text)
                elif isinstance(r, str):
                    memories.append(r)
        
        # Handle dict response with "results" key
        elif isinstance(results, dict):
            results_list = results.get("results") or results.get("memories") or []
            for r in results_list:
                if isinstance(r, dict):
                    memory_text = r.get("memory") or r.get("text") or r.get("content")
                    if memory_text:
                        memories.append(memory_text)
                elif isinstance(r, str):
                    memories.append(r)
        
        if limit:
            memories = memories[:limit]
        
        return memories
    
    def generate_answer(self, question: str, context: List[str]) -> str:
        """Generate answer using LLM with Mem0 context in system prompt."""
        context_text = "\n".join(context) if context else "No relevant memories found."
        
        system_message = f"""You are a helpful assistant with access to the user's memories.
Answer the question based on the memories provided below.
If the information is not available in the memories, say "Unknown" or "I don't know".

IMPORTANT: Give the shortest possible answer. Just provide the direct answer without any explanation or context.
For example, if asked "What is the user's dog's name?" just answer "Buddy" not "The user's dog's name is Buddy."

Memories:
{context_text}"""

        return self._call_llm(question, system_message)
    
    def _call_llm(self, user_message: str, system_message: str = None) -> str:
        """Call the configured LLM with system and user messages."""
        if self.llm_provider == "google":
            return call_google_llm(user_message, system_message)
        elif self.llm_provider == "openrouter":
            return call_openrouter(user_message, system_message)
        else:  # openai
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": user_message})
            
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=500,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
    
    def clear(self) -> None:
        """Clear all memories for this user."""
        try:
            print(f"[Mem0] Clearing all memories for user: {self.user_id}")
            self.client.delete_all(user_id=self.user_id)
            print("[Mem0] Memories cleared successfully")
            self.memory_count = 0
        except Exception as e:
            print(f"[Mem0] Error clearing memories: {e}")


class SupermemoryAdapter(MemoryAdapter):
    """Adapter for Supermemory memory system."""
    
    def __init__(self, api_key: Optional[str] = None, user_id: str = "benchmark_user"):
        super().__init__(user_id)
        
        if not SUPERMEMORY_AVAILABLE:
            raise ImportError("supermemory package not installed. Run: pip install supermemory")
        
        self.api_key = api_key or os.environ.get("SUPERMEMORY_API_KEY")
        if not self.api_key:
            raise ValueError("SUPERMEMORY_API_KEY not provided or set in environment")
        
        self.client = Supermemory(api_key=self.api_key)
        self.llm_provider = get_llm_provider()
    
    @property
    def container_tag(self) -> str:
        """Container tag derived from user_id - changes per benchmark file."""
        return f"benchmark_{self.user_id}"
    
    def ingest_sessions(self, sessions: List[Dict[str, Any]]) -> None:
        """Ingest sessions into Supermemory."""
        print(f"[Supermemory] Ingesting {len(sessions)} sessions with container_tag: {self.container_tag}")
        
        for session in sessions:
            session_text = self.format_session_for_ingestion(session)
            session_id = session.get("id", "unknown")
            
            try:
                result = self.client.memories.add(
                    content=session_text,
                    container_tag=self.container_tag,
                    metadata={
                        "session_id": session_id,
                        "user_id": self.user_id,
                    }
                )
                print(f"[Supermemory] Session {session_id}: Added")
                time.sleep(0.2)  # Rate limiting
            except Exception as e:
                print(f"[Supermemory] Error ingesting session {session_id}: {e}")
        
        # Wait for async processing
        print("[Supermemory] Waiting for processing to complete...")
        time.sleep(5)
    
    def search(self, query: str, top_k: int = 8) -> List[str]:
        """Search Supermemory for relevant memories using hybrid search."""
        try:
            response = self.client.search.memories(
                q=query,
                container_tag=self.container_tag,
                search_mode="hybrid"
            )
            
            context_parts = []
            
            # Debug: show response structure
            response_attrs = list(vars(response).keys()) if hasattr(response, '__dict__') else dir(response)
            print(f"[Supermemory] Response attrs: {[a for a in response_attrs if not a.startswith('_')]}")
            
            # Handle memories (extracted/summarized memories)
            if hasattr(response, 'memories') and response.memories:
                print(f"[Supermemory] Found {len(response.memories)} memories")
                for mem in response.memories:
                    mem_text = self._extract_text_from_item(mem, "memory")
                    if mem_text:
                        context_parts.append(f"[Memory] {mem_text}")
            
            # Handle chunks (raw document chunks)
            if hasattr(response, 'chunks') and response.chunks:
                print(f"[Supermemory] Found {len(response.chunks)} chunks")
                for chunk in response.chunks:
                    chunk_text = self._extract_text_from_item(chunk, "chunk")
                    if chunk_text:
                        context_parts.append(f"[Document] {chunk_text}")
            
            # Handle results (legacy format or combined results)
            if hasattr(response, 'results') and response.results:
                print(f"[Supermemory] Found {len(response.results)} results")
                for result in response.results:
                    # Each result might have both content and chunks
                    result_text = self._extract_text_from_item(result, "result")
                    if result_text:
                        context_parts.append(result_text)
                    
                    # Also check for nested chunks within results
                    if hasattr(result, 'chunks') and result.chunks:
                        for chunk in result.chunks:
                            chunk_text = self._extract_text_from_item(chunk, "nested_chunk")
                            if chunk_text and chunk_text not in context_parts:
                                context_parts.append(f"[Document] {chunk_text}")
            
            # Debug output
            if context_parts:
                print(f"[Supermemory] Extracted {len(context_parts)} context items")
            else:
                print(f"[Supermemory] Warning: No context extracted from response")
                # Dump response for debugging
                for attr in ['memories', 'chunks', 'results', 'total']:
                    if hasattr(response, attr):
                        val = getattr(response, attr)
                        print(f"[Supermemory]   {attr} = {repr(val)[:200] if val else 'None/Empty'}")
            
            return context_parts
            
        except Exception as e:
            print(f"[Supermemory] Search error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _extract_text_from_item(self, item, item_type: str = "item") -> Optional[str]:
        """Extract text content from a Supermemory response item (memory, chunk, or result)."""
        if item is None:
            return None
        
        # If it's already a string, return it
        if isinstance(item, str):
            return item.strip() if item.strip() else None
        
        # Try various attribute names in order of preference
        text_attrs = ['content', 'text', 'memory', 'summary', 'chunk']
        
        for attr in text_attrs:
            if hasattr(item, attr):
                val = getattr(item, attr)
                if val and isinstance(val, str) and val.strip():
                    return val.strip()
        
        # Try dict-like access
        if hasattr(item, 'get'):
            for attr in text_attrs:
                val = item.get(attr)
                if val and isinstance(val, str) and val.strip():
                    return val.strip()
        
        # Try __dict__ access
        if hasattr(item, '__dict__'):
            for attr in text_attrs:
                if attr in item.__dict__:
                    val = item.__dict__[attr]
                    if val and isinstance(val, str) and val.strip():
                        return val.strip()
        
        return None
    
    def generate_answer(self, question: str, context: List[str]) -> str:
        """Generate answer using LLM with Supermemory context in system prompt."""
        if not context:
            context_text = "No relevant memories found."
        else:
            # Context items are already labeled with [Memory] or [Document] prefixes
            context_text = "\n\n---\n\n".join(context)
        
        system_message = f"""You are a helpful assistant with access to the user's memories and documents.
Answer the question based on the memories and documents provided below.
The context includes both extracted memories (key facts) and document chunks (raw conversation excerpts).
Use all available information to provide an accurate answer.

If the information is not available in the context, say "Unknown" or "I don't know".

IMPORTANT: Give the shortest possible answer. Just provide the direct answer without any explanation or context.
For example, if asked "What is the user's dog's name?" just answer "Buddy" not "The user's dog's name is Buddy."

Memories:
{context_text}"""

        return self._call_llm(question, system_message)
    
    def _call_llm(self, user_message: str, system_message: str = None) -> str:
        """Call the configured LLM with system and user messages."""
        if self.llm_provider == "google":
            return call_google_llm(user_message, system_message)
        elif self.llm_provider == "openrouter":
            return call_openrouter(user_message, system_message)
        else:  # openai
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": user_message})
            
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=500,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
    
    def clear(self) -> None:
        """Clear all memories in this container."""
        # Supermemory may not have a direct delete all method
        # This is a placeholder - actual implementation depends on API
        try:
            # Try to delete the container if supported
            pass
        except Exception as e:
            print(f"[Supermemory] Note: Clear not fully implemented: {e}")


class NaiveRAGAdapter(MemoryAdapter):
    """
    Naive RAG adapter using vector similarity search.
    Uses ChromaDB if available, otherwise falls back to sentence-transformers with numpy.
    This provides a baseline comparison against dedicated memory systems.
    """
    
    def __init__(self, user_id: str = "benchmark_user", persist_dir: Optional[str] = None):
        super().__init__(user_id)
        
        self.documents: List[str] = []
        self.embeddings = None
        self.use_chromadb = False
        
        # Try ChromaDB first, fall back to sentence-transformers
        if CHROMADB_AVAILABLE:
            try:
                # Use persistent storage if provided, otherwise in-memory
                if persist_dir:
                    self.chroma_client = chromadb.PersistentClient(path=persist_dir)
                else:
                    self.chroma_client = chromadb.Client()
                
                # Create unique collection name based on user_id
                self.collection_name = f"benchmark_{self._hash_user_id(user_id)}"
                
                # Initialize embedding function (uses default all-MiniLM-L6-v2)
                self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
                
                # Get or create collection
                self.collection = self.chroma_client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_fn,
                )
                self.use_chromadb = True
                print("[NaiveRAG] Using ChromaDB backend")
            except Exception as e:
                print(f"[NaiveRAG] ChromaDB failed ({e}), falling back to sentence-transformers")
                self.use_chromadb = False
        
        if not self.use_chromadb:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "Neither chromadb nor sentence-transformers available. "
                    "Run: pip install sentence-transformers"
                )
            # Use sentence-transformers for embeddings
            print("[NaiveRAG] Using sentence-transformers backend")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.llm_provider = get_llm_provider()
    
    def _hash_user_id(self, user_id: str) -> str:
        """Create a short hash of user_id for collection name."""
        return hashlib.md5(user_id.encode()).hexdigest()[:8]
    
    def ingest_sessions(self, sessions: List[Dict[str, Any]]) -> None:
        """Ingest sessions as individual messages (one document per turn)."""
        documents = []
        
        for session in sessions:
            session_id = session.get("id", "unknown")
            timestamp = session.get("timestamp", 0)
            
            # Ingest each turn as a separate document
            for i, turn in enumerate(session.get("turns", [])):
                speaker = turn.get("speaker", "unknown")
                text = turn.get("text", "")
                # Include session context in each message
                turn_text = f"[Session {session_id}, Turn {i}] {speaker}: {text}"
                documents.append(turn_text)
        
        if self.use_chromadb:
            # Use ChromaDB
            ids = [f"{self.user_id}_doc_{i}" for i in range(len(documents))]
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                try:
                    self.collection.add(
                        documents=batch_docs,
                        ids=batch_ids,
                    )
                except Exception as e:
                    print(f"[NaiveRAG] Error ingesting batch: {e}")
        else:
            # Use sentence-transformers
            self.documents = documents
            print(f"[NaiveRAG] Encoding {len(documents)} documents...")
            self.embeddings = self.model.encode(documents, show_progress_bar=False)
            print(f"[NaiveRAG] Encoded {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 5) -> List[str]:
        """Search for relevant documents using vector similarity."""
        if self.use_chromadb:
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=top_k,
                )
                documents = results.get("documents", [[]])[0]
                return documents
            except Exception as e:
                print(f"[NaiveRAG] Search error: {e}")
                return []
        else:
            # Use numpy cosine similarity
            if self.embeddings is None or len(self.documents) == 0:
                return []
            
            query_embedding = self.model.encode([query])[0]
            
            # Compute cosine similarities
            similarities = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            return [self.documents[i] for i in top_indices]
    
    def generate_answer(self, question: str, context: List[str]) -> str:
        """Generate answer using LLM with retrieved context in system prompt."""
        context_text = "\n\n".join(context) if context else "No relevant documents found."
        
        system_message = f"""You are a helpful assistant with access to retrieved documents.
Answer the question based on the documents provided below.
If the information is not available in the documents, say "Unknown" or "I don't know".

IMPORTANT: Give the shortest possible answer. Just provide the direct answer without any explanation or context.
For example, if asked "What is the user's dog's name?" just answer "Buddy" not "The user's dog's name is Buddy."

Retrieved Documents:
{context_text}"""

        return self._call_llm(question, system_message)
    
    def _call_llm(self, user_message: str, system_message: str = None) -> str:
        """Call the configured LLM with system and user messages."""
        if self.llm_provider == "google":
            return call_google_llm(user_message, system_message)
        elif self.llm_provider == "openrouter":
            return call_openrouter(user_message, system_message)
        else:  # openai
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": user_message})
            
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=500,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
    
    def clear(self) -> None:
        """Clear the stored documents."""
        if self.use_chromadb:
            try:
                self.chroma_client.delete_collection(name=self.collection_name)
                # Recreate empty collection
                self.collection = self.chroma_client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_fn,
                )
            except Exception as e:
                print(f"[NaiveRAG] Error clearing collection: {e}")
        else:
            self.documents = []
            self.embeddings = None


class NoRAGAdapter(MemoryAdapter):
    """
    No RAG baseline - puts ALL sessions directly in the LLM's context window.
    This measures the LLM's ability to answer questions with full conversation history
    but without any retrieval/memory system.
    """
    
    def __init__(self, user_id: str = "benchmark_user"):
        super().__init__(user_id)
        self.all_sessions_text = ""  # Store all sessions as formatted text
        self.llm_provider = get_llm_provider()
    
    def ingest_sessions(self, sessions: List[Dict[str, Any]]) -> None:
        """Store all sessions as formatted text for context window."""
        formatted_sessions = []
        for session in sessions:
            formatted_sessions.append(self.format_session_for_ingestion(session))
        
        self.all_sessions_text = "\n\n---\n\n".join(formatted_sessions)
    
    def search(self, query: str, top_k: int = 5) -> List[str]:
        """Return all sessions as context (no retrieval, just full context)."""
        if self.all_sessions_text:
            return [self.all_sessions_text]
        return []
    
    def generate_answer(self, question: str, context: List[str]) -> str:
        """Generate answer using LLM with ALL sessions in system prompt."""
        # Use the stored sessions text as context
        context_text = self.all_sessions_text if self.all_sessions_text else "No conversation history available."
        
        system_message = f"""You are a helpful assistant with access to the user's full conversation history.
Answer the question based on the conversation history provided below.
If the information is not available in the conversations, say "Unknown" or "I don't know".

IMPORTANT: Give the shortest possible answer. Just provide the direct answer without any explanation or context.
For example, if asked "What is the user's dog's name?" just answer "Buddy" not "The user's dog's name is Buddy."

=== CONVERSATION HISTORY ===
{context_text}
=== END OF HISTORY ==="""

        return self._call_llm(question, system_message)
    
    def _call_llm(self, user_message: str, system_message: str = None) -> str:
        """Call the configured LLM with system and user messages."""
        if self.llm_provider == "google":
            return call_google_llm(user_message, system_message)
        elif self.llm_provider == "openrouter":
            return call_openrouter(user_message, system_message)
        else:  # openai
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": user_message})
            
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=500,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
    
    def clear(self) -> None:
        """No-op for NoRAG."""
        pass


class NebulaAdapter(MemoryAdapter):
    """Adapter for Nebula memory system."""
    
    def __init__(self, api_key: Optional[str] = None, user_id: str = "benchmark_user"):
        super().__init__(user_id)
        
        if not NEBULA_AVAILABLE:
            raise ImportError("nebula-client package not installed. Run: pip install nebula-client")
        
        self.api_key = api_key or os.environ.get("NEBULA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "NEBULA_API_KEY not provided. Set the environment variable (or put it in a local .env)."
            )
        
        self.client = Nebula(api_key=self.api_key)
        self.llm_provider = get_llm_provider()
        self.collection_id = None  # UUID of the collection used for the current file
        self.collection_mode = "auto"  # auto|per_file|shared (resolved in _ensure_collection)
        self.stored_memories = []
        self._collection_id_cache = None  # lazy-loaded from nebula_collection_ids.json
    
    def _benchmark_collection_name(self) -> str:
        """
        Use a small number of collections to avoid hammering collection creation/lookups.

        The evaluator sets `self.user_id = filepath.stem` per benchmark file, e.g.
        "benchmark_complex_009". We use that to pick a stable collection per test set.
        """
        uid = str(self.user_id or "")
        if uid.startswith("benchmark_simple_"):
            return "nebula_benchmark_simple"
        if uid.startswith("benchmark_complex_"):
            return "nebula_benchmark_complex"
        return "nebula_benchmark"

    def _ensure_collection(self) -> str:
        """Ensure collection exists and return its ID."""
        # Control via env var:
        # - NEBULA_COLLECTION_MODE=auto (default): prefer per-file collection id if present in cache; else shared
        # - NEBULA_COLLECTION_MODE=per_file: require per-file cache entry (key == file stem)
        # - NEBULA_COLLECTION_MODE=shared: use shared benchmark collection (nebula_benchmark_simple/complex)
        self.collection_mode = (
            os.environ.get("NEBULA_COLLECTION_MODE", self.collection_mode).strip().lower()
            or "auto"
        )

        # Prefer explicit env vars (avoids any collection API calls)
        env_any = os.environ.get("NEBULA_COLLECTION_ID")
        if env_any:
            self.collection_id = env_any.strip()
            self.collection_mode = "shared"
            return self.collection_id

        collection_name = self._benchmark_collection_name()
        env_specific = None
        if collection_name.endswith("_simple"):
            env_specific = os.environ.get("NEBULA_SIMPLE_COLLECTION_ID")
        elif collection_name.endswith("_complex"):
            env_specific = os.environ.get("NEBULA_COMPLEX_COLLECTION_ID")
        if env_specific:
            self.collection_id = env_specific.strip()
            return self.collection_id

        if self.collection_id:
            return self.collection_id

        # Prefer local id cache to avoid name-based API lookups (faster/more reliable)
        if self._collection_id_cache is None:
            self._collection_id_cache = {}
            cache_path = Path("nebula_collection_ids.json")
            if cache_path.exists():
                try:
                    data = json.loads(cache_path.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        self._collection_id_cache = {
                            str(k): str(v) for k, v in data.items() if k and v
                        }
                except Exception:
                    pass

        # If each file was ingested into its own collection, the cache will contain
        # an entry keyed by the file stem (the evaluator sets self.user_id = filepath.stem).
        cache = self._collection_id_cache or {}
        file_stem = str(self.user_id)
        per_file_id = cache.get(file_stem)

        if self.collection_mode in ("auto", "per_file"):
            if per_file_id:
                self.collection_id = per_file_id
                self.collection_mode = "per_file"
                return self.collection_id
            if self.collection_mode == "per_file":
                raise ValueError(
                    f"[Nebula] NEBULA_COLLECTION_MODE=per_file but no cached id for '{file_stem}'. "
                    "Create/update nebula_collection_ids.json or set NEBULA_COLLECTION_ID."
                )

        cached_id = self._collection_id_cache.get(collection_name) if self._collection_id_cache else None
        if cached_id:
            self.collection_id = cached_id
            self.collection_mode = "shared"
            return self.collection_id
            
        # Avoid name-based lookups; create first, then recover via list on conflict.
        try:
            result = self.client.create_collection(name=collection_name)
            if hasattr(result, 'id'):
                self.collection_id = result.id
            elif isinstance(result, dict):
                self.collection_id = result.get('id')
            else:
                self.collection_id = str(result)

            # Best-effort: persist into cache for future runs
            try:
                if self._collection_id_cache is not None:
                    self._collection_id_cache[collection_name] = str(self.collection_id)
                Path("nebula_collection_ids.json").write_text(
                    json.dumps(self._collection_id_cache or {}, indent=2, ensure_ascii=False, sort_keys=True),
                    encoding="utf-8",
                )
            except Exception:
                pass

            return self.collection_id
        except Exception as e:
            msg = str(e).lower()
            if any(token in msg for token in ("already exists", "conflict", "409")):
                # Recover by scanning collections list (no name endpoint)
                try:
                    limit = 200
                    offset = 0
                    while True:
                        cols = self.client.list_collections(limit=limit, offset=offset)
                        if not cols:
                            break
                        for c in cols:
                            if getattr(c, "name", None) == collection_name and getattr(c, "id", None):
                                self.collection_id = str(c.id)
                                if self._collection_id_cache is not None:
                                    self._collection_id_cache[collection_name] = self.collection_id
                                return self.collection_id
                        if len(cols) < limit:
                            break
                        offset += limit
                except Exception:
                    pass
            print(f"[Nebula] ERROR: Could not get or create collection '{collection_name}': {e}")
            raise ValueError(f"Collection '{collection_name}' not found. Run ingestion first.")
    
    def ingest_sessions(self, sessions: List[Dict[str, Any]]) -> None:
        """Ingest sessions into Nebula."""
        print(f"[Nebula] Ingesting {len(sessions)} sessions for user_id: {self.user_id}")
        
        # Ensure collection exists
        collection_id = self._ensure_collection()
        
        # Sort sessions by timestamp to maintain temporal ordering
        sorted_sessions = sorted(sessions, key=lambda s: s.get("timestamp", 0))
        
        total_turns = 0
        
        for session_idx, session in enumerate(sorted_sessions):
            session_id = session.get("id", f"session_{session_idx}")
            session_timestamp = session.get("timestamp", session_idx)
            turns = session.get("turns", [])
            total_turns += len(turns)
            
            # Build conversation text with role labels
            conversation_text = ""
            for turn in turns:
                speaker = turn.get("speaker", "user")
                text = turn.get("text", "")
                role_label = "User" if speaker == "user" else "Assistant"
                conversation_text += f"{role_label}: {text}\n"
            
            if not conversation_text.strip():
                continue
            
            try:
                # See ingest_nebula.py: this Nebula API expects `collection_id` (not `collection_ref`)
                payload = {
                    "collection_id": collection_id,
                    "engram_type": "document",
                    "raw_text": conversation_text.strip(),
                    "metadata": {
                        "session_id": str(session_id),
                        "timestamp": str(session_timestamp),
                        "source_file_stem": str(self.user_id),
                        "source_file": f"{self.user_id}.json",
                        "num_turns": str(len(turns)),
                    },
                    "ingestion_mode": "fast",
                }
                response = self.client._make_request("POST", "/v1/memories", json_data=payload)
                memory_id = None
                if isinstance(response, dict):
                    results = response.get("results")
                    if isinstance(results, dict):
                        memory_id = results.get("id") or results.get("engram_id")
                memory_info = f" (memory_id: {memory_id})" if memory_id else ""
                print(f"[Nebula] Session {session_id}: {len(turns)} turns - stored{memory_info}")
            except Exception as e:
                print(f"[Nebula] Error ingesting session {session_id}: {e}")
        
        print(f"[Nebula] Ingested {len(sessions)} sessions with {total_turns} total turns")
    
    def search(self, query: str, top_k: int = 10) -> List[str]:
        """Search Nebula for relevant memories (intended behavior).

        This uses Nebula's server-side retrieval pipeline via `self.client.search()`,
        returning a mix of facts (relationships), utterances (raw excerpts), and entity
        descriptions depending on what the API returns.
        """
        try:
            collection_id = self._ensure_collection()
            file_stem = str(self.user_id)

            # For per-file collections, no filter is needed.
            # For shared collections, filter to only the current benchmark file's ingested docs.
            filters = None
            if self.collection_mode != "per_file":
                filters = {"metadata.source_file_stem": {"$eq": file_stem}}

            results = self.client.search(
                query=query,
                collection_ids=[collection_id],
                limit=max(10, top_k),
                filters=filters,
            )

            # Extract a useful text list from MemoryRecall
            memories: List[str] = []

            # Facts (relationships)
            if hasattr(results, "facts") and results.facts:
                for fact in results.facts:
                    if isinstance(fact, dict):
                        subject = fact.get("subject", "")
                        predicate = fact.get("predicate", "")
                        obj = fact.get("object_value", "")
                        if subject and predicate and obj:
                            memories.append(f"{subject} {predicate} {obj}")
                        else:
                            txt = fact.get("fact") or fact.get("text") or fact.get("content") or ""
                            if txt:
                                memories.append(txt)
                    else:
                        memories.append(str(fact))

            # Utterances (raw excerpts)
            if hasattr(results, "utterances") and results.utterances:
                for utt in results.utterances:
                    if isinstance(utt, dict):
                        txt = utt.get("content") or utt.get("text") or utt.get("message") or ""
                    elif hasattr(utt, "content"):
                        txt = utt.content
                    elif hasattr(utt, "text"):
                        txt = utt.text
                    else:
                        txt = str(utt)
                    if txt and txt not in memories:
                        memories.append(txt)

            # Entities (descriptions)
            if hasattr(results, "entities") and results.entities:
                for ent in results.entities:
                    if isinstance(ent, dict):
                        profile = ent.get("profile", {})
                        if isinstance(profile, dict):
                            entity_info = profile.get("entity", {})
                            desc = (
                                entity_info.get("description", "")
                                if isinstance(entity_info, dict)
                                else ""
                            )
                            if desc and desc not in memories:
                                memories.append(desc)

            return memories[:top_k]

        except Exception as e:
            print(f"[Nebula] Search error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def generate_answer(self, question: str, context: List[str]) -> str:
        """Generate answer using LLM with Nebula context in system prompt."""
        context_text = "\n".join(context) if context else "No relevant memories found."
        
        system_message = f"""You are a helpful assistant with access to the user's memories.
Answer the question based on the memories provided below.
If the information is not available in the memories, say "Unknown" or "I don't know".

IMPORTANT: Give the shortest possible answer. Just provide the direct answer without any explanation or context.
For example, if asked "What is the user's dog's name?" just answer "Buddy" not "The user's dog's name is Buddy."

Memories:
{context_text}"""

        return self._call_llm(question, system_message)
    
    def _call_llm(self, user_message: str, system_message: str = None) -> str:
        """Call the configured LLM with system and user messages."""
        if self.llm_provider == "google":
            return call_google_llm(user_message, system_message)
        elif self.llm_provider == "openrouter":
            return call_openrouter(user_message, system_message)
        else:  # openai
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": user_message})
            
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=500,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
    
    def clear(self) -> None:
        """Clear all memories in this collection."""
        try:
            print(f"[Nebula] Clearing collection: {self.collection_id}")
            # Note: Implement based on Nebula's delete API if available
            pass
        except Exception as e:
            print(f"[Nebula] Error clearing memories: {e}")


def create_adapter(adapter_type: str, **kwargs) -> MemoryAdapter:
    """
    Factory function to create memory adapters.
    
    Args:
        adapter_type: One of 'mem0', 'supermemory', 'naive_rag', 'no_rag', 'nebula'
        **kwargs: Additional arguments passed to the adapter constructor
    
    Returns:
        A MemoryAdapter instance
    """
    adapters = {
        "mem0": Mem0Adapter,
        "supermemory": SupermemoryAdapter,
        "naive_rag": NaiveRAGAdapter,
        "no_rag": NoRAGAdapter,
        "nebula": NebulaAdapter,
    }
    
    if adapter_type not in adapters:
        raise ValueError(f"Unknown adapter type: {adapter_type}. Choose from: {list(adapters.keys())}")
    
    return adapters[adapter_type](**kwargs)

