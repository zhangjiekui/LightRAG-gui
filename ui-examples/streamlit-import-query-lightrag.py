import os
import sys
import requests
import asyncio
from contextlib import contextmanager
import logging

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Create working directory first
working_dir = "./dickens"
if not os.path.exists(working_dir):
    os.makedirs(working_dir)

# Configure logger after working directory exists
logger = logging.getLogger("lightrag")
logger.setLevel(logging.DEBUG)  # Set to DEBUG for more detailed logs

# Configure httpx logger to be less verbose
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)  # Only show WARNING and above

# Create handlers if they don't exist
if not logger.handlers:
    # File handler
    log_file = os.path.join(working_dir, "lightrag.log")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)

# Rest of the imports
import streamlit as st
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_embedding, gpt_4o_mini_complete, ollama_model_complete
from lightrag.utils import EmbeddingFunc

# Define helper functions first
def get_llm_config(model_name):
    """Get the LLM configuration based on model name."""
    common_kwargs = {
        "host": "http://localhost:11434",
        "options": {"num_ctx": 32768}
    }
    
    if model_name == "gpt-4o-mini":
        # Don't pass host parameter to OpenAI functions
        return gpt_4o_mini_complete, "gpt-4o-mini"
    elif model_name in ["gemma2:2b", "mistral", "llama2"]:
        # Only pass host parameter to Ollama functions
        return (
            lambda prompt: asyncio.run(ollama_complete(model_name, prompt, **common_kwargs)),
            model_name
        )
    else:
        raise ValueError(f"Unsupported LLM model: {model_name}")

def get_embedding_config(model_name):
    """Get the embedding configuration based on model name."""
    if model_name == "nomic-embed-text":
        # First get a sample embedding to determine the actual dimension
        import ollama
        client = ollama.Client(host="http://localhost:11434")
        sample_embedding = client.embeddings(
            model="nomic-embed-text",
            prompt="test"
        )
        actual_dim = len(sample_embedding["embedding"])
        
        return EmbeddingFunc(
            embedding_dim=actual_dim,
            max_token_size=8192,
            func=lambda texts: ollama_embedding(
                texts,
                embed_model="nomic-embed-text",
                host="http://localhost:11434"
            )
        )
    else:
        raise ValueError(f"Unsupported embedding model: {model_name}")

async def ollama_complete(model_name, prompt, **kwargs):
    """Helper function for Ollama completion."""
    import ollama
    client = ollama.AsyncClient(**kwargs)
    response = await client.generate(model=model_name, prompt=prompt)
    return response['choices'][0]['text']

def test_ollama_embedding():
    """Test if Ollama embedding service is running and accessible."""
    try:
        import ollama
        client = ollama.Client(host="http://localhost:11434")
        # Try a simple embedding request
        client.embeddings(model="nomic-embed-text", prompt="test")
        return True
    except Exception as e:
        st.error(f"""
        ⚠️ Cannot connect to Ollama embedding service. Please ensure:
        1. Ollama is installed and running (http://localhost:11434)
        2. The nomic-embed-text model is installed
        
        Run: `ollama pull nomic-embed-text`
        
        Error details: {str(e)}
        """)
        return False

def display_kg_stats():
    """Display knowledge graph statistics in a popover."""
    try:
        # Get graph stats - access through the correct attribute name
        graph = st.session_state.rag.chunk_entity_relation_graph._graph
        
        stats = {
            "Nodes": graph.number_of_nodes(),
            "Edges": graph.number_of_edges(),
            "Average Degree": round(sum(dict(graph.degree()).values()) / graph.number_of_nodes(), 2) if graph.number_of_nodes() > 0 else 0
        }
        
        # Log the stats
        logger.info(f"Knowledge Graph Stats - Nodes: {stats['Nodes']}, Edges: {stats['Edges']}, Avg Degree: {stats['Average Degree']}")
        
        # Create stats text for popover
        stats_text = f"""
        - Nodes: {stats['Nodes']}
        - Edges: {stats['Edges']}
        - Average Degree: {stats['Average Degree']}
        """
        
        # Display as a popover button
        st.button("🕸️", help=stats_text)
            
    except Exception as e:
        logger.error(f"Error getting graph stats: {str(e)}")
        # Add more detailed error information
        if st.session_state.rag is None:
            logger.error("RAG instance is None")
        else:
            logger.error(f"Available RAG attributes: {dir(st.session_state.rag)}")

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.messages = []
    st.session_state.settings = {
        "search_mode": "hybrid",
        "llm_model": "gpt-4o-mini",
        "embedding_model": "nomic-embed-text",
        "system_message": "You are a helpful AI assistant that answers questions based on the provided documents.",
        "temperature": 0.7
    }
    st.session_state.show_settings = False
    st.session_state.show_import = False
    st.session_state.rag = None

# Function to initialize/reinitialize RAG
def init_rag():
    # Test Ollama embedding service first
    if not test_ollama_embedding():
        st.stop()
        
    working_dir = "./dickens"
    
    # Create working directory if it doesn't exist
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    
    # Initialize RAG with current settings
    llm_func, llm_name = get_llm_config(st.session_state.settings["llm_model"])
    embedding_config = get_embedding_config(st.session_state.settings["embedding_model"])
    
    # Separate kwargs based on model type
    if st.session_state.settings["llm_model"] == "gpt-4o-mini":
        llm_kwargs = {
            "temperature": st.session_state.settings["temperature"],
            "system_prompt": st.session_state.settings["system_message"]
        }
    else:
        llm_kwargs = {
            "host": "http://localhost:11434", 
            "options": {
                "num_ctx": 32768,
                "temperature": st.session_state.settings["temperature"]
            },
            "system_message": st.session_state.settings["system_message"]
        }
    
    st.session_state.rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_func,
        llm_model_name=llm_name,
        llm_model_max_async=4,
        llm_model_max_token_size=32768,
        llm_model_kwargs=llm_kwargs,
        embedding_func=embedding_config
    )
    st.session_state.initialized = True

# Callback functions
def handle_settings_update():
    """Update settings by modifying session state only."""
    st.session_state.show_settings = False
    st.session_state.initialized = False  # Force reinitialization
    # Remove st.rerun()

def handle_chat_clear():
    """Clear chat history by updating session state only."""
    st.session_state.messages = []
    # Remove st.rerun() - Streamlit will automatically rerun when session state changes

def handle_import_toggle():
    st.session_state.show_import = not st.session_state.show_import

def handle_settings_toggle():
    st.session_state.show_settings = not st.session_state.show_settings

@contextmanager
def get_event_loop_context():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

def handle_import(content):
    if st.session_state.rag is not None:
        try:
            with st.spinner("Importing content..."):
                with get_event_loop_context() as loop:
                    success = loop.run_until_complete(st.session_state.rag.ainsert(content))
                    
                    if success:
                        st.success("Content imported successfully!")
                        # Update stats display
                        # display_kg_stats()
                        st.session_state.show_import = False
                        # Remove st.rerun()
                    else:
                        st.error("Failed to import content")
                        
        except Exception as e:
            logger.exception("An error occurred during import.")
            st.error(f"An error occurred: {e}")

# After initializing RAG, display initial stats
if not st.session_state.initialized:
    init_rag()

# UI Layout
st.title("LightRAG Kwaai Day Demo")

# Input and controls
col1, col2, col3, col4, col5 = st.columns([7, 1, 1, 1, 1])

with col1:
    prompt = st.chat_input("Ask a question about your documents...")

with col2:
    display_kg_stats()  # Knowledge Graph stats in a popover

with col3:
    st.button("➕", help="Import Documents", on_click=handle_import_toggle)

with col4:
    st.button("⚙️", help="Query Settings", on_click=handle_settings_toggle)

with col5:
    st.button("🗑️", help="Clear Chat", on_click=handle_chat_clear)

# Import dialog
if st.session_state.show_import:
    with st.expander("Import Documents", expanded=True):
        tags = st.text_input(
            "Tags (optional):",
            help="Add comma-separated tags to help organize your documents"
        )
        
        tab1, tab2, tab3, tab4 = st.tabs(["Paste", "File Upload", "Website", "Test"])
        
        with tab1:
            text_input = st.text_area(
                "Paste text or markdown content:",
                height=200,
                help="Paste your document content here"
            )
            
            if st.button("Import", key="import_text"):
                if text_input:
                    handle_import(text_input)
        
        with tab2:
            uploaded_file = st.file_uploader(
                "Choose a markdown file",
                type=['md', 'txt'],
                help="Upload a markdown (.md) or text (.txt) file"
            )
            
            if uploaded_file is not None:
                if st.button("Import File", key="import_file"):
                    try:
                        content = uploaded_file.read()
                        if isinstance(content, bytes):
                            content = content.decode('utf-8')
                        handle_import(content)
                    except Exception as e:
                        st.error(f"Error importing file: {str(e)}")
        
        with tab3:
            url = st.text_input(
                "Website URL:",
                help="Enter the URL of the webpage you want to import"
            )
            
            if st.button("Import", key="import_url"):
                if url:
                    try:
                        response = requests.get(url)
                        response.raise_for_status()
                        handle_import(response.text)
                    except Exception as e:
                        st.error(f"Error importing website: {str(e)}")
        
        with tab4:
            st.markdown("### Demo Documents")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Import A Christmas Carol"):
                    try:
                        with open("dickens/imports/book.txt", "r", encoding="utf-8") as f:
                            content = f.read()
                            handle_import(content)
                    except Exception as e:
                        st.error(f"Error importing Dickens test book: {str(e)}")
            
            with col2:
                if st.button("Import LightRAG Paper"):
                    try:
                        with open("dickens/imports/2410.05779v2-LightRAG.pdf", "rb") as f:
                            import PyPDF2
                            pdf_reader = PyPDF2.PdfReader(f)
                            content = []
                            for page in pdf_reader.pages:
                                text = page.extract_text()
                                if text.strip():  # Only add non-empty pages
                                    content.append(text)
                                
                            if not content:
                                st.error("No text could be extracted from the PDF")
                            else:
                                combined_content = "\n\n".join(content)
                                handle_import(combined_content)
                    except FileNotFoundError:
                        st.error("PDF file not found. Please ensure the file exists in dickens/imports/")
                    except Exception as e:
                        st.error(f"Error importing LightRAG whitepaper: {str(e)}")

# Add system message configuration
def get_system_message():
    return st.text_area(
        "System Message:",
        value=st.session_state.settings["system_message"],
        help="Customize the AI assistant's behavior"
    )

# Add temperature control
def get_temperature():
    return st.slider(
        "Temperature:",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.settings["temperature"],
        step=0.1,
        help="Controls randomness in responses. Lower values are more focused, higher values more creative."
    )

# Settings dialog
if st.session_state.show_settings:
    with st.expander("Query Settings", expanded=True):
        st.session_state.settings["search_mode"] = st.selectbox(
            "Search mode:",
            ["naive", "local", "global", "hybrid"],
            index=["naive", "local", "global", "hybrid"].index(st.session_state.settings["search_mode"])
        )
        
        st.session_state.settings["llm_model"] = st.selectbox(
            "LLM Model:",
            ["gpt-4o-mini", "gemma2:2b", "mistral", "llama2"],
            index=["gpt-4o-mini", "gemma2:2b", "mistral", "llama2"].index(st.session_state.settings["llm_model"])
        )
        
        st.session_state.settings["embedding_model"] = st.selectbox(
            "Embedding Model:",
            ["nomic-embed-text"],
            index=["nomic-embed-text"].index(st.session_state.settings["embedding_model"])
        )
        
        st.session_state.settings["system_message"] = get_system_message()
        st.session_state.settings["temperature"] = get_temperature()
        
        st.button("Apply Settings", on_click=handle_settings_update)

# Handle chat input
if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.status("Searching and generating response..."):
            query_param = QueryParam(mode=st.session_state.settings["search_mode"])
            try:
                with get_event_loop_context() as loop:
                    response = loop.run_until_complete(st.session_state.rag.aquery(prompt, param=query_param))
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "metadata": {
                        "search_mode": st.session_state.settings["search_mode"],
                        "llm_model": st.session_state.settings["llm_model"],
                        "embedding_model": st.session_state.settings["embedding_model"]
                    }
                })
                
                st.write(response)
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
                fallback_response = "I apologize, but I encountered an error while processing your request."
                
                # Add error response to messages
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": fallback_response,
                    "metadata": {
                        "search_mode": st.session_state.settings["search_mode"],
                        "llm_model": st.session_state.settings["llm_model"],
                        "embedding_model": st.session_state.settings["embedding_model"],
                        "error": str(e)
                    }
                })
                
                st.write(fallback_response)
