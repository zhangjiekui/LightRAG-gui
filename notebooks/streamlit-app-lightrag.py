import os
import sys
import requests
import asyncio
from contextlib import contextmanager
import logging
import xxhash

# Add the context manager right after imports
@contextmanager
def get_event_loop_context():
    """Context manager to handle asyncio event loop."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    yield loop

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import LightRAG packages
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete, openai_embedding
from lightrag.utils import EmbeddingFunc, logger, set_logger

# Configure logging
working_dir = "./dickens"
if not os.path.exists(working_dir):
    os.makedirs(working_dir)
    
set_logger(os.path.join(working_dir, "lightrag.log"))
logger.setLevel(logging.DEBUG)

# Rest of the imports
import streamlit as st

# Define helper functions first
def get_llm_config(model_name):
    """Get the LLM configuration based on model name."""
    if model_name == "gpt-4o-mini":
        return gpt_4o_mini_complete, "gpt-4o-mini"
    else:
        raise ValueError(f"Unsupported LLM model: {model_name}")

def get_embedding_config(model_name):
    """Get the embedding configuration based on model name."""
    if model_name == "gpt-4o-mini":
        return EmbeddingFunc(
            embedding_dim=1536,  # Ada 002 embedding dimension
            max_token_size=8192,
            func=lambda texts: openai_embedding(
                texts,
                model="gpt-4o-mini",  # Changed from text-embedding-ada-002
                api_key=st.session_state.settings["api_key"]
            )
        )
    else:
        raise ValueError(f"Unsupported embedding model: {model_name}")

def test_api_key():
    """Test if OpenAI API key is valid and prompt for input if invalid."""
    if not st.session_state.settings["api_key"]:
        st.error("""
        ⚠️ OpenAI API key is required.
        Please enter your API key in the form below.
        """)
        show_api_key_form()
        return False
        
    try:
        from openai import OpenAI
        client = OpenAI(api_key=st.session_state.settings["api_key"])
        
        # Try a simple embedding request
        response = client.embeddings.create(
            input="test",
            model="gpt-4o-mini"
        )
        return True
        
    except Exception as e:
        st.error(f"""
        ⚠️ API Error. Please ensure:
        1. You have entered a valid OpenAI API key
        2. Your API key has access to the gpt-4o-mini model
        
        Error details: {str(e)}
        """)
        
        show_api_key_form()
        return False

def show_api_key_form():
    """Display the API key input form."""
    with st.form("api_key_form"):
        new_api_key = st.text_input(
            "Enter your OpenAI API key:",
            type="password",
            help="Get your API key from https://platform.openai.com/account/api-keys"
        )
        
        submitted = st.form_submit_button("Save API Key")
        
        if submitted and new_api_key:
            st.session_state.settings["api_key"] = new_api_key
            st.session_state.initialized = False
            st.rerun()

# Initialize session state with API key
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.settings = {
        "search_mode": "hybrid",
        "llm_model": "gpt-4o-mini",
        "embedding_model": "gpt-4o-mini",
        "system_message": "You are a helpful AI assistant that answers questions based on the provided documents.",
        "temperature": 0.7,
        "api_key": "",
        "query_settings": {
            "max_chunks": 5,
            "chunk_similarity_threshold": 0.7,
            "entity_similarity_threshold": 0.7,
            "relationship_similarity_threshold": 0.7
        }
    }
    st.session_state.rag = None
    st.session_state.messages = []

# Function to initialize/reinitialize RAG
def init_rag():
    if not test_api_key():  # Test API key before initializing
        return False
        
    working_dir = "./dickens"
    
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    
    # Initialize RAG with current settings
    llm_func, llm_name = get_llm_config(st.session_state.settings["llm_model"])
    embedding_config = get_embedding_config(st.session_state.settings["embedding_model"])
        
    llm_kwargs = {
        "temperature": st.session_state.settings["temperature"],
        "system_prompt": st.session_state.settings["system_message"],
        "api_key": st.session_state.settings["api_key"],  # Pass API key to LLM
        **st.session_state.settings["query_settings"]  # Add query settings
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
    return True

# Callback functions
def handle_settings_update():
    """Update settings and force RAG reinitialization."""
    st.session_state.initialized = False  # Force reinitialization

def handle_chat_download():
    """Download chat history as markdown."""
    if not st.session_state.messages:
        st.error("No messages to download yet! Start a conversation first.", icon="ℹ️")
        return
        
    from time import strftime
    
    # Create markdown content
    md_lines = [
        "# LightRAG Chat Session\n",
        f"*Exported on {strftime('%Y-%m-%d %H:%M:%S')}*\n",
        "\n## Settings\n",
        f"- Search Mode: {st.session_state.settings['search_mode']}",
        f"- LLM Model: {st.session_state.settings['llm_model']}",
        f"- Embedding Model: {st.session_state.settings['embedding_model']}",
        f"- Temperature: {st.session_state.settings['temperature']}",
        f"- System Message: {st.session_state.settings['system_message']}\n",
        "\n## Conversation\n"
    ]
    
    # Add messages
    for msg in st.session_state.messages:
        # Add role header
        role = "User" if msg["role"] == "user" else "Assistant"
        md_lines.append(f"\n### {role} ({msg['metadata'].get('timestamp', 'N/A')})")
        
        # Add message content
        md_lines.append(f"\n{msg['content']}\n")
        
        # Add metadata if it exists and it's an assistant message
        if msg["role"] == "assistant" and "metadata" in msg:
            metadata = msg["metadata"]
            if "query_info" in metadata:
                md_lines.append(f"\n> {metadata['query_info']}")
            if "error" in metadata:
                md_lines.append(f"\n> ⚠️ Error: {metadata['error']}")
    
    # Convert to string
    md_content = "\n".join(md_lines)
    
    # Create download link
    st.download_button(
        label="Download Chat",
        data=md_content,
        file_name=f"chat_session_{strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        key="download_chat"  # Add unique key
    )

def handle_insert(content):
    """Handle document insertion."""
    if st.session_state.rag is not None:
        try:
            with st.spinner("Inserting content..."):
                with get_event_loop_context() as loop:
                    success = loop.run_until_complete(st.session_state.rag.ainsert(content))
                    
                    if success:
                        st.success("Content inserted successfully!")
                    else:
                        st.error("Failed to insert content")
                        
        except Exception as e:
            logger.exception("An error occurred during insertion.")
            st.error(f"An error occurred: {e}")

# UI Layout
st.markdown("### [LightRAG](https://github.com/HKUDS/LightRAG) [Kwaai](https://www.kwaai.ai/) Day Demo [🔗](https://lightrag.streamlit.app) #alpha")

# Create a container for chat history and AI output
chat_container = st.container()

# After initializing RAG, display initial stats
with chat_container: 
    if not st.session_state.initialized:
        init_rag()

# Create dialog functions using the decorator pattern
@st.dialog("Insert Records")
def show_insert_dialog():
    """Dialog for inserting records from various sources."""
    tags = st.text_input(
        "Tags (optional):",
        help="Add comma-separated tags to help organize your documents"
    )
    
    tab1, tab2, tab3, tab4 = st.tabs(["Paste", "Upload", "Website", "Test"])
    
    with tab1:
        text_input = st.text_area(
            "Paste text or markdown content:",
            height=200,
            help="Paste your document content here"
        )
        
        if st.button("Insert", key="insert"):
            if text_input:
                handle_insert(text_input)
    
    with tab2:
        uploaded_file = st.file_uploader(
            "Choose a markdown file",
            type=['md', 'txt'],
            help="Upload a markdown (.md) or text (.txt) file"
        )
        
        if uploaded_file is not None:
            if st.button("Insert File", key="insert_file"):
                try:
                    content = uploaded_file.read()
                    if isinstance(content, bytes):
                        content = content.decode('utf-8')
                    handle_insert(content)
                except Exception as e:
                    st.error(f"Error inserting file: {str(e)}")
    
    with tab3:
        url = st.text_input(
            "Website URL:",
            help="Enter the URL of the webpage you want to insert"
        )
        
        if st.button("Insert", key="insert_url"):
            if url:
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    handle_insert(response.text)
                except Exception as e:
                    st.error(f"Error inserting website content: {str(e)}")
    
    with tab4:
        st.markdown("### Test Documents")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Insert A Christmas Carol"):
                try:
                    with open("dickens/imports/book.txt", "r", encoding="utf-8") as f:
                        content = f.read()
                        handle_insert(content)
                except Exception as e:
                    st.error(f"Error inserting Dickens test book: {str(e)}")
        
        with col2:
            if st.button("Insert LightRAG Paper"):
                try:
                    with open("dickens/imports/2410.05779v2-LightRAG.pdf", "rb") as f:
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
                            handle_insert(combined_content)
                except FileNotFoundError:
                    st.error("PDF file not found. Please ensure the file exists in dickens/imports/")
                except Exception as e:
                    st.error(f"Error inserting LightRAG whitepaper: {str(e)}")

@st.dialog("Settings")
def show_settings_dialog():
    """Dialog for configuring LightRAG settings."""
    # Add API key input at the top
    api_key = st.text_input(
        "OpenAI API Key:",
        value=st.session_state.settings["api_key"],
        type="password",
        help="Enter your OpenAI API key"
    )
    if api_key != st.session_state.settings["api_key"]:
        st.session_state.settings["api_key"] = api_key
        st.session_state.initialized = False
    
    # Add model selection dropdowns
    st.session_state.settings["llm_model"] = st.selectbox(
        "LLM Model:",
        ["gpt-4o-mini"],  # Add more models as they become available
        index=0
    )
    
    st.session_state.settings["embedding_model"] = st.selectbox(
        "Embedding Model:",
        ["gpt-4o-mini"],  # Add more models as they become available
        index=0
    )
    
    st.session_state.settings["search_mode"] = st.selectbox(
        "Search mode:",
        ["naive", "local", "global", "hybrid"],
        index=["naive", "local", "global", "hybrid"].index(st.session_state.settings["search_mode"])
    )
    
    st.session_state.settings["temperature"] = st.slider(
        "Temperature:",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.settings["temperature"],
        step=0.1
    )
    
    st.session_state.settings["system_message"] = st.text_area(
        "System Message:",
        value=st.session_state.settings["system_message"]
    )
    
    if st.button("Apply Settings"):
        handle_settings_update()
        st.rerun()

# Display chat history in the container
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            col1, col2 = st.columns([20, 1])
            with col1:
                st.write(message["content"])
            with col2:
                if "metadata" in message:
                    metadata = message["metadata"]
                    info_text = f"""
                    🕒 {metadata.get('timestamp', 'N/A')}

                    **Settings:**
                    • Search: {metadata.get('search_mode', 'N/A')}
                    • LLM: {metadata.get('llm_model', 'N/A')}
                    • Embedder: {metadata.get('embedding_model', 'N/A')}
                    • Temperature: {metadata.get('temperature', 'N/A')}
                    """
                    st.button("ℹ️", key=f"info_{message.get('timestamp', id(message))}", help=info_text)

# Create a container for input and controls at the bottom
with st.container():
    # Add a visual separator
    st.markdown("---")
    
    # Input and controls in a row
    col1, col2, col3, col4, col5 = st.columns([8, 1, 1, 1, 1])

    with col1:
        prompt = st.chat_input("Ask a question about your records...")

    with col2:
        if st.button("📁", help="Insert Records"):
            show_insert_dialog()

    with col3:
        if st.button("⚙️", help="Settings"):
            show_settings_dialog()

    with col4:
        if st.button("🕸️", help="Knowledge Graph Stats"):
            show_kg_stats_dialog()

    with col5:
        if st.button("📥", help="Download Options"):
            show_download_dialog()

# Handle chat input
if prompt:
    # Add user message with timestamp
    from time import strftime
    timestamp = strftime("%Y-%m-%d %H:%M:%S")
    date_short = strftime("%Y%m%d")
    prompt_hash = xxhash.xxh64(prompt.encode()).hexdigest()[:8]
    
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt,
        "metadata": {
            "timestamp": timestamp
        }
    })
    
    # Generate response
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        
        with status_placeholder.status("Searching and generating response..."):
            query_param = QueryParam(mode=st.session_state.settings["search_mode"])
            try:
                with get_event_loop_context() as loop:
                    response = loop.run_until_complete(st.session_state.rag.aquery(prompt, param=query_param))
                
                # Add assistant message with timestamp
                timestamp = strftime("%Y-%m-%d %H:%M:%S")
                
                # Create query info string
                query_info = f"{st.session_state.settings['search_mode']}@{st.session_state.settings['llm_model']} #ds/{prompt_hash}/{date_short}"
                
                # Replace status with expander
                with status_placeholder.expander(query_info, expanded=False):
                    st.write("**Query Details:**")
                    st.write(f"- Search Mode: {st.session_state.settings['search_mode']}")
                    st.write(f"- LLM Model: {st.session_state.settings['llm_model']}")
                    st.write(f"- Embedding Model: {st.session_state.settings['embedding_model']}")
                    st.write(f"- Temperature: {st.session_state.settings['temperature']}")
                    st.write(f"- Timestamp: {timestamp}")
                    st.write(f"- Prompt Hash: {prompt_hash}")
                
                # Add response with metadata
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "metadata": {
                        "timestamp": timestamp,
                        "search_mode": st.session_state.settings["search_mode"],
                        "llm_model": st.session_state.settings["llm_model"],
                        "embedding_model": st.session_state.settings["embedding_model"],
                        "temperature": st.session_state.settings["temperature"],
                        "prompt_hash": prompt_hash,
                        "query_info": query_info
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
                        "timestamp": timestamp,
                        "search_mode": st.session_state.settings["search_mode"],
                        "llm_model": st.session_state.settings["llm_model"],
                        "embedding_model": st.session_state.settings["embedding_model"],
                        "error": str(e)
                    }
                })
                
                st.write(fallback_response)

@st.dialog("Knowledge Graph Stats")
def show_kg_stats_dialog():
    """Dialog showing detailed knowledge graph statistics."""
    try:
        if st.session_state.rag is None:
            st.error("Knowledge Graph not initialized yet")
            return
            
        # Get graph stats
        graph = st.session_state.rag.chunk_entity_relation_graph._graph
        
        if graph is None:
            st.error("Knowledge Graph is empty")
            return
            
        # Basic stats
        stats = {
            "Nodes": graph.number_of_nodes(),
            "Edges": graph.number_of_edges(),
            "Average Degree": round(sum(dict(graph.degree()).values()) / graph.number_of_nodes(), 2) if graph.number_of_nodes() > 0 else 0
        }
        
        # Display stats with more detail
        st.markdown("### Knowledge Graph Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Nodes", stats["Nodes"])
        with col2:
            st.metric("Total Edges", stats["Edges"])
        with col3:
            st.metric("Average Degree", stats["Average Degree"])
            
        # Add more detailed information
        if stats["Nodes"] > 0:
            st.markdown("### Node Degree Distribution")
            degrees = dict(graph.degree())
            degree_dist = {}
            for d in degrees.values():
                degree_dist[d] = degree_dist.get(d, 0) + 1
            
            # Create a bar chart of degree distribution
            import plotly.graph_objects as go
            fig = go.Figure(data=[
                go.Bar(x=list(degree_dist.keys()), y=list(degree_dist.values()))
            ])
            fig.update_layout(
                title="Node Degree Distribution",
                xaxis_title="Degree",
                yaxis_title="Count"
            )
            st.plotly_chart(fig)
            
    except Exception as e:
        logger.error(f"Error getting graph stats: {str(e)}")
        st.error(f"Error getting graph stats: {str(e)}")

@st.dialog("Download Options")
def show_download_dialog():
    """Dialog for downloading chat history and records."""
    st.markdown("### Download Options")
    
    tab1, tab2 = st.tabs(["Chat History", "Inserted Records"])
    
    with tab1:
        st.markdown("Download the current chat session as a markdown file.")
        handle_chat_download()
    
    with tab2:
        st.markdown("Download all inserted records as a JSON file.")
        if st.button("Download Records"):
            try:
                if st.session_state.rag is None:
                    st.error("No records available. Initialize RAG first.")
                    return
                    
                # Get records from RAG
                records = st.session_state.rag.get_all_records()
                
                if not records:
                    st.warning("No records found to download.")
                    return
                
                import json
                from time import strftime
                
                # Convert records to JSON
                records_json = json.dumps(records, indent=2)
                
                # Create download button
                st.download_button(
                    label="Download JSON",
                    data=records_json,
                    file_name=f"lightrag_records_{strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
            except Exception as e:
                logger.error(f"Error downloading records: {str(e)}")
                st.error(f"Error downloading records: {str(e)}")
