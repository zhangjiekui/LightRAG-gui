# Add to state initialization
state.insert_text = ""  # For paste dialog
state.website_url = ""  # For website dialog
state.upload_content = None  # For file upload
state.download_format = "markdown"  # For download options
state.status_message = ""
state.query_info = ""
state.show_query_details = False

# Insert Dialog Handlers
def handle_paste_insert(state):
    """Handle text insertion from paste."""
    if not state.insert_text:
        state.status_message = "Please enter some text to insert"
        return
        
    try:
        with get_event_loop_context() as loop:
            loop.run_until_complete(state.rag.ainsert(state.insert_text))
        state.status_message = "Content inserted successfully!"
        state.insert_text = ""  # Clear the input
        state.show_insert = False  # Close dialog
    except Exception as e:
        logger.error(f"Error inserting content: {str(e)}")
        state.status_message = f"Error inserting content: {str(e)}"

def handle_file_upload(state, file):
    """Handle file upload insertion."""
    try:
        content = file.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
            
        with get_event_loop_context() as loop:
            loop.run_until_complete(state.rag.ainsert(content))
        state.status_message = "File inserted successfully!"
        state.show_insert = False
    except Exception as e:
        logger.error(f"Error inserting file: {str(e)}")
        state.status_message = f"Error inserting file: {str(e)}"

def handle_website_insert(state):
    """Handle website content insertion."""
    if not state.website_url:
        state.status_message = "Please enter a URL"
        return
        
    try:
        response = requests.get(state.website_url)
        response.raise_for_status()
        with get_event_loop_context() as loop:
            loop.run_until_complete(state.rag.ainsert(response.text))
        state.status_message = "Website content inserted successfully!"
        state.website_url = ""
        state.show_insert = False
    except Exception as e:
        logger.error(f"Error inserting website content: {str(e)}")
        state.status_message = f"Error inserting website content: {str(e)}"

def insert_test_book(state):
    """Insert test book content."""
    try:
        with open("dickens/imports/book.txt", "r", encoding="utf-8") as f:
            content = f.read()
            with get_event_loop_context() as loop:
                loop.run_until_complete(state.rag.ainsert(content))
        state.status_message = "Test book inserted successfully!"
        state.show_insert = False
    except Exception as e:
        logger.error(f"Error inserting test book: {str(e)}")
        state.status_message = f"Error inserting test book: {str(e)}"

def insert_test_paper(state):
    """Insert test paper content."""
    try:
        with open("dickens/imports/2410.05779v2-LightRAG.pdf", "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            content = []
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text.strip():
                    content.append(text)
                    
            if not content:
                state.status_message = "No text could be extracted from the PDF"
                return
                
            combined_content = "\n\n".join(content)
            with get_event_loop_context() as loop:
                loop.run_until_complete(state.rag.ainsert(combined_content))
        state.status_message = "Test paper inserted successfully!"
        state.show_insert = False
    except Exception as e:
        logger.error(f"Error inserting test paper: {str(e)}")
        state.status_message = f"Error inserting test paper: {str(e)}"

# Knowledge Graph Stats Handler
def show_kg_stats(state):
    """Show knowledge graph statistics."""
    try:
        if state.rag is None:
            state.status_message = "Knowledge Graph not initialized yet"
            return
            
        graph = state.rag.chunk_entity_relation_graph._graph
        if graph is None:
            state.status_message = "Knowledge Graph is empty"
            return
            
        # Calculate stats
        nodes = graph.number_of_nodes()
        edges = graph.number_of_edges()
        avg_degree = round(sum(dict(graph.degree()).values()) / nodes, 2) if nodes > 0 else 0
        
        # Create degree distribution for plotting
        degrees = dict(graph.degree())
        degree_dist = {}
        for d in degrees.values():
            degree_dist[d] = degree_dist.get(d, 0) + 1
            
        # Create plot
        fig = go.Figure(data=[
            go.Bar(x=list(degree_dist.keys()), y=list(degree_dist.values()))
        ])
        fig.update_layout(
            title="Node Degree Distribution",
            xaxis_title="Degree",
            yaxis_title="Count"
        )
        
        # Update state with stats and plot
        state.kg_stats = {
            "nodes": nodes,
            "edges": edges,
            "avg_degree": avg_degree,
            "plot": fig
        }
        state.show_kg_stats = True
        
    except Exception as e:
        logger.error(f"Error getting graph stats: {str(e)}")
        state.status_message = f"Error getting graph stats: {str(e)}"

# Download Handlers
def handle_chat_download(state):
    """Handle chat history download."""
    if not state.messages:
        state.status_message = "No messages to download yet! Start a conversation first."
        return
        
    try:
        # Create markdown content
        md_lines = [
            "# LightRAG Chat Session\n",
            f"*Exported on {strftime('%Y-%m-%d %H:%M:%S')}*\n",
            "\n## Settings\n",
            f"- Search Mode: {state.settings['search_mode']}",
            f"- LLM Model: {state.settings['llm_model']}",
            f"- Embedding Model: {state.settings['embedding_model']}",
            f"- Temperature: {state.settings['temperature']}",
            f"- System Message: {state.settings['system_message']}\n",
            "\n## Conversation\n"
        ]
        
        for msg in state.messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            md_lines.append(f"\n### {role} ({msg['metadata'].get('timestamp', 'N/A')})")
            md_lines.append(f"\n{msg['content']}\n")
            
            if msg["role"] == "assistant" and "metadata" in msg:
                metadata = msg["metadata"]
                if "query_info" in metadata:
                    md_lines.append(f"\n> {metadata['query_info']}")
                if "error" in metadata:
                    md_lines.append(f"\n> ⚠️ Error: {metadata['error']}")
        
        # Save to file
        filename = f"chat_session_{strftime('%Y%m%d_%H%M%S')}.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
            
        state.status_message = f"Chat history saved to {filename}"
        state.show_download = False
        
    except Exception as e:
        logger.error(f"Error downloading chat history: {str(e)}")
        state.status_message = f"Error downloading chat history: {str(e)}"

def handle_records_download(state):
    """Handle inserted records download."""
    try:
        if state.rag is None:
            state.status_message = "No records available. Initialize RAG first."
            return
            
        records = state.rag.get_all_records()
        if not records:
            state.status_message = "No records found to download."
            return
            
        # Save to file
        filename = f"lightrag_records_{strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
            
        state.status_message = f"Records saved to {filename}"
        state.show_download = False
        
    except Exception as e:
        logger.error(f"Error downloading records: {str(e)}")
        state.status_message = f"Error downloading records: {str(e)}"

# Add these functions for message handling
def handle_prompt(state):
    """Handle user prompt and generate response."""
    if not state.current_prompt:
        return
        
    timestamp = strftime("%Y-%m-%d %H:%M:%S")
    date_short = strftime("%Y%m%d")
    prompt_hash = xxhash.xxh64(state.current_prompt.encode()).hexdigest()[:8]
    
    # Add user message
    state.messages.append({
        "role": "user",
        "content": state.current_prompt,
        "metadata": {
            "timestamp": timestamp
        }
    })
    
    # Update UI to show processing
    state.status_message = "Searching and generating response..."
    
    try:
        # Generate response
        query_param = QueryParam(mode=state.settings["search_mode"])
        with get_event_loop_context() as loop:
            response = loop.run_until_complete(state.rag.aquery(state.current_prompt, param=query_param))
        
        # Create metadata
        timestamp = strftime("%Y-%m-%d %H:%M:%S")
        query_info = f"{state.settings['search_mode']}@{state.settings['llm_model']} #ds/{prompt_hash}/{date_short}"
        
        # Add assistant message
        state.messages.append({
            "role": "assistant",
            "content": response,
            "metadata": {
                "timestamp": timestamp,
                "search_mode": state.settings["search_mode"],
                "llm_model": state.settings["llm_model"],
                "embedding_model": state.settings["embedding_model"],
                "temperature": state.settings["temperature"],
                "prompt_hash": prompt_hash,
                "query_info": query_info
            }
        })
        
        # Update query info for display
        state.query_info = f"""
        **Query Details:**
        - Search Mode: {state.settings['search_mode']}
        - LLM Model: {state.settings['llm_model']}
        - Embedding Model: {state.settings['embedding_model']}
        - Temperature: {state.settings['temperature']}
        - Timestamp: {timestamp}
        - Prompt Hash: {prompt_hash}
        """
        
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        logger.error(error_msg)
        
        # Add error message
        state.messages.append({
            "role": "assistant",
            "content": "I apologize, but I encountered an error while processing your request.",
            "metadata": {
                "timestamp": timestamp,
                "search_mode": state.settings["search_mode"],
                "llm_model": state.settings["llm_model"],
                "embedding_model": state.settings["embedding_model"],
                "error": str(e)
            }
        })
    
    finally:
        # Clear prompt and status
        state.current_prompt = ""
        state.status_message = ""

def toggle_query_details(state):
    """Toggle the visibility of query details."""
    state.show_query_details = not state.show_query_details

# Update the layout to include dialogs
layout = """
<!-- Previous layout parts -->

<!-- Insert Dialog -->
<|dialog|open={show_insert}|
### Insert Records

<|tabs|
<|tab|label=Paste|
<|{insert_text}|text_area|label=Paste text or markdown content:|>
<|Insert|button|on_action=handle_paste_insert|>
|>

<|tab|label=Upload|
<|Upload file|file_selector|on_change=handle_file_upload|extensions=.txt,.md|>
|>

<|tab|label=Website|
<|{website_url}|input|label=Website URL:|>
<|Insert|button|on_action=handle_website_insert|>
|>

<|tab|label=Test Documents|
<|Insert A Christmas Carol|button|on_action=insert_test_book|>
<|Insert LightRAG Paper|button|on_action=insert_test_paper|>
|>
|>
|>

<!-- Knowledge Graph Stats Dialog -->
<|dialog|open={show_kg_stats}|
### Knowledge Graph Statistics

<|{kg_stats}|chart|type=plotly|>

**Basic Stats:**
- Nodes: <|{kg_stats["nodes"]}|>
- Edges: <|{kg_stats["edges"]}|>
- Average Degree: <|{kg_stats["avg_degree"]}|>
|>

<!-- Download Dialog -->
<|dialog|open={show_download}|
### Download Options

<|tabs|
<|tab|label=Chat History|
Download the current chat session as a markdown file.
<|Download Chat|button|on_action=handle_chat_download|>
|>

<|tab|label=Inserted Records|
Download all inserted records as a JSON file.
<|Download Records|button|on_action=handle_records_download|>
|>
|>
|>
"""

# Add CSS classes for message styling
def message_class(message):
    """Return CSS class based on message role."""
    return f"message-{message['role']}"
