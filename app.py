import os
import yaml
import gradio as gr
from dotenv import load_dotenv
from rag_pipelines.reranker_rag import RerankerRAG

load_dotenv()

with open("rag_pipelines/reranker_rag/config.yaml", "r") as f:
    config = yaml.safe_load(f)

rag = RerankerRAG(config)

GROQ_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

def get_collections():
    collections = rag.vector_db.list_collections()
    return collections if collections else ["default"]

def get_llm_settings():
    return (
        rag.llm_config.get("model_id", GROQ_MODELS[0]),
        rag.llm_config.get("temperature", 0),
        rag.llm_config.get("max_tokens", 500),
        rag.llm_config.get("top_p", 0.1)
    )

def load_initial_state():
    model, temp, max_tok, top_p = get_llm_settings()
    collections = get_collections()
    collections_with_new = collections + ["Create New Collection"]
    return (
        gr.update(value=model),
        gr.update(value=temp),
        gr.update(value=max_tok),
        gr.update(value=top_p),
        gr.update(choices=collections, value=collections[0] if collections else None),
        gr.update(choices=collections_with_new, value=collections[0] if collections else None)
    )

def refresh_collection_dropdown():
    collections = get_collections()
    return gr.update(choices=collections, value=collections[0] if collections else None)

def refresh_collection_dropdown_with_new():
    collections = get_collections()
    choices = collections + ["Create New Collection"]
    return gr.update(choices=choices, value=collections[0] if collections else None)

def toggle_new_collection_input(selected):
    return gr.update(visible=(selected == "Create New Collection"))

def apply_llm_settings(model_id, temperature, max_tokens, top_p):
    rag.set_llm_params(
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )
    return "LLM parameters updated"

def ask_question(query, collection):
    if not query.strip():
        yield "Please enter a question.", ""
        return
    
    if not collection:
        yield "Please select a collection.", ""
        return
    
    try:
        answer_stream, docs = rag.stream_answer(query, k=4, collection_name=collection)
        
        sources_md = ""
        if docs:
            sources_md = "### Sources\n\n"
            for i, doc in enumerate(docs, 1):
                content_preview = doc.page_content
                source_file = doc.metadata.get("file_path", doc.metadata.get("source", "Unknown"))
                sources_md += f"**[{i}]** `{os.path.basename(source_file)}`\n\n{content_preview}\n\n---\n\n"
        
        full_answer = ""
        for chunk in answer_stream:
            full_answer += chunk.content
            yield full_answer, sources_md
            
    except Exception as e:
        yield f"Error: {str(e)}", ""

def upload_documents(files, collection, new_collection_name):
    if not files:
        return "No files selected", gr.update()
    
    target_collection = new_collection_name.strip() if collection == "Create New Collection" else collection
    
    if not target_collection:
        return "Please enter a collection name", gr.update()
    
    file_paths = [f.name for f in files]
    
    try:
        rag.add_documents(file_paths, collection_name=target_collection)
        file_names = [os.path.basename(f) for f in file_paths]
        return f"Added {len(files)} file(s) to '{target_collection}':\n" + "\n".join(f"• {name}" for name in file_names), refresh_collection_dropdown_with_new()
    except Exception as e:
        return f"Error: {str(e)}", gr.update()

with gr.Blocks(title="Chat with Docs") as app:
    gr.Markdown("# Chat with Docs")
    
    with gr.Sidebar(position="left"):
        gr.Markdown("### LLM Settings")
        
        model_dropdown = gr.Dropdown(
            choices=GROQ_MODELS,
            label="Model"
        )
        temperature_slider = gr.Slider(
            minimum=0, maximum=1, step=0.1,
            label="Temperature"
        )
        max_tokens_slider = gr.Slider(
            minimum=100, maximum=4000, step=100,
            label="Max Tokens"
        )
        top_p_slider = gr.Slider(
            minimum=0, maximum=1, step=0.05,
            label="Top P"
        )
        apply_btn = gr.Button("Apply Settings", variant="primary")
        settings_status = gr.Markdown("")
        
        apply_btn.click(
            fn=apply_llm_settings,
            inputs=[model_dropdown, temperature_slider, max_tokens_slider, top_p_slider],
            outputs=settings_status
        )
    
    with gr.Tabs():
        with gr.TabItem("Q&A"):
            qa_collection = gr.Dropdown(
                label="Select Collection",
                interactive=True
            )
            refresh_qa_btn = gr.Button("Refresh", size="sm")
            
            query_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask something about your documents...",
                lines=2
            )
            submit_btn = gr.Button("Ask", variant="primary")
            
            answer_output = gr.Markdown(label="Answer")
            
            with gr.Accordion("Sources", open=False):
                sources_output = gr.Markdown()
            
            refresh_qa_btn.click(fn=refresh_collection_dropdown, outputs=qa_collection)
            submit_btn.click(
                fn=ask_question,
                inputs=[query_input, qa_collection],
                outputs=[answer_output, sources_output]
            )
            query_input.submit(
                fn=ask_question,
                inputs=[query_input, qa_collection],
                outputs=[answer_output, sources_output]
            )
        
        with gr.TabItem("Add Documents"):
            doc_collection = gr.Dropdown(
                label="Select Collection",
                interactive=True
            )
            refresh_doc_btn = gr.Button("Refresh", size="sm")
            
            new_collection_input = gr.Textbox(
                label="New Collection Name",
                placeholder="Enter collection name...",
                visible=False
            )
            
            file_upload = gr.File(
                label="Upload Documents",
                file_count="multiple",
                file_types=[".pdf", ".docx"]
            )
            upload_btn = gr.Button("Upload", variant="primary")
            upload_status = gr.Markdown("")
            
            doc_collection.change(
                fn=toggle_new_collection_input,
                inputs=doc_collection,
                outputs=new_collection_input
            )
            refresh_doc_btn.click(fn=refresh_collection_dropdown_with_new, outputs=doc_collection)
            upload_btn.click(
                fn=upload_documents,
                inputs=[file_upload, doc_collection, new_collection_input],
                outputs=[upload_status, doc_collection]
            )

    app.load(
        fn=load_initial_state,
        outputs=[
            model_dropdown,
            temperature_slider,
            max_tokens_slider,
            top_p_slider,
            qa_collection,
            doc_collection
        ]
    )

if __name__ == "__main__":
    app.launch(theme=gr.themes.Base())
