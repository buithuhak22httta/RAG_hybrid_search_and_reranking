import gradio as gr
from src.utils import query_dbqa

with gr.Blocks() as demo:
    gr.Markdown("# Q&A Bot with SVTECHGPT")
    with gr.Tab("Knowledge Bot"):
        chatbot = gr.components.Chatbot(label="SVTECH Assistant")
        msg = gr.components.Textbox(label="Input query")
        clear = gr.ClearButton([msg])
        msg.submit(fn=query_dbqa, inputs=[msg], outputs=[msg, chatbot])

if __name__ == "__main__":
    demo.launch()
