import gradio as gr
from src.utils import query_dbqa


def format_response(response):
    response = response.replace("(", "").replace(")", "")
    response = response.replace("\\n", "<br>")
    return response


def query_dbqa_formatted(query, history):
    response = query_dbqa(query)
    if not isinstance(response, str):
        response = str(response)
    formatted_response = format_response(response)
    history.append((query, formatted_response))
    return "", history


with gr.Blocks() as demo:
    gr.Markdown("# Q&A Bot with SVTECHGPT")
    with gr.Tab("Knowledge Bot"):
        chatbot = gr.Chatbot(label="SVTECH Assistant")
        msg = gr.Textbox(label="Input query")
        clear = gr.ClearButton([msg, chatbot])

        def process_query(query, history):
            return query_dbqa_formatted(query, history)

        msg.submit(process_query, inputs=[msg, chatbot], outputs=[msg, chatbot])
        clear.click(lambda: None, None, chatbot)

if __name__ == "__main__":
    demo.launch()
