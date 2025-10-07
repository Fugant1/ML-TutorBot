import gradio as gr
import requests
import time
import json

CHAT_URL = "http://localhost:8000/chat?input={input}"

def stream_chat(message: str, history: list):
    """
    Handles the chat logic, streaming the response to the UI.
    This function is a generator, yielding updates to the UI step-by-step.
    """
    history = history or []
    history.append([message, ""])

    yield {
        chatbot: history,
        user_input: gr.Textbox(value="", interactive=False, placeholder="Generating response..."),
        send_btn: gr.Button("...", interactive=False),
        stop_btn: gr.Button("Stop", variant="stop", visible=True),
    }

    try:
        response = requests.get(CHAT_URL.format(input=message), timeout=20)
        response.raise_for_status()  
        
        full_response = response.json().get("output", "Sorry, I couldn't get a response.")

        for i in range(0, len(full_response), 5):
            time.sleep(0.01) 
            chunk = full_response[i:i+5]
            history[-1][1] += chunk
            yield {chatbot: history}

    except requests.exceptions.RequestException as e:
        history[-1][1] = f"⚠️ **Network Error:** Could not connect to the backend. Please ensure it's running. \n\n*Details: {e}*"
        yield {chatbot: history}
        
    except json.JSONDecodeError:
        history[-1][1] = f"⚠️ **Backend Error:** Received an invalid response from the backend. \n\n*Response Text: {response.text}*"
        yield {chatbot: history}

    except Exception as e:
        history[-1][1] = f"⚠️ **An unexpected error occurred:** {e}"
        yield {chatbot: history}

    finally:
        yield {
            user_input: gr.Textbox(value="", interactive=True, placeholder="Type your ML/Data Science question here..."),
            send_btn: gr.Button("Send", variant="primary", interactive=True),
            stop_btn: gr.Button("Stop", variant="stop", visible=False, interactive=True),
        }

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="sky",
).set(
    body_background_fill_dark="#111111",
    block_background_fill_dark="#111111",
    button_primary_background_fill_dark="linear-gradient(90deg, #358BCA, #5E4DB2)",
)

with gr.Blocks(theme=theme, title="ML Tutor Bot 🤖") as demo:
    gr.Markdown(
        """
        <div style="text-align: center;">
            <h1>🤖 ML Tutor Bot</h1>
            <p>Your personal assistant for Machine Learning, Data Science, and Python!</p>
        </div>
        """
    )
    
    chatbot = gr.Chatbot(
        label="Chat History",
        bubble_full_width=False,
        avatar_images=(None, "https://www.gradio.app/images/gradio-logo.svg"), # User avatar, Bot avatar
        height=500,
    )
    
    with gr.Row():
        user_input = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Type your ML/Data Science question here...",
            container=False, 
        )
        send_btn = gr.Button("Send", variant="primary", scale=1, min_width=150)

    with gr.Row():
        clear_btn = gr.Button("🗑️ Clear History", variant="secondary")
        stop_btn = gr.Button("Stop", variant="stop", visible=False)

    gr.Examples(
        examples=[
            "Explain the difference between classification and regression.",
            "Write a Python function to calculate the factorial of a number.",
            "What are the main assumptions of linear regression?",
        ],
        inputs=user_input,
        label="Example Questions"
    )

    with gr.Accordion("💡 Tips for Asking Questions", open=False):
        gr.Markdown(
            """
            - **Be Specific:** Instead of "How does ML work?", try "Explain how a decision tree makes a prediction."
            - **Ask for Code:** You can ask for Python snippets, e.g., "Show me how to use pandas to read a CSV file."
            - **Request Analogies:** If a concept is hard, ask for a real-world analogy, e.g., "Explain overfitting using an analogy."
            """
        )
    
    chat_submission = user_input.submit(
        stream_chat, 
        [user_input, chatbot], 
        [chatbot, user_input, send_btn, stop_btn]
    )
    send_btn.click(
        stream_chat, 
        [user_input, chatbot], 
        [chatbot, user_input, send_btn, stop_btn]
    )

    stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[chat_submission])

    clear_btn.click(lambda: ([], ""), None, [chatbot, user_input])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)