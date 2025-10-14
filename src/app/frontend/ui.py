import gradio as gr
import requests
import time
import json
import os

BOT_AVATAR_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),  # pasta do ui.py
    "static",
    "ML-TutorBot.jpg"
)

CHAT_URL = "http://localhost:8000/chat?input={input}"

# --- Custom CSS for styling (versão final e mais abrangente) ---
custom_css = """
/* --- AUMENTA O TAMANHO DE QUALQUER AVATAR NO CHATBOT --- */

/* 1. Alvo: QUALQUER contêiner de avatar dentro do chatbot com nosso ID. */
#chatbot_com_avatar_grande .avatar-container {
    width: 65px !important;
    height: 65px !important;
}

/* 2. Alvo: A imagem dentro de QUALQUER contêiner de avatar. */
#chatbot_com_avatar_grande .avatar-container img {
    width: 100% !important;
    height: 100% !important;
    object-fit: cover !important;
    transform: scale(1.3) !important;
}

/* --- ESTILOS ANTERIORES (CONTINUAM IGUAIS) --- */

#send_btn, #clear_btn, #stop_btn {
    border-radius: 100px !important;
}
#user_input_textbox textarea {
    border-radius: 100px !important;
    padding-left: 15px !important;
    padding-right: 15px !important;
}
#action_button_row {
    justify-content: center;
    gap: 12px;
}
#clear_btn {
    flex-grow: 0 !important;
    max-width: 250px !important;
}
"""

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

# Pass the custom CSS to the Blocks layout
with gr.Blocks(theme=theme, title="ML Tutor Bot 🤖", css=custom_css) as demo:
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
        # --- ATENÇÃO: SUBSTITUA O CAMINHO DA IMAGEM ABAIXO ---
        # O primeiro elemento é o avatar do usuário (None para padrão), o segundo é o avatar do bot.
        # Pode ser uma URL (ex: "https://www.example.com/bot_avatar.png") ou um caminho local (ex: "./bot_avatar.png")
        avatar_images=(None, BOT_AVATAR_PATH), # Exemplo: Use uma URL ou um caminho local aqui
        height=500,
        elem_id="chatbot_com_avatar_grande"
    )
    
    with gr.Row():
        user_input = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Type your ML/Data Science question here...",
            container=False, 
            elem_id="user_input_textbox", # Adicionado elem_id para estilizar com CSS
        )
        # Add elem_id to apply specific CSS
        send_btn = gr.Button("Send", variant="primary", scale=1, min_width=150, elem_id="send_btn")

    # Add elem_id to the Row to center its content
    with gr.Row(elem_id="action_button_row"):
        # Add elem_id to apply specific CSS
        clear_btn = gr.Button("🗑️ Clear History", variant="secondary", elem_id="clear_btn")
        stop_btn = gr.Button("Stop", variant="stop", visible=False, elem_id="stop_btn")

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