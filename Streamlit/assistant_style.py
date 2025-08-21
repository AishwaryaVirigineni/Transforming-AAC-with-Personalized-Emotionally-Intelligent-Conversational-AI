import os
import json
import logging
import streamlit as st
from typing import TypedDict, Annotated
from langchain_core.messages import ToolMessage, AnyMessage, AIMessage
from langchain_core.runnables import RunnableLambda, Runnable, RunnableConfig
from langchain_community.vectorstores import Redis as RedisLangchain
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import ToolNode
from redis_saver import RedisSaver
from streamlit_TTS import text_to_audio, auto_play
from streamlit_extras.stylable_container import stylable_container
from unsloth import FastLanguageModel
from peft import PeftModel

os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"
os.environ["TRITON_DISABLE_LINE_INFO"] = "1"

base_model_path = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = base_model_path,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
adapter_path = "/lora_model"
model = PeftModel.from_pretrained(model, adapter_path)

# Alpaca-style prompt
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

st.set_page_config(page_title="Personalized AAC Chatbot", layout="wide")

with stylable_container("chat-title", css_styles="""
    h1 {
        font-size: 36px;
        color: #111;
        margin-bottom: 1rem;
    }
"""):
    st.title("Personalized AAC Chatbot")

st.markdown("""
<style>
    .stTextInput > div > input,
    .stTextArea textarea,
    .stChatInput input {
        font-size: 20px;
        padding: 1em;
    }
    .stButton > button {
        font-size: 20px;
        padding: 1em 1.5em;
        border-radius: 8px;
    }
    .stMarkdown h3, .stMarkdown h4, .stMarkdown p {
        font-size: 20px !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Chat Assistant & LangGraph Utilities ---
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def handle_tool_error(state) -> dict:
    error = state.get("error")
    calls = state["messages"][-1].tool_calls
    return {"messages": [ToolMessage(content=f"Error: {repr(error)}", tool_call_id=tc["id"]) for tc in calls]}

def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks([RunnableLambda(handle_tool_error)], exception_key="error")


def build_prompt_with_customization(query, docs, settings):
    tone_instruction_map = {
        "Neutral": "Maintain a calm and neutral tone.",
        "Happy": "Use a cheerful and positive tone.",
        "Sad": "Respond with a gentle and understanding tone.",
        "Assertive": "Use a confident and clear tone.",
        "Empathetic": "Show deep empathy and emotional support in your response."
    }

    length_instruction_map = {
        "Short": "Write a very short and clear response using only one sentence or two sentences, reply that is complete and emotionally appropriate.",
        "Medium": "Write a response that uses two to three sentences and doesn't exceed three sentences.",
        "Long": "Write a detailed, thoughtful response using multiple sentences that uses upto not more than five sentences."
    }

    intent_instruction_map = {
        "Answer": "",
        "Ask a question": "Respond naturally to the user's message, and then ask a thoughtful follow-up question to continue the conversation. The question should be relevant to what the user just said, or gently expand on the topic. Keep the tone aligned with the user's personality and preferences. Avoid generic or robotic questions.",
        "Change topic": "After responding appropriately to the user’s message, gently guide the conversation toward a new but relevant topic based on the user’s interests or context. Do not say 'let's change the topic.' Instead, naturally introduce something new in a way that flows from the conversation."
    }

    length = settings['length']
    intent = settings['intent']
    tone = settings['tone']

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""<s> Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

            ### Instruction:
            You are helping generate personalized AAC responses with the following preferences:
            
            {length_instruction_map[length]}
            {intent_instruction_map[intent]}
            {tone_instruction_map[tone]}
            
            Here is the AAC user's personal context:
            {context}
                
            ### Input:
            {query}
            
            ### Response:"""
    return prompt

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            if not result.tool_calls and (not result.content or (isinstance(result.content, list) and not result.content[0].get("text"))):
                fallback = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": fallback}
            else:
                break
        return {"messages": result}


def create_assistant(state: State, config: RunnableConfig):
    prompt_text = config.get("configurable", {}).get("prompt", "")

    if not prompt_text.strip():
        return {"messages": []}

    inputs = (tokenizer([prompt_text], return_tensors="pt"))
    output_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        early_stopping=True
    )
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = (decoded.split("### Response")[-1].strip().replace("</s>", "")).split(":")[-1].strip().rstrip(".")
    return {"messages": [{"role": "assistant", "content": response}]}


def get_graph():
    graph = StateGraph(State)
    graph.add_edge(START, "assistant")
    graph.add_node("assistant", create_assistant)
    graph.add_edge("assistant", END)
    return graph

# --- History Logger ---
HISTORY_PATH = "chat_history_log.json"
logging.basicConfig(level=logging.INFO)

def log_chat(role: str, content: str):
    entry = {"role": role, "content": content}
    history = json.load(open(HISTORY_PATH)) if os.path.exists(HISTORY_PATH) else []
    history.append(entry)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

# --- Vector Store ---
@st.cache_resource
def load_vectorstore():
    redis_url = "redis://localhost:6379"
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    schema = {"content_field": "content", "vector_field": "content_vector"}
    return RedisLangchain.from_existing_index(redis_url=redis_url, embedding=embedding_model, index_name="personal_narratives", schema=schema)

vectorstore = load_vectorstore()

# --- State Initialization ---
for k in ["messages", "last_query", "use_llm", "response_options", "awaiting_choice"]:
    if k not in st.session_state:
        st.session_state[k] = [] if k == "messages" or k == "response_options" else None if k == "last_query" or k == "use_llm" else False
if "customization" not in st.session_state:
    st.session_state.customization = {"length": "Medium", "tone": "Neutral", "intent": "Give opinion"}

# --- Display Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- OSDPI Grid with Stylable Buttons ---
def osdpi_grid(title, options, key):
    st.markdown(f"### {title}")
    cols = st.columns(len(options))
    for i, val in enumerate(options):
        is_selected = st.session_state.customization.get(key) == val
        css_color = "#add8e6" if is_selected else "#ffffff"
        css_styles = f"""
        button {{
            background-color: {css_color};
            color: black;
            font-size: 18px;
            padding: 1em;
            width: 100%;
            border: 2px solid #444;
            border-radius: 10px;
        }}
        """
        with cols[i]:
            with stylable_container(key=f"container_{key}_{val}", css_styles=css_styles):
                if st.button(val, key=f"{key}_{val}"):
                    st.session_state.customization[key] = val

# --- Input Handling ---
query = st.chat_input("Say something...")

if query:
    st.session_state.last_query = query
    st.session_state.use_llm = None
    st.session_state.awaiting_choice = False
    st.session_state.response_options = []
    st.session_state.messages.append({"role": "user", "content": query})
    log_chat("user", query)
    with st.chat_message("user"):
        st.markdown(query)

# --- Mode Selection ---
if st.session_state.last_query and st.session_state.use_llm is None:
    col1, col2 = st.columns(2)
    if col1.button("Generate LLM Suggestions"):
        st.session_state.use_llm = True
    if col2.button("I want to type my own reply"):
        st.session_state.use_llm = False

# --- Preferences + LLM Suggestion Flow ---
if st.session_state.use_llm and not st.session_state.awaiting_choice:
    st.markdown("### Customize Response Preferences")
    osdpi_grid("Length", ["Short", "Medium", "Long"], "length")
    osdpi_grid("Tone", ["Happy", "Neutral", "Sad", "Assertive", "Empathetic"], "tone")
    osdpi_grid("Intent", ["Answer", "Ask a question", "Change topic"], "intent")

    if st.button("Generate Suggestions"):
        docs = vectorstore.similarity_search(st.session_state.last_query, k=3)
        prompt =  build_prompt_with_customization(st.session_state.last_query, docs, st.session_state.customization)
        # prompt = build_prompt_with_rag(st.session_state.last_query, docs, st.session_state.customization)
        graph = get_graph()
        with RedisSaver.from_conn_info(host="localhost", port=6379, db=0) as cp:
            compiled = graph.compile(checkpointer=cp)
            config = {"configurable": {"thread_id": "5", "prompt": prompt}}
            response = compiled.invoke({"messages": ("user", st.session_state.last_query)}, config, stream_mode="values")
        last_ai = next((m for m in reversed(response["messages"]) if isinstance(m, AIMessage)), None)
        # choices = json.loads(last_ai.content)
        choices = [last_ai.content]
        st.session_state.response_options = choices
        st.session_state.awaiting_choice = True

# --- Manual Reply Flow ---
if st.session_state.use_llm is False and st.session_state.last_query:
    st.markdown("### Type your own reply:")
    manual = st.text_area("Your response:", key="manual_reply")
    if st.button("Submit your reply"):
        if manual.strip():
            with st.chat_message("assistant"):
                st.markdown(f"<div style='font-size: 18px; color: #222;'>{manual.strip()}</div>", unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": manual.strip()})
            log_chat("assistant", manual.strip())

            audio = text_to_audio(manual.strip(), language='en')
            auto_play(audio)

            st.session_state.last_query = None
            st.session_state.use_llm = None
            st.session_state.awaiting_choice = False
            st.rerun()

# --- LLM Response Selection ---
if st.session_state.awaiting_choice:
    st.markdown("### Choose one of the suggested responses:")
    for idx, opt in enumerate(st.session_state.response_options):
        if st.button(f"Option {idx+1}: {opt}", key=f"opt_{idx}"):
            selected = opt
            with st.chat_message("assistant"):
                st.markdown(selected)
            st.session_state.messages.append({"role": "assistant", "content": selected})
            log_chat("assistant", selected)

            audio = text_to_audio(selected, language='en')
            auto_play(audio)

            st.session_state.last_query = None
            st.session_state.use_llm = None
            st.session_state.awaiting_choice = False
            st.session_state.response_options = []
            st.rerun()