import streamlit as st
from groq import Groq
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Fetch the API key from the .env file
groq_api_key = os.getenv('GROQ_API_KEY')

# Function to read system prompt from file
def load_system_prompt(file_path="system_prompt.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        return "You are a helpful AI assistant providing career guidance for startup job seekers."

def main():
    """
    Main function to set up the Streamlit chat interface.
    """

    left_co, cent_co, last_co = st.columns(3)

    # Place the image in the center column
    with cent_co:
        st.image("knvb-logo-vector-2022.svg", width=250)
    st.markdown(
        """
        <div style="text-align: center;">
            <h2>KNVB Parent Portal</h2>
            <h3 style="color: gray;">Helping parents navigate agents, clubs, and their children's journey to becoming professional footballers.</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Quick-start conversation prompts
    st.subheader("💡 Start with a Question:")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⚽ How do I find the right agent for my child?"):
            st.session_state.chat_history.append({"role": "user", "content": "How do I find the right agent for my child?"})
    with col2:
        if st.button("🏆 What clubs are best for young players?"):
            st.session_state.chat_history.append({"role": "user", "content": "What clubs are best for young players?"})
    with col3:
        if st.button("📅 How can I manage my child's training schedule?"):
            st.session_state.chat_history.append({"role": "user", "content": "How can I manage my child's training schedule?"})

    st.markdown("---")

    # Initialize chat history if not already in session_state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat messages from history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Load system prompt
    system_prompt = load_system_prompt()

    model = 'llama-3.3-70b-versatile'
    conversational_memory_length = 5

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    # Create a Groq Langchain chat object
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=model
    )

    # Accept user input from chat input box
    prompt = st.chat_input("🚀 Ask me anything about your child’s football development...")

    # If user typed a prompt or clicked a predefined button, process input
    if prompt:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

    # Construct chat prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}"),
        ]
    )

    # Create conversation chain
    conversation = LLMChain(
        llm=groq_chat,
        prompt=prompt_template,
        verbose=True,
        memory=memory,
    )
    
    # Generate response if there's any new input
    if st.session_state.chat_history:
        last_message = st.session_state.chat_history[-1]["content"]
        response = conversation.predict(human_input=last_message, )
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
