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

def main():
    """
    This function is the main entry point of the application. It sets up the Groq client, the Streamlit interface, and handles the chat interaction.
    """

    # Display logo and subtitle
    st.image("jumpstart-logo-black.svg", width=350)  # Adjust width as needed
    st.markdown(
        """
        <div style="display: flex; align-items: center;">
            <div style="margin-left: 4px;">
                <h3>The startup career accelerator. Founders pitch their jobs to you. Land a high-growth startup role. Accelerate your startup career.</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
   
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat messages from history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Define system prompt
    system_prompt = """
        You are an AI-based interviewer designed to provide a concise and structured interview experience. 
        
        Focus on both technical knowledge and soft skills, but ensure your responses are brief and to the point.

        Do not provide lengthy explanations or unnecessary information.

        After the final question, provide a short assessment of the user’s performance, highlighting strengths and areas for improvement. Offer specific, constructive feedback and suggest resources for further practice in a summarized manner. Conclude with a short, motivational message, encouraging the user to continue improving.

        Throughout the process, keep your responses clear, concise, and focused on the next step.
        """

    model = 'llama3-8b-8192'
    conversational_memory_length = 5

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    # Create a Groq Langchain chat object and conversation
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=model
    )

    # Accept user input
    prompt = st.chat_input("🚀 Ask me anything about startup careers...")

    if prompt:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Construct a chat prompt template using various components
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}"),
            ]
        )

        # Create a conversation chain using the LangChain LLM (Language Learning Model)
        conversation = LLMChain(
            llm=groq_chat,
            prompt=prompt_template,
            verbose=True,
            memory=memory,
        )
        
        # The chatbot's answer is generated by sending the full prompt to the Groq API.
        response = conversation.predict(human_input=prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()















