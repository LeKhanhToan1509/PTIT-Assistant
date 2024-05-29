import sys
sys.path.append('D:/demo')
import streamlit as st
import dotenv, os
import time
from llms.gpt3_5 import GPT3_5
from vectorDB.read import PdfAction
from prompts.question import Prompt
dotenv.load_dotenv()
api_key = os.environ.get("OPENSSL_API_KEY")



if __name__ == "__main__":
    # DataBase
    pdf = PdfAction(directory_path="Data/vectorStore", path="Data")
    if("vectorStore" not in os.listdir("Data")):
        pdf.create_vector_database()
    db = pdf.load_vector_database()
    # LLM
    llm = GPT3_5(api_key = api_key)
    # View
    st.write("CHAT BOT")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        def response_generator():
            context = pdf.get_context(question=prompt)
            questionPrompt = Prompt(context = context, input=prompt)
            question = questionPrompt.create_prompt()
            response = llm.invoke(question)
            for word in response.content.split():
                yield word + " "
                time.sleep(0.03)
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator()) 
            st.session_state.messages.append({"role": "assistant", "content": response})


