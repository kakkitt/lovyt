import streamlit as st
from langserve import RemoteRunnable
from pprint import pprint

st.title('Welcome to Speckle Server')
input_text = st.text_input('Ask a Speckle related question here')

if input_text:
    with st.spinner("Processing..."):
        try:
            app = RemoteRunnable("http://localhost:8000/speckle_chat/")
            for output in app.stream({"input": input_text}):
                for key, value in output.items():
                    # Node
                    pprint(f"Node '{key}':")
                pprint("\n---\n")
            output = value['generation']  
            st.write(output)
        
        except Exception as e:
            st.error(f"Error: {e}")
