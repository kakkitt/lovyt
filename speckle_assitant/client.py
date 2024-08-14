import streamlit as st
from langserve import RemoteRunnable

def main():
    st.title('Welcome to Speckle Server')
    
    input_text = st.text_input('Ask a Speckle-related question here')

    if input_text:
        with st.spinner("Processing..."):
            try:
                app = RemoteRunnable("http://localhost:8000/speckle_chat/")
                
                # Stream the output
                output_container = st.empty()
                for output in app.stream({"input": input_text}):
                    for key, value in output.items():
                        output_container.text(f"Node '{key}':")
                        output_container.json(value)
                        
                # Display final output
                final_output = value.get('generation', 'No final output generated.')
                st.subheader("Final Answer:")
                st.write(final_output)
        
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()