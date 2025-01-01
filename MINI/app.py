import streamlit as st
from parse_image import parse_image, extract_text_from_image, preprocess_image
from utils.llm_helper import refine_text_with_llm, query_llm_with_context
from PIL import Image

st.set_page_config(layout="wide")

# Custom CSS to improve the layout
st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title centered at the top
st.title("Enhanced Medical Report Parser and Chat Assistant")

# Create two columns for the layout
left_col, right_col = st.columns([1, 1])

# Initialize session state
if 'refined_text' not in st.session_state:
    st.session_state.refined_text = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'submitted' not in st.session_state:
    st.session_state.submitted = False

# Callback for form submission
def handle_submit():
    st.session_state.submitted = True

# Left column - Image Upload and Processing
with left_col:
    st.header("Upload Medical Report")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a medical report image",
        type=["png", "jpg", "jpeg"],
        help="Select a clear image of your medical report"
    )
    
    if uploaded_file:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        image.save("uploaded_image.png")
        st.image(image, caption="Uploaded Report", use_column_width=True)
        
        with st.spinner("Processing report..."):
            try:
                # Parse the image
                parsed_sections = parse_image("uploaded_image.png")
                
                if parsed_sections:
                    # Create an expander for parsed sections
                    with st.expander("View Parsed Sections", expanded=False):
                        for section, df in parsed_sections.items():
                            st.subheader(section)
                            st.dataframe(df, use_container_width=True)
                    
                    # Combine parsed text for context
                    st.session_state.refined_text = "\n".join(
                        f"{section}:\n{df.to_string(index=False)}"
                        for section, df in parsed_sections.items()
                    )
                    
                else:
                    st.warning("Structured parsing failed. Using OCR and LLM fallback...")
                    raw_text = extract_text_from_image(preprocess_image("uploaded_image.png"))
                    st.session_state.refined_text = refine_text_with_llm(raw_text)
                    
                    with st.expander("View Extracted Text", expanded=False):
                        st.text(st.session_state.refined_text)
                
                st.success("Report processed successfully!")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.session_state.refined_text = None

# Right column - Chat Interface
with right_col:
    st.header("Chat with Your Report")
    
    if st.session_state.refined_text:
        # Add a brief instruction
        st.markdown("""
            Ask questions about your medical report. For example:
            - What are the key findings?
            - Explain the test results
            - What are the recommended actions?
        """)
        
        # Display chat history
        for q, a in st.session_state.chat_history:
            st.info(f"You: {q}")
            st.success(f"Assistant: {a}")
        
        # Chat form
        with st.form(key="chat_form", clear_on_submit=True):
            user_query = st.text_input("Type your question here:")
            submit_button = st.form_submit_button("Send", on_click=handle_submit)
        
        # Process the query after form submission
        if st.session_state.submitted:
            if user_query:  # Only process if there's actually a query
                with st.spinner("Generating response..."):
                    response = query_llm_with_context(st.session_state.refined_text, user_query)
                    st.session_state.chat_history.append((user_query, response))
            st.session_state.submitted = False
            st.rerun()
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    else:
        st.info("Please upload a medical report to start chatting.")