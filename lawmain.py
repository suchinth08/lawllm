import streamlit as st
from PIL import Image
from lawchain import get_lpphelper_chain,process_llm_response

#st.title( "Lakna Reddy & Associates 🤖")
col1, mid, col2 = st.columns(3)
image = Image.open('lawimage2.jpg')
with col1:
    st.image(image, width=150)
with col2:
    st.markdown("## Lakna Reddy & Associates")    

question = st.text_input("Question: ")
@st.cache_resource
def load_qa_chain():
    chain = get_lpphelper_chain()
    return chain

if question:
    chain = load_qa_chain()
    #response = chain.run(question)
    #llm_response = process_llm_response(response)
    with st.spinner('Generating response...'):
        response = chain.invoke(question)
        print(response)
        #answer = response['result']
        answer = process_llm_response(response)
        st.header("Answer")        
        st.write(answer.replace("<pad>",""))