import os
import traceback
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# App title and presentation
st.title('Generación Aumentada por Recuperación (RAG) 💬')
st.write("Versión de Python:", platform.python_version())

# Load and display image
try:
    image = Image.open('Chat_pdf.png')
    st.image(image, width=350)
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# Sidebar information
with st.sidebar:
    st.subheader("Este Agente te ayudará a realizar análisis sobre el PDF cargado")

# Get API key from user
ke = st.text_input('Ingresa tu Clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

# PDF uploader
pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

# Process the PDF if uploaded
if pdf is not None and ke:
    try:
        # Extract text from PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
        
        # Validation: Check if text was actually extracted
        if not text.strip():
            st.error("⚠️ No se pudo extraer texto del PDF. Es posible que sea un documento escaneado o una imagen sin formato de texto.")
        else:
            st.info(f"Texto extraído: {len(text)} caracteres")
            
            # Split text into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=500,
                chunk_overlap=20,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            
            # Validation: Check if chunks were generated
            if not chunks:
                st.error("⚠️ El documento no generó fragmentos de texto válidos para procesar.")
            else:
                st.success(f"Documento dividido en {len(chunks)} fragmentos")
                
                # Create embeddings and knowledge base
                embeddings = OpenAIEmbeddings()
                knowledge_base = FAISS.from_texts(chunks, embeddings)
                
                # User question interface
                st.subheader("Escribe qué quieres saber sobre el documento")
                user_question = st.text_area(" ", placeholder="Escribe tu pregunta aquí...")
                
                # Process question when submitted
                if user_question:
                    docs = knowledge_base.similarity_search(user_question)
                    
                    # Utilizes ChatOpenAI for modern chat models
                    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
                    
                    # Load QA chain
                    chain = load_qa_chain(llm, chain_type="stuff")
                    
                    # Run the chain
                    response = chain.run(input_documents=docs, question=user_question)
                    
                    # Display the response
                    st.markdown("### Respuesta:")
                    st.markdown(response)
                
    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        st.error(traceback.format_exc())
elif pdf is not None and not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")
else:
    st.info("Por favor carga un archivo PDF para comenzar")
