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

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="RAG ChatPDF",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS Injection for UI enhancement using the provided palette
# Palette: #EBB3F2 (Light Purple), #6650F2 (Primary Purple), #503FBF (Dark Purple), #79D9AC (Mint), #F27052 (Coral)
st.markdown(
    """
    <style>
    /* Typography and Headers */
    h1, h2, h3 {
        color: #6650F2 !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Styling with a soft gradient using #EBB3F2 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(235, 179, 242, 0.15) 0%, rgba(255,255,255,0) 100%);
        border-right: 1px solid rgba(235, 179, 242, 0.5);
    }
    
    /* Primary Buttons */
    .stButton > button {
        background-color: #6650F2 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        background-color: #503FBF !important;
        box-shadow: 0 4px 12px rgba(80, 63, 191, 0.2) !important;
        color: white !important;
    }
    
    /* File Uploader border using the Mint Green (#79D9AC) */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #79D9AC !important;
        background-color: rgba(121, 217, 172, 0.05) !important;
        border-radius: 12px !important;
        transition: background-color 0.3s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        background-color: rgba(121, 217, 172, 0.15) !important;
    }
    
    /* Text Inputs Focus state */
    .stTextInput > div > div > input:focus, 
    .stTextArea > div > div > textarea:focus {
        border-color: #6650F2 !important;
        box-shadow: 0 0 0 2px rgba(102, 80, 242, 0.3) !important;
    }
    
    /* Custom divider line */
    hr {
        border-top: 2px solid rgba(235, 179, 242, 0.4) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title and presentation
st.title('Generación Aumentada por Recuperación (RAG) 💬')
st.caption(f"Versión de Python: {platform.python_version()} | Motor: GPT-4o-mini")

# Load and display image gracefully
try:
    image = Image.open('Chat_pdf.png')
    # Using columns to center the image better in the new layout
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, use_column_width=True)
except Exception as e:
    pass # Silently fail on image missing for a cleaner UI

st.divider()

# Sidebar configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    st.markdown("Este Agente te ayudará a realizar análisis sobre el PDF cargado.")
    # Get API key from user
    ke = st.text_input('Ingresa tu Clave de OpenAI', type="password", help="Tu clave API no se guarda.")
    
    if not ke:
        # Utilizing Coral (#F27052) implicitly through Streamlit's warning state or custom markdown
        st.markdown(
            '<div style="padding:10px; border-radius:8px; background-color:rgba(242, 112, 82, 0.1); color:#F27052; font-weight:500;">'
            '⚠️ Por favor ingresa tu clave de API para continuar.</div>', 
            unsafe_allow_html=True
        )
    else:
        os.environ['OPENAI_API_KEY'] = ke
        st.success("Clave configurada correctamente.")

# PDF uploader
pdf = st.file_uploader("📂 Carga tu archivo PDF aquí", type="pdf")

# Process the PDF if uploaded
if pdf is not None and ke:
    try:
        # Added a spinner for better UX during processing
        with st.spinner("Procesando y analizando el documento..."):
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text
            
            # Validation: Check if text was actually extracted
            if not text.strip():
                st.markdown(
                    '<div style="padding:15px; border-left: 5px solid #F27052; background-color:rgba(242, 112, 82, 0.1); border-radius:4px;">'
                    '<strong>Error:</strong> No se pudo extraer texto del PDF. Es posible que sea un documento escaneado.'
                    '</div>', 
                    unsafe_allow_html=True
                )
            else:
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
                    st.info(f"✅ Documento procesado: **{len(chunks)} fragmentos** indexados.")
                    
                    # Create embeddings and knowledge base
                    embeddings = OpenAIEmbeddings()
                    knowledge_base = FAISS.from_texts(chunks, embeddings)
                    
        st.divider()
        
        # User question interface
        st.subheader("💡 ¿Qué te gustaría saber?")
        user_question = st.text_area("Formula tu pregunta basada en el texto", placeholder="Ej: ¿Cuáles son las conclusiones principales del autor?", height=100)
        
        # We use a button to trigger the API call instead of running on every keystroke
        if st.button("Consultar al Documento") and user_question:
            with st.spinner("Generando respuesta..."):
                docs = knowledge_base.similarity_search(user_question)
                
                # Utilizes ChatOpenAI for modern chat models
                llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=user_question)
                
                # Display the response in a styled container
                st.markdown("### 📝 Respuesta:")
                with st.container():
                    st.markdown(
                        f'<div style="padding:20px; border: 1px solid #79D9AC; border-radius: 10px; background-color:rgba(121, 217, 172, 0.05);">'
                        f'{response}'
                        f'</div>', 
                        unsafe_allow_html=True
                    )
                
    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        with st.expander("Ver detalles del error"):
            st.error(traceback.format_exc())
            
elif pdf is None:
    st.markdown("<br><p style='text-align: center; color: gray;'>Carga un archivo PDF en la zona superior para comenzar.</p>", unsafe_allow_html=True)
