import os
import time
import glob
import traceback
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from gtts import gTTS
import platform

# 1. Page Configuration
st.set_page_config(
    page_title="RAG ChatPDF",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. File Cleanup Utility (Prevents server storage from filling up with old MP3s)
try:
    os.mkdir("temp")
except FileExistsError:
    pass

def remove_temp_files(days=1):
    mp3_files = glob.glob("temp/*mp3")
    if mp3_files:
        now = time.time()
        for f in mp3_files:
            if os.stat(f).st_mtime < now - (days * 86400):
                try:
                    os.remove(f)
                except:
                    pass

remove_temp_files(1)

# 3. Custom CSS Injection
st.markdown(
    """
    <style>
    h1, h2, h3 { color: #6650F2 !important; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(235, 179, 242, 0.15) 0%, rgba(255,255,255,0) 100%);
        border-right: 1px solid rgba(235, 179, 242, 0.5);
    }
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
    }
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #79D9AC !important;
        background-color: rgba(121, 217, 172, 0.05) !important;
        border-radius: 12px !important;
    }
    hr {
        border-top: 2px solid rgba(235, 179, 242, 0.4) !important;
        margin-top: 2rem; margin-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.title('Generación Aumentada por Recuperación (RAG) 💬')
st.caption(f"Versión de Python: {platform.python_version()} | Motor: GPT-4o-mini")

# Image loading
try:
    image = Image.open('Chat_pdf.png')
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, use_column_width=True)
except Exception:
    pass 

st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    ke = st.text_input('Ingresa tu Clave de OpenAI', type="password", help="Tu clave API no se guarda.")
    
    if not ke:
        st.markdown(
            '<div style="padding:10px; border-radius:8px; background-color:rgba(242, 112, 82, 0.1); color:#F27052; font-weight:500;">'
            '⚠️ Por favor ingresa tu clave de API para continuar.</div>', 
            unsafe_allow_html=True
        )
    else:
        os.environ['OPENAI_API_KEY'] = ke
        st.success("Clave configurada correctamente.")
        
    st.markdown("---")
    st.markdown("### 🔊 Salida de Audio")
    enable_audio = st.checkbox("Habilitar lectura de respuestas (TTS)", value=False)
    audio_lang = st.selectbox("Idioma de lectura", options=["es", "en"], index=0, format_func=lambda x: "Español" if x == "es" else "English")

# PDF uploader
pdf = st.file_uploader("📂 Carga tu archivo PDF aquí", type="pdf")

# Process PDF
if pdf is not None and ke:
    try:
        with st.spinner("Procesando y analizando el documento..."):
            pdf_reader = PdfReader(pdf)
            text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
            
            if not text.strip():
                st.markdown('<div style="padding:15px; border-left: 5px solid #F27052; background-color:rgba(242, 112, 82, 0.1); border-radius:4px;"><strong>Error:</strong> No se pudo extraer texto del PDF.</div>', unsafe_allow_html=True)
            else:
                text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=20, length_function=len)
                chunks = text_splitter.split_text(text)
                
                if chunks:
                    st.info(f"✅ Documento procesado: **{len(chunks)} fragmentos** indexados.")
                    knowledge_base = FAISS.from_texts(chunks, OpenAIEmbeddings())
                    
        st.markdown("---")
        
        # QA Interface
        st.subheader("💡 ¿Qué te gustaría saber?")
        user_question = st.text_area("Formula tu pregunta basada en el texto", placeholder="Ej: ¿Cuáles son las conclusiones principales?", height=100)
        
        if st.button("Consultar al Documento") and user_question:
            with st.spinner("Generando respuesta..."):
                docs = knowledge_base.similarity_search(user_question)
                llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=user_question)
                
                st.markdown("### 📝 Respuesta:")
                st.markdown(
                    f'<div style="padding:20px; border: 1px solid #79D9AC; border-radius: 10px; background-color:rgba(121, 217, 172, 0.05);">{response}</div><br>', 
                    unsafe_allow_html=True
                )
                
                # Text-to-Speech Execution
                if enable_audio:
                    with st.spinner("Generando pista de audio..."):
                        audio_filename = f"temp/resp_{int(time.time())}.mp3"
                        tts = gTTS(text=response, lang=audio_lang)
                        tts.save(audio_filename)
                        
                        with open(audio_filename, "rb") as audio_file:
                            st.audio(audio_file.read(), format="audio/mp3")
                
    except Exception as e:
        st.error(f"Error al procesar: {str(e)}")
        with st.expander("Ver detalles del error"):
            st.error(traceback.format_exc())
            
elif pdf is None:
    st.markdown("<br><p style='text-align: center; color: gray;'>Carga un archivo PDF en la zona superior para comenzar.</p>", unsafe_allow_html=True)
