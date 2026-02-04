import streamlit as st
import os

# Intentar importar google-genai, si falla usar google-generativeai
try:
    from google import genai
    from google.genai import types
    USE_VERTEX = True
except ImportError:
    import google.generativeai as genai
    USE_VERTEX = False
    st.warning("⚠️ Usando google-generativeai en lugar de google-genai. Algunas funciones pueden diferir.")

import base64

# Configuración de la página
st.set_page_config(
    page_title="Chatbot Oferta Pro a Pro",
    page_icon="🤖",
    layout="wide"
)

# Función para inicializar el cliente de Gemini
@st.cache_resource
def init_gemini_client():
    """Inicializa el cliente de Gemini con configuración cached"""
    try:
        if USE_VERTEX:
            client = genai.Client(
                vertexai=True,
                project="cf-esproapro-bic-pro-ou",
                location="europe-central2",
            )
            return client
        else:
            # Para google-generativeai necesitas configurar la API key
            # genai.configure(api_key="tu_api_key_aqui")
            st.error("Para usar google-generativeai necesitas configurar tu API key")
            return None
    except Exception as e:
        st.error(f"Error al inicializar el cliente de Gemini: {str(e)}")
        return None

# Función para obtener la configuración del modelo
def get_model_config():
    """Retorna la configuración del modelo"""
    if USE_VERTEX:
        tools = [
            types.Tool(
                retrieval=types.Retrieval(
                    vertex_rag_store=types.VertexRagStore(
                        rag_resources=[
                            types.VertexRagStoreRagResource(
                                rag_corpus="projects/503088742824/locations/europe-west3/ragCorpora/6917529027641081856"
                            )
                        ],
                        similarity_top_k=20,
                    )
                )
            )
        ]

        config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            max_output_tokens=65535,
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="OFF"
                )
            ],
            tools=tools,
            system_instruction=[
                types.Part.from_text(
                    text="Enfócate el contexto que te aporto en tu corpus rag. siempre enfoca tus respuesta a este contexto.  "
                )
            ]
        )
    else:
        # Configuración básica para google-generativeai
        config = {
            'temperature': 1,
            'top_p': 0.95,
            'max_output_tokens': 65535,
        }
    
    return config

# Función para generar respuesta del chatbot
def generate_response(client, messages, config):
    """Genera respuesta usando el modelo Gemini"""
    try:
        # Convertir mensajes de Streamlit al formato de Gemini
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg["content"])]
                )
            )
        
        # Generar respuesta usando streaming
        response_text = ""
        response_placeholder = st.empty()
        
        for chunk in client.models.generate_content_stream(
            model="gemini-2.5-flash-lite",
            contents=contents,
            config=config,
        ):
            if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                continue
            
            response_text += chunk.text
            # Actualizar la respuesta en tiempo real
            response_placeholder.markdown(f"🤖 **Asistente:** {response_text}")
        
        return response_text
        
    except Exception as e:
        st.error(f"Error al generar respuesta: {str(e)}")
        return f"Lo siento, hubo un error al procesar tu solicitud: {str(e)}."

# Función principal de la aplicación
def main():
    st.title("🤖 Pro AI Pro")
    st.markdown("---")
    
    # Inicializar cliente
    client = init_gemini_client()
    if not client:
        st.stop()
    
    # Obtener configuración
    config = get_model_config()
    
    # Inicializar historial de chat en session_state
    if "messages" not in st.session_state:
        # Incluir mensajes iniciales del ejemplo original
        st.session_state.messages = [
            
        ]
    
    # Sidebar con configuraciones
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Botón para limpiar historial
        if st.button("🗑️ Limpiar Historial", type="secondary"):
            st.session_state.messages = []
            st.rerun()
        
        # Información del modelo
        st.markdown("### 📊 Información de este Bot")
        st.info("""
        Este bot utiliza el modelo LLM gemini-2.5-flash-lite y cuenta con el contexto de todas las fichas técnicas del área de oferta
        
        """)
        
        # Instrucciones del sistema
        st.markdown("### 📋 Como usar este chat?")
        st.text_area(
            "Instrucciones:",
            value="Preguntale como si fuera una persona, se claro con tu pregunta, si bien parece magia pero no lo es. \n\n Si no logras la respuesta correcta, itera y repregunta hasta que lo logres",
            disabled=True,
            height=240
        )
    
    # Mostrar historial de mensajes
    st.markdown("### 💬 Conversación")
    
    # Contenedor para el chat
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"👤 **Tú:** {message['content']}")
            else:
                st.markdown(f"🤖 **Asistente:** {message['content']}")
        
        st.markdown("---")
    
    # Input para nuevo mensaje
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Escribe tu mensaje:",
                placeholder="Ejemplo: armame una tabla con 5 productos que sean entrantes...",
                key="user_input"
            )
        
        with col2:
            submit_button = st.form_submit_button("📤 Enviar", type="primary")
    
    # Procesar nuevo mensaje
    if submit_button and user_input:
        # Agregar mensaje del usuario
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Mostrar mensaje del usuario inmediatamente
        st.markdown(f"👤 **Tú:** {user_input}")
        
        # Generar respuesta
        with st.spinner("🤖 Generando respuesta..."):
            response = generate_response(client, st.session_state.messages, config)
        
        # Agregar respuesta del asistente
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
        
        # Rerun para actualizar la interfaz
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🔧 Chatbot powered by Google Gemini 2.5 Flash Lite"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
