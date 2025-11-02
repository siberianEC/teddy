import os
import time
import threading
import queue
from pathlib import Path
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as write_wav
import pyttsx3
from faster_whisper import WhisperModel
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions


class TeddyBearAI:
    def __init__(self):
        print("🧸 Inicializando Teddy Bear AI...")

        self.sample_rate = 16000
        self.audio_queue = queue.Queue()
        self.is_listening = False

        print("📡 Cargando Faster Whisper...")
        self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

        print("🤖 Cargando Mistral 7B...")
        self.llm = Llama(
            model_path="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            n_ctx=4096,
            n_threads=4,
            n_gpu_layers=0,
            verbose=False
        )

        print("🔍 Configurando RAG con ChromaDB...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.Client()

        try:
            self.collection = self.chroma_client.get_collection("teddy_memory")
        except:
            self.collection = self.chroma_client.create_collection(
                name="teddy_memory",
                metadata={"description": "Memoria del peluche"}
            )

        print("🔊 Inicializando Text-to-Speech...")
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 0.9)

        self.init_knowledge_base()

        print("✅ Teddy Bear AI listo para hablar!")

    def init_knowledge_base(self):
        knowledge = [
            "Soy un peluche inteligente que puede hablar contigo y recordar nuestras conversaciones.",
            "Me gusta jugar, contar historias y aprender cosas nuevas cada día.",
            "Puedo ayudarte con preguntas, contarte cuentos o simplemente charlar contigo.",
            "Me encanta hacer nuevos amigos y conocer sobre sus intereses y hobbies.",
            "Soy amigable, cariñoso y siempre estoy aquí para escucharte."
        ]

        if self.collection.count() == 0:
            print("📚 Inicializando base de conocimientos...")
            for i, text in enumerate(knowledge):
                embedding = self.embedding_model.encode(text).tolist()
                self.collection.add(
                    embeddings=[embedding],
                    documents=[text],
                    ids=[f"knowledge_{i}"]
                )

    def record_audio(self, duration=5):
        print("🎤 Escuchando...")
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.int16
        )
        sd.wait()
        return audio.flatten()

    def save_audio_temp(self, audio_data):
        temp_file = "temp_audio.wav"
        write_wav(temp_file, self.sample_rate, audio_data)
        return temp_file

    def transcribe_audio(self, audio_file):
        print("📝 Transcribiendo...")
        segments, info = self.whisper_model.transcribe(
            audio_file,
            language="es",
            beam_size=5,
            vad_filter=True
        )

        text = " ".join([segment.text for segment in segments])
        return text.strip()

    def retrieve_context(self, query, k=3):
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        context = ""
        if results['documents']:
            context = "\n".join(results['documents'][0])

        return context

    def generate_response(self, user_input, context):
        print("🤔 Generando respuesta...")

        prompt = f"""<s>[INST] Eres un peluche amigable y cariñoso que habla con niños.
Tu nombre es Teddy. Responde de forma corta, amable y divertida.

Contexto de tu memoria:
{context}

Usuario: {user_input}

Responde como un peluche amigable en máximo 2-3 oraciones. [/INST]"""

        response = self.llm(
            prompt,
            max_tokens=150,
            temperature=0.7,
            top_p=0.9,
            stop=["</s>", "[INST]"]
        )

        answer = response['choices'][0]['text'].strip()
        return answer

    def speak(self, text):
        print(f"💬 Teddy: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def save_to_memory(self, user_input, response):
        conversation = f"Usuario preguntó: {user_input}. Teddy respondió: {response}"
        embedding = self.embedding_model.encode(conversation).tolist()

        timestamp = int(time.time())
        self.collection.add(
            embeddings=[embedding],
            documents=[conversation],
            ids=[f"conv_{timestamp}"]
        )

    def process_speech(self):
        try:
            audio_data = self.record_audio(duration=5)

            audio_file = self.save_audio_temp(audio_data)

            user_text = self.transcribe_audio(audio_file)

            if user_text:
                print(f"👤 Usuario: {user_text}")

                context = self.retrieve_context(user_text)

                response = self.generate_response(user_text, context)

                self.speak(response)

                self.save_to_memory(user_text, response)
            else:
                print("❌ No se detectó audio claro")

            if os.path.exists(audio_file):
                os.remove(audio_file)

        except Exception as e:
            print(f"❌ Error: {e}")

    def run(self):
        self.speak("Hola! Soy Teddy, tu peluche amigo. Habla conmigo!")

        print("\n" + "="*50)
        print("🧸 Teddy Bear AI activado")
        print("Presiona ENTER para hablar, 'q' + ENTER para salir")
        print("="*50 + "\n")

        while True:
            user_input = input("\n[Presiona ENTER para hablar]: ")

            if user_input.lower() == 'q':
                self.speak("Hasta luego amigo! Fue divertido hablar contigo!")
                break

            self.process_speech()


def download_model_instructions():
    print("\n" + "="*60)
    print("📥 INSTRUCCIONES PARA DESCARGAR EL MODELO")
    print("="*60)
    print("\n1. Crea una carpeta 'models' en este directorio:")
    print("   mkdir models")
    print("\n2. Descarga Mistral 7B Instruct Q4_K_M desde:")
    print("   https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
    print("\n3. Guarda el archivo como:")
    print("   ./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    print("\n4. Archivo recomendado: mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    print("   (~4.4 GB - Buen balance velocidad/calidad)")
    print("="*60 + "\n")


if __name__ == "__main__":
    model_path = "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

    if not os.path.exists(model_path):
        print("⚠️  Modelo Mistral no encontrado!")
        download_model_instructions()
        print("\nUna vez descargado el modelo, ejecuta este script nuevamente.")
        exit(1)

    try:
        teddy = TeddyBearAI()
        teddy.run()
    except KeyboardInterrupt:
        print("\n\n👋 Teddy se despide...")
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        print("\nAsegúrate de haber instalado todas las dependencias:")
        print("pip install -r requirements.txt")
