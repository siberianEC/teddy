import os
import sys
import time
import threading
import queue
import logging
import subprocess
import platform
from pathlib import Path
import multiprocessing

if platform.system() == "Darwin":
    multiprocessing.set_start_method('spawn', force=True)

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as write_wav
from faster_whisper import WhisperModel
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import atexit
import signal

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class TeddyBearAI:
    def __init__(self):
        print("🧸 Inicializando Teddy Bear AI...")
        logging.info("Starting TeddyBearAI initialization")

        self.sample_rate = 16000
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.tts_engine = None
        self.use_system_tts = False
        self.is_macos = platform.system() == "Darwin"
        self.chroma_client = None
        self.collection = None

        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        print("📡 Cargando Faster Whisper...")
        logging.info("Loading Faster Whisper model")
        try:
            self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
            logging.info("Whisper model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load Whisper model: {e}")
            raise

        print("🤖 Cargando Mistral 7B...")
        logging.info("Loading Mistral 7B model")
        try:
            self.llm = Llama(
                model_path="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                n_ctx=4096,
                n_threads=4,
                n_gpu_layers=0,
                verbose=False
            )
            logging.info("Mistral 7B model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load Mistral 7B model: {e}")
            raise

        print("🔍 Configurando RAG con ChromaDB...")
        logging.info("Setting up ChromaDB")
        try:
            logging.info("Loading sentence transformer model (this may take a moment)...")
            import torch
            torch.set_num_threads(1)

            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_model.to('cpu')

            logging.info("Sentence transformer loaded successfully")

            settings = chromadb.Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=False
            )
            self.chroma_client = chromadb.Client(settings)

            try:
                self.collection = self.chroma_client.get_collection("teddy_memory")
            except:
                self.collection = self.chroma_client.create_collection(
                    name="teddy_memory",
                    metadata={"description": "Memoria del peluche"}
                )
            logging.info("ChromaDB configured successfully")
        except Exception as e:
            logging.error(f"Failed to setup ChromaDB: {e}")
            raise

        print("🔊 Inicializando Text-to-Speech...")
        logging.info("Initializing Text-to-Speech")
        self._init_tts()

        self.init_knowledge_base()

        print("✅ Teddy Bear AI listo para hablar!")
        logging.info("TeddyBearAI initialization completed")

    def _init_tts(self):
        """Initialize TTS with fallback mechanisms for macOS compatibility"""
        try:
            if self.is_macos:
                logging.info("Detected macOS, using system 'say' command to avoid pyttsx3 crashes")
                self.tts_engine = None
                self.use_system_tts = True

                try:
                    result = subprocess.run(['say', '-v', '?'], check=True, capture_output=True, timeout=5)
                    if result.returncode == 0:
                        logging.info("macOS 'say' command configured successfully")
                    else:
                        raise RuntimeError("'say' command not available")
                except Exception as e:
                    logging.error(f"macOS 'say' command check failed: {e}")
                    logging.warning("Continuing without TTS verification")
            else:
                import pyttsx3
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 0.9)
                logging.info("pyttsx3 initialized successfully")
                self.use_system_tts = False

        except Exception as e:
            logging.error(f"TTS initialization failed: {e}")
            if self.is_macos:
                logging.info("Attempting to use macOS 'say' command as last resort")
                self.use_system_tts = True
                self.tts_engine = None
            else:
                raise RuntimeError(f"Could not initialize TTS engine: {e}")

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logging.info(f"Received signal {signum}, shutting down gracefully...")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        """Clean up resources properly to avoid semaphore leaks"""
        logging.info("Starting cleanup...")
        try:
            if self.tts_engine:
                try:
                    self.tts_engine.stop()
                    del self.tts_engine
                    self.tts_engine = None
                    logging.info("TTS engine stopped")
                except Exception as e:
                    logging.warning(f"Error stopping TTS: {e}")

            if self.chroma_client:
                try:
                    if self.collection:
                        del self.collection
                        self.collection = None

                    self.chroma_client.clear_system_cache()
                    del self.chroma_client
                    self.chroma_client = None
                    logging.info("ChromaDB cleaned up")
                except Exception as e:
                    logging.warning(f"Error cleaning ChromaDB: {e}")

            import gc
            gc.collect()
            logging.info("Garbage collection completed")

        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

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
            logging.info("Encoding knowledge base entries...")
            for i, text in enumerate(knowledge):
                logging.debug(f"Encoding knowledge entry {i+1}/{len(knowledge)}")
                try:
                    embedding = self.embedding_model.encode(
                        text,
                        show_progress_bar=False,
                        convert_to_numpy=True
                    ).tolist()
                    self.collection.add(
                        embeddings=[embedding],
                        documents=[text],
                        ids=[f"knowledge_{i}"]
                    )
                except Exception as e:
                    logging.error(f"Failed to encode knowledge entry {i}: {e}")
                    raise
            logging.info("Knowledge base initialized successfully")

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
        try:
            query_embedding = self.embedding_model.encode(
                query,
                show_progress_bar=False,
                convert_to_numpy=True
            ).tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )

            context = ""
            if results['documents']:
                context = "\n".join(results['documents'][0])

            return context
        except Exception as e:
            logging.error(f"Error retrieving context: {e}")
            return ""

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
        """Speak text using TTS with fallback mechanisms"""
        print(f"💬 Teddy: {text}")
        logging.info(f"Speaking: {text}")

        try:
            if self.use_system_tts and self.is_macos:
                subprocess.run(['say', text], check=True, capture_output=True, timeout=30)
                logging.debug("Spoke using macOS 'say' command")
            elif self.tts_engine:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                logging.debug("Spoke using pyttsx3")
            else:
                logging.warning("No TTS engine available, text only output")
        except subprocess.TimeoutExpired:
            logging.error("TTS 'say' command timed out")
            print("⚠️  TTS timed out")
        except Exception as e:
            logging.error(f"Error during speech: {e}")
            print(f"⚠️  TTS Error: {e}")
            if self.is_macos:
                try:
                    subprocess.run(['say', text], check=False, capture_output=True, timeout=10)
                except:
                    logging.error("Fallback TTS also failed")

    def save_to_memory(self, user_input, response):
        try:
            conversation = f"Usuario preguntó: {user_input}. Teddy respondió: {response}"
            embedding = self.embedding_model.encode(
                conversation,
                show_progress_bar=False,
                convert_to_numpy=True
            ).tolist()

            timestamp = int(time.time())
            self.collection.add(
                embeddings=[embedding],
                documents=[conversation],
                ids=[f"conv_{timestamp}"]
            )
            logging.debug("Conversation saved to memory")
        except Exception as e:
            logging.error(f"Error saving to memory: {e}")

    def process_speech(self):
        audio_file = None
        try:
            logging.info("Starting speech processing")
            audio_data = self.record_audio(duration=5)

            audio_file = self.save_audio_temp(audio_data)

            user_text = self.transcribe_audio(audio_file)

            if user_text:
                print(f"👤 Usuario: {user_text}")
                logging.info(f"Transcribed: {user_text}")

                context = self.retrieve_context(user_text)

                response = self.generate_response(user_text, context)

                self.speak(response)

                self.save_to_memory(user_text, response)
            else:
                print("❌ No se detectó audio claro")
                logging.warning("No clear audio detected")

        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"❌ Error: {e}")
            logging.error(f"Error in process_speech: {e}", exc_info=True)
        finally:
            if audio_file and os.path.exists(audio_file):
                try:
                    os.remove(audio_file)
                    logging.debug("Cleaned up temporary audio file")
                except Exception as e:
                    logging.warning(f"Could not remove temp file: {e}")

    def run(self):
        try:
            self.speak("Hola! Soy Teddy, tu peluche amigo. Habla conmigo!")

            print("\n" + "="*50)
            print("🧸 Teddy Bear AI activado")
            print("Presiona ENTER para hablar, 'q' + ENTER para salir")
            print("="*50 + "\n")

            while True:
                try:
                    user_input = input("\n[Presiona ENTER para hablar]: ")

                    if user_input.lower() == 'q':
                        self.speak("Hasta luego amigo! Fue divertido hablar contigo!")
                        break

                    self.process_speech()
                except KeyboardInterrupt:
                    print("\n\n👋 Interrumpido por usuario...")
                    break
        finally:
            self.cleanup()


def check_system_requirements():
    """Check if all system requirements are met"""
    print("\n" + "="*60)
    print("🔍 VERIFICANDO REQUISITOS DEL SISTEMA")
    print("="*60 + "\n")

    all_ok = True

    print("✓ Python version:", sys.version)

    try:
        import sounddevice as sd
        print("✓ sounddevice installed")
        devices = sd.query_devices()
        print(f"✓ Found {len(devices)} audio devices")
    except Exception as e:
        print(f"✗ sounddevice error: {e}")
        all_ok = False

    try:
        import numpy
        print("✓ numpy installed")
    except:
        print("✗ numpy not installed")
        all_ok = False

    try:
        from faster_whisper import WhisperModel
        print("✓ faster-whisper installed")
    except:
        print("✗ faster-whisper not installed")
        all_ok = False

    try:
        from llama_cpp import Llama
        print("✓ llama-cpp-python installed")
    except:
        print("✗ llama-cpp-python not installed")
        all_ok = False

    try:
        from sentence_transformers import SentenceTransformer
        print("✓ sentence-transformers installed")
    except:
        print("✗ sentence-transformers not installed")
        all_ok = False

    try:
        import chromadb
        print("✓ chromadb installed")
    except:
        print("✗ chromadb not installed")
        all_ok = False

    if platform.system() == "Darwin":
        print("✓ macOS detected - will use 'say' command for TTS")
        result = subprocess.run(['which', 'say'], capture_output=True)
        if result.returncode == 0:
            print("✓ 'say' command available")
        else:
            print("✗ 'say' command not found")
            all_ok = False
    else:
        try:
            import pyttsx3
            print("✓ pyttsx3 installed")
        except:
            print("✗ pyttsx3 not installed")
            all_ok = False

    model_path = "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    if os.path.exists(model_path):
        print(f"✓ Mistral model found at {model_path}")
    else:
        print(f"✗ Mistral model NOT found at {model_path}")
        all_ok = False

    print("\n" + "="*60)
    if all_ok:
        print("✅ All requirements met!")
    else:
        print("⚠️  Some requirements are missing. Please install them first.")
    print("="*60 + "\n")

    return all_ok


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
    print("🧸 Teddy Bear AI - Speech to Speech con RAG Local")
    print("="*60 + "\n")

    if not check_system_requirements():
        print("\n⚠️  Por favor instala las dependencias faltantes:")
        print("pip install -r requirements.txt\n")

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
        logging.error(f"Critical error: {e}", exc_info=True)
        print(f"\n❌ Error crítico: {e}")
        print("\nAsegúrate de haber instalado todas las dependencias:")
        print("pip install -r requirements.txt")
