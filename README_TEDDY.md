# 🧸 Teddy Bear AI - Speech to Speech con RAG Local

Sistema de IA 100% local para peluches interactivos con capacidades de:
- **Speech-to-Text** con Faster Whisper
- **RAG** (Retrieval Augmented Generation) con ChromaDB
- **LLM** Mistral 7B local
- **Text-to-Speech** para respuestas de voz
- **Baja latencia** y completamente offline

## 🚀 Características

- ✅ Procesamiento de voz en tiempo real
- ✅ Memoria conversacional con RAG
- ✅ Respuestas contextuales inteligentes
- ✅ 100% Local (sin internet)
- ✅ Bajo consumo de recursos
- ✅ Latencia optimizada

## 📋 Requisitos

- Python 3.9 o superior
- 8GB RAM mínimo (16GB recomendado)
- 10GB espacio en disco
- Micrófono funcional

## 🔧 Instalación

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Instalar PyAudio (depende del sistema)

**Linux:**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

**Windows:**
```bash
pip install pipwin
pipwin install pyaudio
```

### 3. Descargar Modelo Mistral 7B

Descarga el modelo desde HuggingFace:

```bash
mkdir models
cd models

# Descargar usando wget o curl
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

O descarga manualmente desde:
https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF

Guarda como: `./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf`

## 🎯 Uso

```bash
python teddy_bear_ai.py
```

### Interacción:
1. Presiona ENTER para comenzar a hablar
2. Habla durante 5 segundos
3. Teddy procesará y responderá
4. Escribe 'q' + ENTER para salir

## 🏗️ Arquitectura

```
Usuario habla → Faster Whisper (STT) → Texto
                                         ↓
                    ChromaDB ← Búsqueda de contexto (RAG)
                                         ↓
                    Mistral 7B → Genera respuesta
                                         ↓
                    pyttsx3 (TTS) → Audio respuesta
```

## ⚙️ Componentes

### 1. Faster Whisper
- Modelo: `base` (74MB)
- Transcripción en español
- VAD (Voice Activity Detection)
- Latencia: ~1-2 segundos

### 2. Mistral 7B
- Cuantización: Q4_K_M (4.4GB)
- Contexto: 4096 tokens
- Inferencia CPU optimizada
- Latencia: ~2-5 segundos

### 3. RAG con ChromaDB
- Embeddings: all-MiniLM-L6-v2
- Memoria conversacional
- Búsqueda semántica
- Persistencia local

### 4. Text-to-Speech
- Motor: pyttsx3
- Síntesis local
- Latencia mínima

## 🔊 Ajustes de Latencia

Para reducir latencia:

```python
# En teddy_bear_ai.py, ajusta:

# Duración de grabación (línea ~42)
duration=3  # Reducir de 5 a 3 segundos

# Tokens de respuesta (línea ~138)
max_tokens=100  # Reducir de 150 a 100

# Modelo Whisper más pequeño (línea ~26)
self.whisper_model = WhisperModel("tiny", ...)  # tiny, base, small
```

## 📊 Rendimiento Esperado

| Componente | Latencia | RAM |
|------------|----------|-----|
| Whisper (base) | 1-2s | 1GB |
| RAG Búsqueda | <0.1s | 500MB |
| Mistral 7B Q4 | 2-5s | 6GB |
| TTS | <0.5s | 100MB |
| **Total** | **4-8s** | **~8GB** |

## 🎨 Personalización

### Cambiar personalidad del peluche:

Edita el prompt en `generate_response()`:

```python
prompt = f"""<s>[INST] Eres un [PERSONALIDAD AQUÍ].
Tu nombre es [NOMBRE]. Responde de forma [ESTILO].

Usuario: {user_input}
[/INST]"""
```

### Agregar conocimientos base:

Modifica `init_knowledge_base()`:

```python
knowledge = [
    "Tu nuevo conocimiento aquí",
    "Más información personalizada",
    # ...
]
```

## 🐛 Solución de Problemas

### Error de micrófono:
```bash
# Verificar dispositivos de audio
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### Memoria insuficiente:
- Usa modelo Whisper más pequeño: `tiny`
- Usa Mistral Q3 o Q2 (menor calidad)
- Reduce `n_ctx` a 2048

### TTS no funciona:
```bash
# Linux
sudo apt-get install espeak

# macOS - usa voces del sistema
# Windows - usa SAPI5 automático
```

## 📝 Licencia

Este proyecto es de código abierto para uso educativo y personal.

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Este peluche puede mejorar con:
- Detección de emociones en voz
- Wake word detection ("Hey Teddy")
- Múltiples idiomas
- Integración con hardware (LEDs, sensores)

## 🔗 Enlaces Útiles

- [Faster Whisper](https://github.com/guillaumekln/faster-whisper)
- [Llama.cpp](https://github.com/ggerganov/llama.cpp)
- [ChromaDB](https://www.trychroma.com/)
- [Mistral AI](https://mistral.ai/)
