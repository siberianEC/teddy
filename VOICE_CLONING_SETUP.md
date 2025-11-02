# Configuración de Clonación de Voz con XTTS-v2

## Requisitos

1. Instala las dependencias actualizadas:
```bash
pip install -r requirements.txt
```

## Cómo Usar la Clonación de Voz

### Paso 1: Grabar una Muestra de Voz

Necesitas crear un archivo de audio WAV con la voz que deseas clonar:

- **Duración recomendada**: 6-10 segundos
- **Formato**: WAV (mono o estéreo)
- **Calidad**: Grabación limpia, sin ruido de fondo
- **Contenido**: Habla clara y natural en español

Puedes grabar usando:

```bash
# En macOS
sox -d voice_sample.wav

# O usando Python
python3 << EOF
import sounddevice as sd
import scipy.io.wavfile as wavfile
import numpy as np

sample_rate = 22050
duration = 10  # segundos

print("Grabando en 3... 2... 1...")
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
sd.wait()
wavfile.write('voice_sample.wav', sample_rate, audio)
print("Grabación completada: voice_sample.wav")
EOF
```

### Paso 2: Coloca el Archivo de Muestra

Guarda tu archivo de audio como:
```
./voice_sample.wav
```

En el mismo directorio donde está `teddy_bear_ai.py`

### Paso 3: Ejecuta Teddy Bear AI

```bash
python3 teddy_bear_ai.py
```

El sistema automáticamente:
1. Detectará el archivo `voice_sample.wav`
2. Usará XTTS-v2 para clonar esa voz
3. Generará respuestas con la voz clonada en español

## Opciones Avanzadas

Si deseas usar un archivo diferente, edita la línea en `teddy_bear_ai.py`:

```python
self.voice_sample_path = "./tu_archivo_de_voz.wav"
```

## Fallback

Si no existe `voice_sample.wav`, el sistema automáticamente usará la voz de macOS "Paulina" como respaldo.

## Notas de Rendimiento

- XTTS-v2 requiere más recursos que las voces del sistema
- Con GPU: generación rápida (~2-3 segundos)
- Sin GPU (CPU): puede tardar 5-10 segundos por respuesta
- La primera vez que se use tardará más mientras descarga el modelo (~2GB)
