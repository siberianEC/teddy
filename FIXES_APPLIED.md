# Teddy Bear AI - Segmentation Fault Fixes Applied (v3 - FINAL)

## Summary of Changes

This document describes all fixes applied to resolve segmentation faults and semaphore leaks in the Teddy Bear AI application on macOS.

## Root Causes Identified

1. **pyttsx3 Segmentation Fault**: pyttsx3 library crashes on macOS with Python 3.12+ due to threading issues
2. **sentence-transformers Segmentation Fault**: PyTorch multiprocessing using 'fork' method causes crashes on macOS
3. **ChromaDB Semaphore Leaks**: Persistent ChromaDB creates semaphores that aren't cleaned up properly
4. **No Resource Cleanup**: Missing proper cleanup handlers for multiprocessing resources

## Critical Fixes Applied (v3)

### 1. Force Multiprocessing 'spawn' Method on macOS

**CRITICAL FIX**: Set multiprocessing to use 'spawn' instead of 'fork' on macOS before any imports.

```python
import multiprocessing
if platform.system() == "Darwin":
    multiprocessing.set_start_method('spawn', force=True)
```

This prevents the segmentation fault in sentence-transformers/PyTorch which occurs because:
- macOS uses 'fork' by default which is unsafe with multi-threaded libraries
- PyTorch and transformers use threads that don't survive 'fork'
- 'spawn' creates a clean process without inheriting thread state

### 2. Disable Threading in PyTorch and Tokenizers

Set environment variables before any model loading:

```python
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
```

And explicitly configure PyTorch:

```python
import torch
torch.set_num_threads(1)
```

This eliminates threading conflicts that cause crashes.

### 3. Completely Bypass pyttsx3 on macOS

pyttsx3 is **NEVER** loaded on macOS:

```python
if self.is_macos:
    self.use_system_tts = True
    self.tts_engine = None
    # Use only 'say' command
```

### 4. Configure sentence-transformers Safely

All encoding operations now use safe parameters:

```python
embedding = self.embedding_model.encode(
    text,
    show_progress_bar=False,    # Disable tqdm which can cause issues
    convert_to_numpy=True        # Ensure numpy arrays
).tolist()
```

### 5. ChromaDB Ephemeral Mode

```python
settings = chromadb.Settings(
    anonymized_telemetry=False,  # No telemetry threads
    allow_reset=True,             # Allow cleanup
    is_persistent=False           # No file locks or semaphores
)
```

### 6. Enhanced Cleanup with Garbage Collection

```python
def cleanup(self):
    # Delete all objects
    del self.collection
    del self.chroma_client
    # Force garbage collection
    import gc
    gc.collect()
```

## All Fixes Summary

### Import Order and Configuration
1. Set multiprocessing spawn mode FIRST (before any other imports)
2. Disable tokenizer parallelism
3. Limit OpenMP threads to 1
4. Import all libraries after configuration

### Model Loading
1. Explicitly set PyTorch to single-threaded mode
2. Force sentence-transformers to CPU
3. Disable progress bars in all encoding operations
4. Use convert_to_numpy=True for consistent behavior

### TTS Handling
1. On macOS: Use only 'say' command, never pyttsx3
2. On other OS: Use pyttsx3 normally
3. Non-blocking timeout protection on 'say' command
4. Graceful degradation to text-only if TTS fails

### Resource Management
1. Ephemeral ChromaDB (no persistence)
2. Explicit deletion of all objects
3. Cache clearing before deletion
4. Forced garbage collection
5. Signal handlers for SIGINT/SIGTERM
6. Atexit cleanup registration

### Error Handling
1. Try-except around all model operations
2. Comprehensive logging at every step
3. Graceful degradation on failures
4. Clear error messages with stack traces

## Technical Explanation

### Why 'spawn' Instead of 'fork'?

On macOS, the default 'fork' method:
- Copies the entire process including thread states
- PyTorch/transformers have background threads
- These threads don't survive fork properly
- Results in corrupted memory and segmentation faults

The 'spawn' method:
- Starts a fresh Python interpreter
- No thread state is inherited
- Clean initialization of all libraries
- No segmentation faults

### Why Disable Parallelism?

Multiple levels of parallelism (PyTorch threads + tokenizer threads + ChromaDB threads):
- Create race conditions
- Cause memory corruption
- Trigger semaphore leaks
- Result in segmentation faults

Single-threaded mode:
- Sequential execution (slower but stable)
- No race conditions
- No thread-related crashes
- Proper resource cleanup

### Why Ephemeral ChromaDB?

Persistent ChromaDB:
- Creates semaphores for file locking
- These semaphores persist after process death
- macOS resource tracker detects leaks
- Can exhaust system semaphore limit

Ephemeral ChromaDB:
- No file system operations
- No semaphores created
- Memory-only operation
- Clean shutdown

## How to Use

### Installation

```bash
pip install -r requirements.txt
```

### Running

```bash
python teddy_bear_ai.py
```

### Expected Output on macOS

```
🧸 Teddy Bear AI - Speech to Speech con RAG Local
============================================================

🔍 VERIFICANDO REQUISITOS DEL SISTEMA
============================================================

✓ Python version: 3.12.x
✓ sounddevice installed
✓ Found X audio devices
✓ numpy installed
✓ faster-whisper installed
✓ llama-cpp-python installed
✓ sentence-transformers installed
✓ chromadb installed
✓ macOS detected - will use 'say' command for TTS
✓ 'say' command available
✓ Mistral model found

============================================================
✅ All requirements met!
============================================================

🧸 Inicializando Teddy Bear AI...
2025-xx-xx xx:xx:xx,xxx - INFO - Starting TeddyBearAI initialization
📡 Cargando Faster Whisper...
2025-xx-xx xx:xx:xx,xxx - INFO - Loading Faster Whisper model
2025-xx-xx xx:xx:xx,xxx - INFO - Whisper model loaded successfully
🤖 Cargando Mistral 7B...
2025-xx-xx xx:xx:xx,xxx - INFO - Loading Mistral 7B model
2025-xx-xx xx:xx:xx,xxx - INFO - Mistral 7B model loaded successfully
🔍 Configurando RAG con ChromaDB...
2025-xx-xx xx:xx:xx,xxx - INFO - Setting up ChromaDB
2025-xx-xx xx:xx:xx,xxx - INFO - Loading sentence transformer model (this may take a moment)...
2025-xx-xx xx:xx:xx,xxx - INFO - Sentence transformer loaded successfully
2025-xx-xx xx:xx:xx,xxx - INFO - ChromaDB configured successfully
🔊 Inicializando Text-to-Speech...
2025-xx-xx xx:xx:xx,xxx - INFO - Initializing Text-to-Speech
2025-xx-xx xx:xx:xx,xxx - INFO - Detected macOS, using system 'say' command to avoid pyttsx3 crashes
2025-xx-xx xx:xx:xx,xxx - INFO - macOS 'say' command configured successfully
📚 Inicializando base de conocimientos...
2025-xx-xx xx:xx:xx,xxx - INFO - Encoding knowledge base entries...
2025-xx-xx xx:xx:xx,xxx - INFO - Knowledge base initialized successfully
✅ Teddy Bear AI listo para hablar!
2025-xx-xx xx:xx:xx,xxx - INFO - TeddyBearAI initialization completed
💬 Teddy: Hola! Soy Teddy, tu peluche amigo. Habla conmigo!
2025-xx-xx xx:xx:xx,xxx - INFO - Speaking: Hola! Soy Teddy, tu peluche amigo. Habla conmigo!

==================================================
🧸 Teddy Bear AI activado
Presiona ENTER para hablar, 'q' + ENTER para salir
==================================================
```

### Key Success Indicators

1. **No segmentation faults** during initialization
2. **No semaphore leak warnings** on exit
3. **All models load successfully** with progress logged
4. **TTS works** using macOS 'say' command
5. **Clean exit** when pressing Ctrl+C or typing 'q'

## Testing the Fixes

Run these tests to verify everything works:

1. **Initialization Test**: Run the app - should complete without crashes
2. **TTS Test**: Should hear "Hola! Soy Teddy..." in Spanish
3. **Audio Test**: Press ENTER, speak for 5 seconds - should transcribe
4. **Response Test**: Should generate and speak a response
5. **Exit Test**: Press Ctrl+C or type 'q' - should exit cleanly without warnings

## Known Limitations

1. **Memory not persisted**: Conversations lost between runs (ephemeral ChromaDB)
2. **Single-threaded**: Slower than multi-threaded but stable
3. **macOS-specific**: Some fixes only apply to macOS
4. **'say' command only**: No pyttsx3 voice customization on macOS
5. **Initial load time**: 30-60 seconds to load all models

## What Changed from Previous Versions

**v1**: Added error handling and TTS fallback (didn't work)
**v2**: Disabled pyttsx3 on macOS completely (helped but still crashed)
**v3**: Fixed multiprocessing method and threading (FINAL FIX)

The key insight was that the segmentation fault wasn't just pyttsx3 - it was **ANY** library using multiprocessing with 'fork' method on macOS. By forcing 'spawn' mode and disabling all parallelism, we eliminated all crash sources.

## Troubleshooting

### If segmentation fault still occurs:

1. Check Python version: `python3 --version` (should be 3.12+)
2. Verify multiprocessing: Add `print(multiprocessing.get_start_method())` after line 13 - should print 'spawn'
3. Check environment: Verify `TOKENIZERS_PARALLELISM=false` is set
4. Review logs: Look for which component crashes (Whisper, Mistral, Transformers, ChromaDB)

### If semaphore leak warning appears:

1. Verify ChromaDB settings: Check `is_persistent=False`
2. Check cleanup is called: Should see "Starting cleanup..." in logs
3. Verify garbage collection: Should see "Garbage collection completed"
4. Check for orphaned processes: `ps aux | grep python`

### If TTS doesn't work on macOS:

1. Test 'say' command: `say "test"` in Terminal
2. Check permissions: System Preferences > Security & Privacy > Microphone
3. Verify 'say' is available: `which say`
4. Check logs for TTS errors

## Additional Notes

- Requires at least 8GB RAM (16GB recommended)
- Initial model download: ~5GB (Mistral + Whisper + sentence-transformers)
- First run takes longer (models downloaded/cached)
- macOS audio permissions required
- Mistral model must be in `./models/` directory

## Why This Solution Works

1. **'spawn' multiprocessing**: Creates clean processes without thread state corruption
2. **Single-threaded operation**: Eliminates all race conditions and thread conflicts
3. **Ephemeral ChromaDB**: No file locks or persistent semaphores
4. **Native TTS**: Uses stable macOS 'say' command instead of crashy pyttsx3
5. **Explicit cleanup**: Forces proper resource deallocation
6. **No progress bars**: Eliminates tqdm threading issues

All segmentation faults and semaphore leaks should now be resolved.
