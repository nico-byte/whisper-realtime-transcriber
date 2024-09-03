import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import numpy as np
from whisper_realtime_transcriber.InputStreamGenerator import InputStreamGenerator
from whisper_realtime_transcriber.WhisperModel import WhisperModel

@pytest.fixture
def mock_inputstream_generator():
    generator = MagicMock(spec=InputStreamGenerator)
    generator.samplerate = 16000
    generator.data_ready_event = AsyncMock()
    generator.temp_ndarray = np.random.rand(16000).astype(np.float32)
    return generator

@pytest.fixture
def mock_whisper_model():
    with patch('transformers.WhisperForConditionalGeneration') as mock_model_class, \
         patch('transformers.WhisperProcessor') as mock_processor_class:
        
        mock_model = MagicMock()
        mock_processor = MagicMock()
        
        mock_model_class.from_pretrained.return_value = mock_model
        mock_processor_class.from_pretrained.return_value = mock_processor
        
        yield mock_model, mock_processor

@pytest.mark.asyncio
async def test_transcribe(mock_inputstream_generator, mock_whisper_model):
    mock_model, mock_processor = mock_whisper_model
    
    whisper_model = WhisperModel(
        inputstream_generator=mock_inputstream_generator,
        continuous=False,
        verbose=False
    )
    
    await whisper_model._transcribe()

    assert whisper_model.transcription != "" and isinstance(whisper_model.transcription, str)

@pytest.mark.asyncio
async def test_run_inference_non_continuous(mock_inputstream_generator, mock_whisper_model):
    mock_model, mock_processor = mock_whisper_model
    mock_inputstream_generator.data_ready_event.wait = AsyncMock()
    
    whisper_model = WhisperModel(
        inputstream_generator=mock_inputstream_generator,
        continuous=False,
        verbose=False
    )
    
    await whisper_model.run_inference()

    assert mock_inputstream_generator.data_ready_event.wait.called
    assert whisper_model.transcription != "" and isinstance(whisper_model.transcription, str)

@pytest.mark.asyncio
async def test_run_inference_continuous(mock_inputstream_generator, mock_whisper_model):
    mock_model, mock_processor = mock_whisper_model
    mock_inputstream_generator.data_ready_event.wait = AsyncMock()
    
    whisper_model = WhisperModel(
        inputstream_generator=mock_inputstream_generator,
        continuous=True,
        verbose=False
    )
    
    # Run the inference in continuous mode and break it after some iterations
    async def side_effect(*args, **kwargs):
        if not hasattr(side_effect, "counter"):
            side_effect.counter = 0
        side_effect.counter += 1
        if side_effect.counter >= 2:
            whisper_model.continuous = False
        return True
    
    mock_inputstream_generator.data_ready_event.wait.side_effect = side_effect
    
    await whisper_model.run_inference()
    
    assert side_effect.counter == 2
    assert whisper_model.transcription != "" and isinstance(whisper_model.transcription, str)

@pytest.mark.asyncio
async def test_print_transcriptions(mock_inputstream_generator, mock_whisper_model, capsys):
    mock_model, mock_processor = mock_whisper_model
    
    whisper_model = WhisperModel(
        inputstream_generator=mock_inputstream_generator,
        continuous=False,
        verbose=True
    )
    
    whisper_model.transcription = "This is a test transcription."
    
    await whisper_model._print_transcriptions()
    
    captured = capsys.readouterr()
    assert "This is a test transcription." in captured.out