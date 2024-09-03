import pytest
import numpy as np
import asyncio
from unittest.mock import patch, AsyncMock
from whisper_realtime_transcriber.InputStreamGenerator import InputStreamGenerator  # Replace with the actual module name

def test_initialization():
    gen = InputStreamGenerator(samplerate=8000, blocksize=2000, adjustment_time=3, min_chunks=4, memory_safe=False, verbose=False)
    
    assert gen.samplerate == 8000
    assert gen._blocksize == 2000
    assert gen._adjustment_time == 3
    assert gen._min_chunks == 4
    assert gen.memory_safe is False
    assert gen.verbose is False
    assert gen.data_ready_event.is_set() is False
    assert gen._global_ndarray is None
    

@pytest.mark.asyncio
async def test_generate():
    mock_stream = AsyncMock()

    with patch('sounddevice.InputStream', return_value=mock_stream):
        gen = InputStreamGenerator()
        generator = gen._generate()
        mock_stream.__enter__.return_value = mock_stream

        async def async_mock_get():
            await asyncio.sleep(0.1)
            return np.array([1, 2, 3]), 'mock_status'

        mock_queue = AsyncMock()
        mock_queue.get.side_effect = async_mock_get

        with patch('asyncio.Queue', return_value=mock_queue):
            indata, status = await generator.__anext__()

            assert np.array_equal(indata, np.array([1, 2, 3]))
            assert status == 'mock_status'
