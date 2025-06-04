# 🔁 OpenAI RealtimeAI WebSocket Event Automation

This project automates the full event flow of OpenAI's `realtimeai` WebSocket-based API using Python and `pytest`. It simulates the client-side event lifecycle — from session initiation to audio input/output and final response validation — helping developers and testers verify correct behavior, sequencing, and payload integrity.

## Key Features

- Establishes WebSocket connection with OpenAI's `gpt-4o-realtime-preview` model
- Automates all supported client events:
  - `session.update`
  - `input_audio_buffer.append`
  - `input_audio_buffer.commit`
  - `input_audio_buffer.clear`
  - `conversation.item.create`
  - `conversation.item.retrieve`
  - `response.create`
- Loads test payloads from a structured data folder(JSON)
- Pytest-based modular test suite
- Response validation and sequence assertions
- Integrated reporting (HTML)

## Install dependencies
pip install -r requirements.txt

## Set up environment variables
OPENAI_API_KEY=your_openai_api_key

## Running Tests
pytest -v

## For HTML reporting:
pytest -s tests/test_E2E_audio_to_audio.py --html=Report/e2e_audio_report.html --self-contained-html