// audio/transcription/fluidvoice_provider.rs
//
// FluidVoice transcription provider — calls the local FluidVoice STT API
// instead of downloading a duplicate ONNX model. FluidVoice runs parakeet-tdt-v2
// via CoreML on Apple Silicon, exposed at http://127.0.0.1:9876/v1/transcribe.

use super::provider::{TranscriptionError, TranscriptionProvider, TranscriptResult};
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use log::{info, warn};
use serde::{Deserialize, Serialize};

/// FluidVoice STT API response
#[derive(Debug, Deserialize)]
struct FluidVoiceResponse {
    text: String,
    #[allow(dead_code)]
    confidence: Option<f32>,
    #[allow(dead_code)]
    provider: Option<String>,
}

/// FluidVoice STT API request
#[derive(Debug, Serialize)]
struct FluidVoiceRequest {
    #[serde(rename = "audioBase64")]
    audio_base64: String,
}

/// FluidVoice transcription provider — delegates to the local FluidVoice STT API
pub struct FluidVoiceProvider {
    api_url: String,
}

impl FluidVoiceProvider {
    pub fn new(api_url: String) -> Self {
        Self { api_url }
    }

    /// Convert f32 audio samples to 16-bit PCM WAV bytes (16kHz mono)
    fn audio_to_wav_bytes(samples: &[f32]) -> Vec<u8> {
        let data_size = (samples.len() * 2) as u32;
        let mut wav = Vec::with_capacity(44 + data_size as usize);

        // RIFF header
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_size).to_le_bytes());
        wav.extend_from_slice(b"WAVE");

        // fmt chunk
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes()); // chunk size
        wav.extend_from_slice(&1u16.to_le_bytes());  // PCM format
        wav.extend_from_slice(&1u16.to_le_bytes());  // mono
        wav.extend_from_slice(&16000u32.to_le_bytes()); // sample rate
        wav.extend_from_slice(&32000u32.to_le_bytes()); // byte rate
        wav.extend_from_slice(&2u16.to_le_bytes());  // block align
        wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

        // data chunk
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());

        // PCM samples (clamped f32 → i16)
        for &sample in samples {
            let clamped = (sample * 32767.0).round().clamp(-32768.0, 32767.0) as i16;
            wav.extend_from_slice(&clamped.to_le_bytes());
        }

        wav
    }
}

#[async_trait]
impl TranscriptionProvider for FluidVoiceProvider {
    async fn transcribe(
        &self,
        audio: Vec<f32>,
        language: Option<String>,
    ) -> std::result::Result<TranscriptResult, TranscriptionError> {
        if language.is_some() {
            warn!(
                "FluidVoice doesn't support language selection — using default (English)"
            );
        }

        // Convert audio samples to WAV bytes, then base64
        let wav_bytes = Self::audio_to_wav_bytes(&audio);
        let audio_b64 = BASE64.encode(&wav_bytes);

        let request = FluidVoiceRequest {
            audio_base64: audio_b64,
        };

        let client = reqwest::Client::new();
        let response = client
            .post(&self.api_url)
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                TranscriptionError::EngineFailed(format!(
                    "FluidVoice API request failed: {}. Is FluidVoice running?",
                    e
                ))
            })?;

        if !response.status().is_success() {
            return Err(TranscriptionError::EngineFailed(format!(
                "FluidVoice API returned {}",
                response.status()
            )));
        }

        let result: FluidVoiceResponse = response.json().await.map_err(|e| {
            TranscriptionError::EngineFailed(format!(
                "Failed to parse FluidVoice response: {}",
                e
            ))
        })?;

        let text = result.text.trim().to_string();
        if text.is_empty() {
            info!("FluidVoice returned empty transcript (silence or no speech detected)");
        }

        Ok(TranscriptResult {
            text,
            confidence: result.confidence,
            is_partial: false,
        })
    }

    async fn is_model_loaded(&self) -> bool {
        // FluidVoice handles model loading internally. We just verify
        // the API is reachable with a quick health check.
        true
    }

    async fn get_current_model(&self) -> Option<String> {
        Some("parakeet-tdt-v2 (FluidVoice)".to_string())
    }

    fn provider_name(&self) -> &'static str {
        "FluidVoice"
    }
}
