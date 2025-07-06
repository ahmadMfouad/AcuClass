use cpal::{Device, Stream, StreamConfig, SampleFormat};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::{HeapRb, HeapProducer, HeapConsumer};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::time;
use log::{info, error};

use crate::{
    AudioProcessor, MfccExtractor, StandardScaler, NoiseClassifier,
    NoiseLevel, Result
};

pub struct StreamProcessor {
    audio_processor: AudioProcessor,
    feature_extractor: MfccExtractor,
    scaler: StandardScaler,
    classifier: NoiseClassifier,
    sample_rate: u32,
    buffer_duration: Duration,
    prediction_interval: Duration,
}

impl StreamProcessor {
    pub fn new(model_path: &str) -> Result<Self> {
        let sample_rate = 16000;
        
        Ok(Self {
            audio_processor: AudioProcessor::new(sample_rate),
            feature_extractor: MfccExtractor::new(sample_rate as f32, 13),
            scaler: StandardScaler::new(),
            classifier: NoiseClassifier::new(model_path)?,
            sample_rate,
            buffer_duration: Duration::from_secs(30), // 30 seconds buffer
            prediction_interval: Duration::from_secs(30), // Predict every 30 seconds
        })
    }

    pub async fn start_stream_processing(&self) -> Result<()> {
        let host = cpal::default_host();
        let device = host.default_input_device()
            .ok_or_else(|| anyhow::anyhow!("No input device available"))?;

        info!("Using input device: {}", device.name()?);

        let config = device.default_input_config()?;
        info!("Default input config: {:?}", config);

        let sample_format = config.sample_format();
        let config: StreamConfig = config.into();

        // Create ring buffer for audio data
        let buffer_size = (self.sample_rate as f64 * self.buffer_duration.as_secs_f64()) as usize;
        let ring_buffer = HeapRb::<f32>::new(buffer_size * 2); // Double buffer for safety
        let (producer, consumer) = ring_buffer.split();

        let producer = Arc::new(Mutex::new(producer));
        let consumer = Arc::new(Mutex::new(consumer));

        // Start audio capture stream
        let stream = self.create_input_stream(&device, &config, sample_format, producer.clone())?;
        stream.play()?;

        info!("Started audio capture stream");

        // Start processing task
        let consumer_clone = consumer.clone();
        let mut processor = self.clone_for_async();
        
        tokio::spawn(async move {
            processor.process_audio_buffer(consumer_clone).await;
        });

        // Keep the stream alive
        tokio::signal::ctrl_c().await?;
        info!("Shutting down stream processor");

        Ok(())
    }

    fn create_input_stream(
        &self,
        device: &Device,
        config: &StreamConfig,
        sample_format: SampleFormat,
        producer: Arc<Mutex<HeapProducer<f32>>>,
    ) -> Result<Stream> {
        let channels = config.channels as usize;
        let target_sample_rate = self.sample_rate;
        let source_sample_rate = config.sample_rate.0;

        let stream = match sample_format {
            SampleFormat::F32 => device.build_input_stream(
                config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    Self::process_input_data(data, channels, source_sample_rate, target_sample_rate, &producer);
                },
                |err| error!("Audio stream error: {}", err),
                None,
            )?,
            SampleFormat::I16 => device.build_input_stream(
                config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let float_data: Vec<f32> = data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
                    Self::process_input_data(&float_data, channels, source_sample_rate, target_sample_rate, &producer);
                },
                |err| error!("Audio stream error: {}", err),
                None,
            )?,
            SampleFormat::U16 => device.build_input_stream(
                config,
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    let float_data: Vec<f32> = data.iter().map(|&s| (s as f32 - 32768.0) / 32768.0).collect();
                    Self::process_input_data(&float_data, channels, source_sample_rate, target_sample_rate, &producer);
                },
                |err| error!("Audio stream error: {}", err),
                None,
            )?,
            _ => return Err(anyhow::anyhow!("Unsupported sample format: {:?}", sample_format)),
        };

        Ok(stream)
    }

    fn process_input_data(
        data: &[f32],
        channels: usize,
        source_sample_rate: u32,
        target_sample_rate: u32,
        producer: &Arc<Mutex<HeapProducer<f32>>>,
    ) {
        // Convert to mono
        let mono_data: Vec<f32> = if channels == 1 {
            data.to_vec()
        } else {
            data.chunks(channels)
                .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
                .collect()
        };

        // Simple resampling if needed
        let resampled_data = if source_sample_rate != target_sample_rate {
            Self::simple_resample(&mono_data, source_sample_rate, target_sample_rate)
        } else {
            mono_data
        };

        // Push to ring buffer
        if let Ok(mut producer) = producer.lock() {
            for &sample in &resampled_data {
                if producer.push(sample).is_err() {
                    // Buffer full, skip this sample
                    break;
                }
            }
        }
    }

    fn simple_resample(data: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
        let ratio = to_rate as f32 / from_rate as f32;
        let new_length = (data.len() as f32 * ratio) as usize;
        let mut resampled = Vec::with_capacity(new_length);

        for i in 0..new_length {
            let src_index = i as f32 / ratio;
            let src_index_floor = src_index.floor() as usize;
            
            if src_index_floor < data.len() {
                resampled.push(data[src_index_floor]);
            }
        }

        resampled
    }

    async fn process_audio_buffer(&mut self, consumer: Arc<Mutex<HeapConsumer<f32>>>) {
        let mut interval = time::interval(self.prediction_interval);
        let buffer_size = (self.sample_rate as f64 * self.buffer_duration.as_secs_f64()) as usize;

        loop {
            interval.tick().await;

            let audio_data = {
                let mut consumer = consumer.lock().unwrap();
                let available = consumer.len();
                
                if available < buffer_size {
                    info!("Not enough audio data for prediction ({} < {})", available, buffer_size);
                    continue;
                }

                let mut buffer = vec![0.0f32; buffer_size];
                for i in 0..buffer_size {
                    if let Some(sample) = consumer.pop() {
                        buffer[i] = sample;
                    }
                }
                buffer
            };

            match self.process_audio_chunk(&audio_data).await {
                Ok(prediction) => {
                    info!("Noise level prediction: {}", prediction.as_str());
                }
                Err(e) => {
                    error!("Error processing audio chunk: {}", e);
                }
            }
        }
    }

    async fn process_audio_chunk(&mut self, audio: &[f32]) -> Result<NoiseLevel> {
        // Extract 2-second chunks
        let chunks = self.audio_processor.extract_chunks(audio, 2.0);
        
        if chunks.is_empty() {
            return Ok(NoiseLevel::High); // Default to high noise if no chunks
        }

        let mut predictions = Vec::new();

        // Process each chunk
        for chunk in chunks {
            let features = self.feature_extractor.extract_features(&chunk)?;
            let scaled_features = self.scaler.transform_single(&features);
            let prediction = self.classifier.predict_single(&scaled_features)?;
            predictions.push(prediction);
        }

        // Aggregate predictions
        self.classifier.aggregate_predictions(&predictions)
            .ok_or_else(|| anyhow::anyhow!("Failed to aggregate predictions"))
    }

    // Helper method to clone for async context
    fn clone_for_async(&self) -> Self {
        // Note: This is a simplified approach. In production, you might want to 
        // share the model between instances or handle this differently
        Self {
            audio_processor: AudioProcessor::new(self.sample_rate),
            feature_extractor: MfccExtractor::new(self.sample_rate as f32, 13),
            scaler: StandardScaler::new(),
            classifier: NoiseClassifier::new("./src/model.onnx").unwrap(), // Use the actual model path
            sample_rate: self.sample_rate,
            buffer_duration: self.buffer_duration,
            prediction_interval: self.prediction_interval,
        }
    }
}
