use audio_noise_classifier::{
    AudioProcessor, MfccExtractor, StandardScaler, NoiseClassifier, StreamProcessor,
    Result, NoiseLevel
};
use log::info;
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        println!("Usage:");
        println!("  {} file <audio_file> [model_path]", args[0]);
        println!("  {} stream [model_path]", args[0]);
        return Ok(());
    }

    let model_path = args.get(3)
        .or_else(|| args.get(2).filter(|_| args[1] == "stream"))
        .map(|s| s.as_str())
        .unwrap_or("./src/model.onnx");

    match args[1].as_str() {
        "file" => {
            if args.len() < 3 {
                println!("Please provide an audio file path");
                return Ok(());
            }
            process_file(&args[2], model_path).await?;
        }
        "stream" => {
            process_stream(model_path).await?;
        }
        _ => {
            println!("Invalid command. Use 'file' or 'stream'");
        }
    }

    Ok(())
}

async fn process_file(audio_file: &str, model_path: &str) -> Result<()> {
    info!("Processing audio file: {}", audio_file);
    info!("Using model: {}", model_path);

    let audio_processor = AudioProcessor::new(16000);
    let feature_extractor = MfccExtractor::new(16000.0, 13);
    let scaler = StandardScaler::new();
    let mut classifier = NoiseClassifier::new(model_path)?;

    // Load audio
    let audio = audio_processor.load_audio_file(audio_file)?;
    info!("Loaded audio with {} samples ({:.2} seconds)", 
          audio.len(), 
          audio.len() as f32 / 16000.0);

    // Extract 2-second chunks
    let chunks = audio_processor.extract_chunks(&audio, 2.0);
    info!("Extracted {} chunks", chunks.len());

    if chunks.is_empty() {
        println!("No chunks extracted from audio file");
        return Ok(());
    }

    // Process each chunk
    let mut chunk_predictions = Vec::new();

    for (i, chunk) in chunks.iter().enumerate() {
        let features = feature_extractor.extract_features(chunk)?;
        let scaled_features = scaler.transform_single(&features);
        let prediction = classifier.predict_single(&scaled_features)?;
        
        chunk_predictions.push(prediction);
        
        println!("Chunk {} predicted class: {}", i + 1, prediction.as_str());
    }

    // Aggregate predictions
    let aggregated_prediction = classifier.aggregate_predictions(&chunk_predictions);
    
    match aggregated_prediction {
        Some(prediction) => {
            println!("\n=== FINAL RESULT ===");
            println!("Aggregated predicted class: {}", prediction.as_str());
            
            // Print summary statistics
            let mut counts = [0; 3];
            for &pred in &chunk_predictions {
                counts[pred as usize] += 1;
            }
            
            println!("\nChunk-wise breakdown:");
            println!("  Low Noise:  {} chunks", counts[0]);
            println!("  Mid Noise:  {} chunks", counts[1]);
            println!("  High Noise: {} chunks", counts[2]);
        }
        None => {
            println!("No valid predictions due to processing error.");
        }
    }

    Ok(())
}

async fn process_stream(model_path: &str) -> Result<()> {
    info!("Starting real-time audio stream processing");
    info!("Using model: {}", model_path);
    
    let stream_processor = StreamProcessor::new(model_path)?;
    stream_processor.start_stream_processing().await?;

    Ok(())
}
