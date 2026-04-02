use anyhow::{Error as E, Result};
use candle::{DType, Device, IndexOp, Module, Tensor, D};
use candle_transformers::models::stable_diffusion;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
pub enum StableDiffusionVersion {
    #[value(name = "v1-5")]
    V1_5,
    #[value(name = "v2-1")]
    V2_1,
    #[value(name = "xl")]
    Xl,
    #[value(name = "turbo")]
    Turbo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelFile {
    Tokenizer,
    Tokenizer2,
    Clip,
    Clip2,
    Unet,
    Vae,
}

impl ModelFile {
    fn get(
        &self,
        version: StableDiffusionVersion,
        use_f16: bool,
    ) -> Result<std::path::PathBuf> {
        let (repo, path) = match self {
            Self::Tokenizer => {
                let repo = match version {
                    StableDiffusionVersion::V1_5 | StableDiffusionVersion::V2_1 => {
                        "openai/clip-vit-base-patch32"
                    }
                    StableDiffusionVersion::Xl | StableDiffusionVersion::Turbo => {
                        "openai/clip-vit-large-patch14"
                    }
                };
                (repo, "tokenizer.json".to_string())
            }
            Self::Tokenizer2 => (
                "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                "tokenizer.json".to_string(),
            ),
            Self::Clip => (
                version.repo(),
                version.clip_file(use_f16).to_string(),
            ),
            Self::Clip2 => (
                version.repo(),
                version.clip2_file(use_f16).to_string(),
            ),
            Self::Unet => (
                version.repo(),
                version.unet_file(use_f16).to_string(),
            ),
            Self::Vae => {
                if matches!(
                    version,
                    StableDiffusionVersion::Xl | StableDiffusionVersion::Turbo
                ) && use_f16
                {
                    (
                        "madebyollin/sdxl-vae-fp16-fix",
                        "diffusion_pytorch_model.safetensors".to_string(),
                    )
                } else {
                    (
                        version.repo(),
                        version.vae_file(use_f16).to_string(),
                    )
                }
            }
        };
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(repo.to_string());
        let filename = repo.get(&path)?;
        Ok(filename)
    }
}

impl StableDiffusionVersion {
    fn repo(&self) -> &'static str {
        match self {
            Self::V1_5 => "runwayml/stable-diffusion-v1-5",
            Self::V2_1 => "stabilityai/stable-diffusion-2-1",
            Self::Xl => "stabilityai/stable-diffusion-xl-base-1.0",
            Self::Turbo => "stabilityai/sdxl-turbo",
        }
    }

    fn unet_file(&self, use_f16: bool) -> &'static str {
        if use_f16 {
            "unet/diffusion_pytorch_model.fp16.safetensors"
        } else {
            "unet/diffusion_pytorch_model.safetensors"
        }
    }

    fn vae_file(&self, use_f16: bool) -> &'static str {
        if use_f16 {
            "vae/diffusion_pytorch_model.fp16.safetensors"
        } else {
            "vae/diffusion_pytorch_model.safetensors"
        }
    }

    fn clip_file(&self, use_f16: bool) -> &'static str {
        if use_f16 {
            "text_encoder/model.fp16.safetensors"
        } else {
            "text_encoder/model.safetensors"
        }
    }

    fn clip2_file(&self, use_f16: bool) -> &'static str {
        if use_f16 {
            "text_encoder_2/model.fp16.safetensors"
        } else {
            "text_encoder_2/model.safetensors"
        }
    }
}

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if candle::utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else if candle::utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else {
        eprintln!("No GPU detected, falling back to CPU");
        Ok(Device::Cpu)
    }
}

fn save_image(img: &Tensor, path: &str) -> Result<()> {
    let (channel, height, width) = img.dims3()?;
    if channel != 3 {
        anyhow::bail!("expected 3-channel image, got {channel}");
    }
    let img = img.to_device(&Device::Cpu)?;
    let img = (img.clamp(0f32, 1f32)? * 255.0)?.to_dtype(DType::U8)?;
    let pixels = img.flatten_all()?.to_vec1::<u8>()?;
    let image_buf: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        image::ImageBuffer::from_raw(width as u32, height as u32, pixels)
            .ok_or_else(|| anyhow::anyhow!("failed to create image buffer"))?;
    image_buf.save(path)?;
    println!("Saved image to {path}");
    Ok(())
}

fn text_embeddings(
    prompt: &str,
    uncond_prompt: &str,
    sd_version: StableDiffusionVersion,
    sd_config: &stable_diffusion::StableDiffusionConfig,
    use_f16: bool,
    device: &Device,
    dtype: DType,
    use_guide_scale: bool,
    first: bool,
) -> Result<Tensor> {
    let clip_config = if first {
        &sd_config.clip
    } else {
        sd_config.clip2.as_ref().unwrap()
    };

    let tokenizer_file = if first {
        ModelFile::Tokenizer.get(sd_version, use_f16)?
    } else {
        ModelFile::Tokenizer2.get(sd_version, use_f16)?
    };
    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(E::msg)?;

    let pad_id = match &clip_config.pad_with {
        Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
        None => {
            let eos = tokenizer.encode("", false).map_err(E::msg)?;
            *eos.get_ids().last().unwrap()
        }
    };

    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let max_len = clip_config.max_position_embeddings;
    if tokens.len() > max_len {
        tokens.truncate(max_len);
    }
    while tokens.len() < max_len {
        tokens.push(pad_id);
    }

    let tokens = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;

    let clip_weights = if first {
        ModelFile::Clip.get(sd_version, use_f16)?
    } else {
        ModelFile::Clip2.get(sd_version, use_f16)?
    };
    let text_model = stable_diffusion::build_clip_transformer(
        clip_config,
        clip_weights,
        device,
        dtype,
    )?;

    let text_embeddings = text_model.forward(&tokens)?;

    if use_guide_scale {
        let mut uncond_tokens = tokenizer
            .encode(uncond_prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        if uncond_tokens.len() > max_len {
            uncond_tokens.truncate(max_len);
        }
        while uncond_tokens.len() < max_len {
            uncond_tokens.push(pad_id);
        }
        let uncond_tokens = Tensor::new(uncond_tokens.as_slice(), device)?.unsqueeze(0)?;
        let uncond_embeddings = text_model.forward(&uncond_tokens)?;
        Tensor::cat(&[uncond_embeddings, text_embeddings], 0).map_err(Into::into)
    } else {
        Ok(text_embeddings)
    }
}

pub struct RunArgs {
    pub prompt: String,
    pub uncond_prompt: String,
    pub cpu: bool,
    pub final_image: String,
    pub n_steps: Option<usize>,
    pub sd_version: StableDiffusionVersion,
    pub use_f16: bool,
    pub seed: Option<u64>,
    pub guidance_scale: Option<f64>,
    pub height: Option<usize>,
    pub width: Option<usize>,
    pub num_samples: usize,
}

pub fn run(args: RunArgs) -> Result<()> {
    let device = device(args.cpu)?;
    let dtype = if args.use_f16 { DType::F16 } else { DType::F32 };
    let sd_version = args.sd_version;

    let which_clips: Vec<bool> = match sd_version {
        StableDiffusionVersion::Xl | StableDiffusionVersion::Turbo => vec![true, false],
        _ => vec![true],
    };

    let (default_guidance, default_steps, vae_scale) = match sd_version {
        StableDiffusionVersion::Turbo => (0.0, 1_usize, 0.13025),
        _ => (7.5, 30_usize, 0.18215),
    };

    let guidance_scale = args.guidance_scale.unwrap_or(default_guidance);
    let n_steps = args.n_steps.unwrap_or(default_steps);
    let use_guide_scale = guidance_scale > 1.0;

    let height = args.height.unwrap_or_else(|| match sd_version {
        StableDiffusionVersion::V1_5 | StableDiffusionVersion::Turbo => 512,
        StableDiffusionVersion::V2_1 => 768,
        StableDiffusionVersion::Xl => 1024,
    });
    let width = args.width.unwrap_or_else(|| match sd_version {
        StableDiffusionVersion::V1_5 | StableDiffusionVersion::Turbo => 512,
        StableDiffusionVersion::V2_1 => 768,
        StableDiffusionVersion::Xl => 1024,
    });

    let sd_config = match sd_version {
        StableDiffusionVersion::V1_5 => {
            stable_diffusion::StableDiffusionConfig::v1_5(None, Some(height), Some(width))
        }
        StableDiffusionVersion::V2_1 => {
            stable_diffusion::StableDiffusionConfig::v2_1(None, Some(height), Some(width))
        }
        StableDiffusionVersion::Xl => {
            stable_diffusion::StableDiffusionConfig::sdxl(None, Some(height), Some(width))
        }
        StableDiffusionVersion::Turbo => {
            stable_diffusion::StableDiffusionConfig::sdxl_turbo(None, Some(height), Some(width))
        }
    };

    if let Some(seed) = args.seed {
        device.set_seed(seed)?;
    }

    println!("Building text embeddings ({sd_version:?})...");
    let embeddings: Vec<Tensor> = which_clips
        .iter()
        .map(|first| {
            text_embeddings(
                &args.prompt,
                &args.uncond_prompt,
                sd_version,
                &sd_config,
                args.use_f16,
                &device,
                dtype,
                use_guide_scale,
                *first,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let text_embeddings = Tensor::cat(&embeddings, D::Minus1)?;
    println!("Text embeddings shape: {:?}", text_embeddings.shape());

    println!("Loading VAE...");
    let vae_weights = ModelFile::Vae.get(sd_version, args.use_f16)?;
    let vae = sd_config.build_vae(vae_weights, &device, dtype)?;

    println!("Loading UNet...");
    let unet_weights = ModelFile::Unet.get(sd_version, args.use_f16)?;
    let unet = sd_config.build_unet(unet_weights, &device, 4, false, dtype)?;

    println!("Building scheduler ({n_steps} steps)...");
    let mut scheduler = sd_config.build_scheduler(n_steps)?;

    for sample_idx in 0..args.num_samples {
        println!(
            "Generating sample {}/{}...",
            sample_idx + 1,
            args.num_samples
        );

        let latents = Tensor::randn(
            0f32,
            1f32,
            (1, 4, height / 8, width / 8),
            &device,
        )?;
        let mut latents = (latents * scheduler.init_noise_sigma())?;
        let timesteps = scheduler.timesteps().to_vec();

        for (step, &timestep) in timesteps.iter().enumerate() {
            println!("  Step {}/{n_steps} (t={timestep})...", step + 1);

            let latent_model_input = if use_guide_scale {
                Tensor::cat(&[&latents, &latents], 0)?
            } else {
                latents.clone()
            };

            let latent_model_input =
                scheduler.scale_model_input(latent_model_input, timestep)?;

            let noise_pred =
                unet.forward(&latent_model_input, timestep as f64, &text_embeddings)?;

            let noise_pred = if use_guide_scale {
                let chunks = noise_pred.chunk(2, 0)?;
                let (uncond, text) = (&chunks[0], &chunks[1]);
                (uncond + ((text - uncond)? * guidance_scale)?)?
            } else {
                noise_pred
            };

            latents = scheduler.step(&noise_pred, timestep, &latents)?;
        }

        println!("Decoding latents with VAE...");
        let images = vae.decode(&(&latents / vae_scale)?)?;
        let images = ((images / 2.)? + 0.5)?;

        let image = images.i(0)?;
        let output_path = if args.num_samples > 1 {
            let stem = args.final_image.trim_end_matches(".png");
            format!("{stem}_{sample_idx}.png")
        } else {
            args.final_image.clone()
        };
        save_image(&image, &output_path)?;
    }

    Ok(())
}
