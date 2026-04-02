use anyhow::Result;
use clap::Parser;

#[derive(Parser)]
#[command(name = "candle-diffusion")]
#[command(about = "Stable Diffusion image generation using Candle (pure Rust, no C++ deps)")]
struct Args {
    /// The prompt to generate an image from.
    #[arg(
        long,
        default_value = "A very realistic photo of a rusty robot walking on a sandy beach"
    )]
    prompt: String,

    /// The unconditional prompt (negative prompt).
    #[arg(long, default_value = "")]
    uncond_prompt: String,

    /// Force CPU execution (disables Metal/CUDA).
    #[arg(long)]
    cpu: bool,

    /// Output image path.
    #[arg(long, default_value = "sd_final.png")]
    final_image: String,

    /// Number of diffusion steps (default varies by model version).
    #[arg(long)]
    n_steps: Option<usize>,

    /// Stable Diffusion version to use.
    #[arg(long, value_enum, default_value = "v2-1")]
    sd_version: candle_diffusion::StableDiffusionVersion,

    /// Use half-precision (f16) weights.
    #[arg(long)]
    use_f16: bool,

    /// Random seed for reproducible generation.
    #[arg(long)]
    seed: Option<u64>,

    /// Classifier-free guidance scale (default varies by model version).
    #[arg(long)]
    guidance_scale: Option<f64>,

    /// Image height (must be divisible by 8).
    #[arg(long)]
    height: Option<usize>,

    /// Image width (must be divisible by 8).
    #[arg(long)]
    width: Option<usize>,

    /// Number of samples to generate.
    #[arg(long, default_value_t = 1)]
    num_samples: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let run_args = candle_diffusion::RunArgs {
        prompt: args.prompt,
        uncond_prompt: args.uncond_prompt,
        cpu: args.cpu,
        final_image: args.final_image,
        n_steps: args.n_steps,
        sd_version: args.sd_version,
        use_f16: args.use_f16,
        seed: args.seed,
        guidance_scale: args.guidance_scale,
        height: args.height,
        width: args.width,
        num_samples: args.num_samples,
    };

    candle_diffusion::run(run_args)
}
