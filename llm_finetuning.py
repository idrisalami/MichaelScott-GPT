import modal
import os

# Create a Modal app
app = modal.App("michael-scott-fine-tuning")

# Define the container image with all dependencies
# Use the official PyTorch image which has better compatibility
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel",
        add_python="3.10"
    )
    .apt_install("git")
    # Install packages separately to avoid dependency conflicts
    .pip_install("torch>=2.3.0")
    .pip_install(
        "xformers",
        "numpy",
        "tqdm",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "bitsandbytes",
    )
    # Install trl and unsloth last to avoid conflicts
    .pip_install("trl>=0.7.9,!=0.9.0,!=0.9.1,!=0.9.2,!=0.9.3")
    .pip_install("unsloth[cu121-torch230]")
    # Add local Python modules to the image (Modal best practice)
    .add_local_python_source("utils")
)

# Mount paths
VOLUME_PATH = "/vol"
CHECKPOINT_DIR = f"{VOLUME_PATH}/checkpoints_finetuning"
GGUF_MODEL_DIR = f"{VOLUME_PATH}/gguf_models"

# Create Modal volume for storing models
volume = modal.Volume.from_name("michael-scott-models", create_if_missing=True)

# Local data paths (for loading data on local machine)
SRC_PATH = "data/src.txt"
TGT_PATH = "data/tgt.txt"

@app.function(
    image=image,
    gpu="T4",  # Options: "T4", "A10G", "A100-40GB", "A100-80GB", "H100"
    volumes={VOLUME_PATH: volume},
    timeout=3600 * 6,  # 6 hours timeout
)
def LLM_finetuning(
    model_name: str,
    max_seq_length: int,
    src_lines: list[str],
    tgt_lines: list[str],
    dtype=None,
    load_in_4bit=True,
):
    # Fixed: Import inside Modal function for proper serialization
    from unsloth import FastLanguageModel
    from datasets import Dataset  # Fixed: Missing import
    import torch
    import os

    os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"  # avoid torchvision import
    
    # Fixed: Check GPU availability and print device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Warning: No GPU detected, using CPU")
    
    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.float16,
        load_in_4bit=load_in_4bit,
    )

    if getattr(model.config, "torch_dtype", None) is None:
        model.config.torch_dtype = torch.float16

    print("Model loaded successfully!")

    def format_prompt(input_text, output_text):
        return f"### Input: {input_text}\n### Output: {output_text}<|endoftext|>"
    
    print(f"Formatting {len(src_lines)} training examples...")
    formatted_data = [format_prompt(src, tgt) for src, tgt in zip(src_lines, tgt_lines)]
    dataset = Dataset.from_dict({"text": formatted_data})
    print(f"Dataset created with {len(dataset)} examples")

    # Add LoRA adapters
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,  # LoRA rank - higher = more capacity, more memory
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=128,  # LoRA scaling factor (usually 2x rank)
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",     # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized version
        random_state=3407,
        use_rslora=False,  # Rank stabilized LoRA
        loftq_config=None, # LoftQ
    )
    print("LoRA adapters added successfully!")

    from trl import SFTTrainer
    from transformers import TrainingArguments

    # Training arguments optimized for Unsloth
    print("Setting up trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,  # Effective batch size = 8
            warmup_steps=10,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=25,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            save_strategy="epoch",
            save_total_limit=2,
            dataloader_pin_memory=False,
            report_to="none", # Disable Weights & Biases logging
        ),
    )

    # Train the model
    print("Starting training...")
    print("=" * 60)
    trainer_stats = trainer.train()
    print("=" * 60)
    print("Training completed!")

    # Save GGUF model to Modal volume
    print("\nSaving model to GGUF format...")
    gguf_save_path = os.path.join(GGUF_MODEL_DIR, "michael_scott_model")
    
    # Fixed: Ensure parent directory exists before saving
    try:
        os.makedirs(gguf_save_path, exist_ok=True)
        print(f"Created directory: {gguf_save_path}")
    except Exception as e:
        print(f"Warning: Could not create directory {gguf_save_path}: {e}")
        print("Saving to volume root instead...")
        gguf_save_path = VOLUME_PATH
    
    # Save the model in GGUF format
    model.save_pretrained_gguf(gguf_save_path, tokenizer, quantization_method="q4_k_m")
    print(f"Model saved to: {gguf_save_path}")
    
    # Find the generated GGUF file
    gguf_files = []
    try:
        for root, dirs, files in os.walk(gguf_save_path):
            for file in files:
                if file.endswith(".gguf"):
                    gguf_files.append(os.path.join(root, file))
    except Exception as e:
        print(f"Error scanning for GGUF files: {e}")
    
    if gguf_files:
        print(f"\nFound {len(gguf_files)} GGUF file(s):")
        for gguf_file in gguf_files:
            file_size = os.path.getsize(gguf_file) / (1024 ** 3)  # Size in GB
            print(f"  - {gguf_file} ({file_size:.2f} GB)")
        
        # Fixed: Commit the volume to persist the model (Modal best practice)
        print("\nCommitting volume to persist model...")
        volume.commit()
        print("✓ Volume committed successfully!")
        
        return {
            "training_stats": trainer_stats,
            "gguf_file_path": gguf_files[0],
            "all_gguf_files": gguf_files,
            "model_saved": True
        }
    else:
        print("✗ No GGUF file found after saving!")
        # Still commit the volume in case other files were saved
        volume.commit()
        return {
            "training_stats": trainer_stats,
            "gguf_file_path": None,
            "all_gguf_files": [],
            "model_saved": False
        }

# Function to download the trained model from Modal volume
@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
)
def download_model_from_volume():
    """Download the GGUF model from Modal volume to local machine"""
    import os
    
    # List files in the GGUF model directory
    if os.path.exists(GGUF_MODEL_DIR):
        files = os.listdir(GGUF_MODEL_DIR)
        print(f"Files in {GGUF_MODEL_DIR}: {files}")
        
        # Find GGUF files
        gguf_files = []
        for root, dirs, filenames in os.walk(GGUF_MODEL_DIR):
            for filename in filenames:
                if filename.endswith('.gguf'):
                    gguf_files.append(os.path.join(root, filename))
        
        if gguf_files:
            print(f"Found {len(gguf_files)} GGUF file(s):")
            for gguf_file in gguf_files:
                print(f"  - {gguf_file}")
            return gguf_files
        else:
            print("No GGUF files found in volume")
            return []
    else:
        print(f"Directory {GGUF_MODEL_DIR} does not exist")
        return []

# Fixed: Use @app.local_entrypoint for the main function (Modal best practice)
@app.local_entrypoint()
def main(
    model_name: str = "unsloth/llama-2-7b-chat-bnb-4bit",
    max_seq_length: int = 2048,
    src_path: str = SRC_PATH,
    tgt_path: str = TGT_PATH,
):
    """
    Main entrypoint for fine-tuning LLM on Modal.
    
    Usage:
        modal run llm_finetuning.py
        modal run llm_finetuning.py --model-name "unsloth/llama-3-8b-bnb-4bit"
    """
    # Fixed: Import inside local function (only runs locally)
    from utils.data_loader import get_local_data
    
    print("=" * 80)
    print("LLM Fine-Tuning with Modal & Unsloth")
    print("=" * 80)
    
    # Load data locally (Modal best practice: load data on local machine, pass to remote)
    print(f"\nLoading training data from local files...")
    print(f"  Source: {src_path}")
    print(f"  Target: {tgt_path}")
    
    try:
        src_lines, tgt_lines = get_local_data(src_path, tgt_path)
        print(f"✓ Loaded {len(src_lines)} training examples")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Run fine-tuning on Modal (remote GPU execution)
    print(f"\nStarting fine-tuning on Modal...")
    print(f"  Model: {model_name}")
    print(f"  Max sequence length: {max_seq_length}")
    print(f"  Training examples: {len(src_lines)}")
    print("\nThis will run on a remote GPU. Progress will be shown below:")
    print("-" * 80)
    
    result = LLM_finetuning.remote(
        model_name=model_name,
        max_seq_length=max_seq_length,
        src_lines=src_lines,
        tgt_lines=tgt_lines
    )
    
    print("-" * 80)
    print("\n✓ Fine-tuning completed!")
    print(f"  Model saved: {result['model_saved']}")
    if result['model_saved']:
        print(f"  GGUF file: {result['gguf_file_path']}")
        if result.get('all_gguf_files'):
            print(f"  Total GGUF files: {len(result['all_gguf_files'])}")
    
    # Show how to download the model
    print("\n" + "=" * 80)
    print("To download the trained model to your local machine:")
    print("=" * 80)
    print("\nOption 1: Use Modal CLI")
    if result['model_saved'] and result.get('gguf_file_path'):
        remote_path = result['gguf_file_path']
        print(f"  modal volume get michael-scott-models {remote_path} ./local_model.gguf")
    print("\nOption 2: List all files in the volume")
    print(f"  modal volume ls michael-scott-models")
    print("\nOption 3: Download entire directory")
    print(f"  modal volume get michael-scott-models {GGUF_MODEL_DIR} ./downloaded_models/")
    
    print("\n" + "=" * 80)
    print("Fine-tuning complete! 🎉")
    print("=" * 80)