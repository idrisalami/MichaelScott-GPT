"""
Improved Inference Script for Michael Scott Transformer
========================================================
This script provides better text generation with temperature, top-k, and top-p sampling
to reduce repetitive outputs and improve quality.

Usage:
    python inference_improved.py
"""

import torch
import json
from pathlib import Path

import os
os.chdir('/Users/idrishouiralami/Documents/Coding/GPT')
import sys
sys.path.append('/Users/idrishouiralami/Documents/Coding/GPT')

from gpt_tokenizers.tiktoken import TiktokenTokenizer, VocabInfo
from utils.transformer import build_transformer


class MichaelScottBot:
    """Chatbot that generates Michael Scott-style responses"""
    
    def __init__(self, model_dir="./models", device=None, tok=TiktokenTokenizer):
        """
        Initialize the chatbot.
        
        Args:
            model_dir: Directory containing model weights and config
            device: Device to run on ('cuda', 'mps', or 'cpu')
        """
        self.model_dir = Path(model_dir)
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        print(f"🔧 Using device: {self.device}")
        
        # Load configuration
        config_path = self.model_dir / "config_tiktoken_2.json"
        with open(config_path) as f:
            self.config = json.load(f)
        
        print(f"📋 Loaded config: {self.config['d_model']}d, {self.config['n_layers']} layers")
        
        # Initialize tokenizer
        self.tok = tok(vocab=VocabInfo(
            pad_id=self.config["specials"]["PAD"],
            bos_id=self.config["specials"]["BOS"],
            eos_id=self.config["specials"]["EOS"],
            shift=self.config["specials"]["SHIFT"],
        ))
        
        # Build model
        self.model = build_transformer(
            src_vocab_size=self.config["vocab_size"],
            tgt_vocab_size=self.config["vocab_size"],
            src_seq_len=self.config["src_seq_len"],
            tgt_seq_len=self.config["tgt_seq_len"],
            d_model=self.config["d_model"],
            N=self.config["n_layers"],
            h=self.config["n_heads"],
            dropout=self.config["dropout"],
            d_ff=self.config["d_ff"],
        )
        
        # Load weights
        weights_path = self.model_dir / "weights_tiktoken_2.pt"
        checkpoint = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device).eval()
        
        print(f"✅ Model loaded from {weights_path}")
        
        # Extract constants
        self.PAD = self.config["specials"]["PAD"]
        self.BOS = self.config["specials"]["BOS"]
        self.EOS = self.config["specials"]["EOS"]
        self.MAX_SRC_LEN = self.config["max_src_len"]
        self.MAX_TGT_LEN = self.config["max_tgt_len"]
    
    def _pad_to(self, ids, length):
        """Pad or truncate sequence to specified length"""
        if len(ids) >= length:
            return ids[:length]
        return ids + [self.PAD] * (length - len(ids))
    
    def _enc_pad_mask(self, x):
        """Create encoder padding mask"""
        return (x != self.PAD).unsqueeze(1).unsqueeze(2)
    
    def _dec_mask_from(self, dec_in):
        """Create decoder mask (padding + causal)"""
        B, T = dec_in.size()
        tgt_pad = (dec_in != self.PAD).unsqueeze(1).unsqueeze(2)
        tgt_causal = torch.tril(
            torch.ones(T, T, dtype=torch.bool, device=dec_in.device)
        ).unsqueeze(0).unsqueeze(1)
        return tgt_pad & tgt_causal
    
    @torch.no_grad()
    def generate(
        self,
        context: str,
        max_new_tokens: int = 80,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        num_return_sequences: int = 1,
    ):
        """
        Generate Michael Scott-style response with advanced sampling.
        
        Args:
            context: Input dialogue context (e.g., "[JIM] Is Dwight the assistant regional manager?")
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random, lower = more focused)
                        Recommended: 0.7-1.0
            top_k: Keep only top k tokens with highest probability (0 = disabled)
                   Recommended: 40-100
            top_p: Nucleus sampling - keep tokens with cumulative probability >= top_p
                   Recommended: 0.85-0.95
            repetition_penalty: Penalty for repeating tokens (>1.0 = discourage repetition)
                               Recommended: 1.1-1.5
            num_return_sequences: Number of different responses to generate
        
        Returns:
            str or list[str]: Generated response(s)
        """
        self.model.eval()
        
        # Encode source
        src_ids = self._pad_to(self.tok.encode(context), self.MAX_SRC_LEN)
        src = torch.tensor(src_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        src_mask = self._enc_pad_mask(src)
        enc_out = self.model.encode(src, src_mask)
        
        # Generate multiple sequences if requested
        results = []
        
        for _ in range(num_return_sequences):
            # Start with BOS token
            dec = torch.tensor([[self.BOS]], dtype=torch.long, device=self.device)
            generated_tokens = []
            
            # Track token frequencies for repetition penalty
            token_counts = {}
            
            # Generate tokens
            for step in range(max_new_tokens):
                tgt_mask = self._dec_mask_from(dec)
                dec_out = self.model.decode(enc_out, src_mask, dec, tgt_mask)
                logits = self.model.project(dec_out)[:, -1, :]  # (1, vocab_size)
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for token_id, count in token_counts.items():
                        if token_id < logits.size(-1):
                            # Reduce probability of repeated tokens
                            logits[0, token_id] /= (repetition_penalty ** count)
                
                # Apply temperature
                logits = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
                    logits[indices_to_remove] = -float('inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Keep at least one token
                    sorted_indices_to_remove[..., 0] = False
                    
                    # Scatter back to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = -float('inf')
                
                # Sample from the filtered distribution
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Update token counts for repetition penalty
                token_id = next_token.item()
                token_counts[token_id] = token_counts.get(token_id, 0) + 1
                
                # Append to sequence
                dec = torch.cat([dec, next_token], dim=1)
                generated_tokens.append(token_id)
                
                # Stop if EOS token
                if token_id == self.EOS:
                    break
            
            # Decode the generated sequence
            response = self.tok.decode(dec[0].tolist())
            results.append(response)
        
        # Return single string or list based on num_return_sequences
        return results[0] if num_return_sequences == 1 else results
    
    def generate_greedy(self, context: str, max_new_tokens: int = 80):
        """
        Generate with greedy decoding (always pick most likely token).
        This is the old method - usually produces repetitive output.
        """
        return self.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            repetition_penalty=1.0,
        )
    
    def generate_creative(self, context: str, max_new_tokens: int = 80):
        """Generate with creative/diverse sampling"""
        return self.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=100,
            top_p=0.95,
            repetition_penalty=1.3,
        )
    
    def generate_balanced(self, context: str, max_new_tokens: int = 80):
        """Generate with balanced sampling (recommended default)"""
        return self.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
        )
    
    def generate_focused(self, context: str, max_new_tokens: int = 80):
        """Generate with focused/conservative sampling"""
        return self.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_k=30,
            top_p=0.85,
            repetition_penalty=1.1,
        )
    
    def chat(self):
        """Interactive chat mode"""
        print("\n" + "=" * 80)
        print("🎬 MICHAEL SCOTT CHATBOT")
        print("=" * 80)
        print("\nType your dialogue context and get Michael's response!")
        print("Examples:")
        print("  [JIM] Is Dwight the assistant regional manager?")
        print("  [PAM] Michael, we need to talk about the budget.")
        print("\nCommands:")
        print("  'quit' or 'exit' - Exit the chat")
        print("  'settings' - Adjust generation parameters")
        print("=" * 80 + "\n")
        
        # Default settings
        settings = {
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
            "max_tokens": 80,
        }
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit']:
                    print("\nThat's what she said! Goodbye! 👋")
                    break
                
                if user_input.lower() == 'settings':
                    print("\nCurrent settings:")
                    for key, value in settings.items():
                        print(f"  {key}: {value}")
                    print("\nEnter new values (or press Enter to keep current):")
                    
                    for key in settings:
                        new_value = input(f"  {key} [{settings[key]}]: ").strip()
                        if new_value:
                            try:
                                settings[key] = type(settings[key])(new_value)
                            except ValueError:
                                print(f"    Invalid value, keeping {settings[key]}")
                    continue
                
                # Generate response
                print("\nMichael: ", end="", flush=True)
                response = self.generate(
                    user_input,
                    max_new_tokens=settings["max_tokens"],
                    temperature=settings["temperature"],
                    top_k=settings["top_k"],
                    top_p=settings["top_p"],
                    repetition_penalty=settings["repetition_penalty"],
                )
                print(response + "\n")
                
            except KeyboardInterrupt:
                print("\n\nThat's what she said! Goodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


def main():
    """Main function for testing"""
    
    # Initialize bot
    bot = MichaelScottBot()
    
    # Test examples
    test_contexts = [
        "[JIM] Is Dwight the assistant regional manager?",
        "[PAM] Michael, we need to talk about the budget.",
        "[DWIGHT] I am the best salesman in this office.",
        "[RYAN] I have a new idea for the company.",
    ]
    
    print("\n" + "=" * 80)
    print("🎬 TESTING DIFFERENT SAMPLING STRATEGIES")
    print("=" * 80)
    
    for context in test_contexts:
        print(f"\n📝 Context: {context}")
        print("-" * 80)
        
        # Test different strategies
        print("\n🎯 Focused (conservative):")
        print(bot.generate_focused(context))
        
        print("\n⚖️  Balanced (recommended):")
        print(bot.generate_balanced(context))
        
        print("\n🎨 Creative (diverse):")
        print(bot.generate_creative(context))
        
        print("\n" + "=" * 80)
    
    # Start interactive chat
    print("\n\nStarting interactive chat mode...\n")
    bot.chat()


if __name__ == "__main__":
    main()

