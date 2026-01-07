import torch
import torch.nn as nn
from typing import Optional, List, Union
from training.train import TrainConfig

class MichaelScottBot:
    """Flexible text generation wrapper for trained models"""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        masks,
        device: str = "cpu",
        config: Optional[TrainConfig] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.masks = masks
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        
        # Extract config values
        self.pad_id = config.pad_id if config else 0
        self.bos_id = config.bos_id if hasattr(config, 'bos_id') else 1
        self.eos_id = config.eos_id if hasattr(config, 'eos_id') else 2
        self.max_src_len = config.max_src_len if hasattr(config, 'max_src_len') else 256
    
    def _pad_to(self, ids: List[int], length: int) -> List[int]:
        """Pad or truncate sequence"""
        if len(ids) >= length:
            return ids[:length]
        return ids + [self.pad_id] * (length - len(ids))
    
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
    ) -> Union[str, List[str]]:
        """Generate text with flexible sampling parameters"""
        
        # Encode source
        src_ids = self._pad_to(self.tokenizer.encode(context), self.max_src_len)
        src = torch.tensor(src_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        src_mask = self.masks.encoder(src)
        enc_out = self.model.encode(src, src_mask)
        
        results = []
        
        for _ in range(num_return_sequences):
            dec = torch.tensor([[self.bos_id]], dtype=torch.long, device=self.device)
            token_counts = {}
            
            for step in range(max_new_tokens):
                _, _, tgt_mask = self.masks.decoder(dec)
                dec_out = self.model.decode(enc_out, src_mask, dec, tgt_mask)
                logits = self.model.project(dec_out)[:, -1, :]
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for token_id, count in token_counts.items():
                        if token_id < logits.size(-1):
                            logits[0, token_id] /= (repetition_penalty ** count)
                
                # Apply temperature
                logits = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
                    logits[indices_to_remove] = -float('inf')
                
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('inf')
                
                # Sample
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                token_id = next_token.item()
                token_counts[token_id] = token_counts.get(token_id, 0) + 1
                
                dec = torch.cat([dec, next_token], dim=1)
                
                if token_id == self.eos_id:
                    break
            
            response = self.tokenizer.decode(dec[0].tolist())
            results.append(response)
        
        return results[0] if num_return_sequences == 1 else results
    
    # Convenience methods
    def greedy(self, context: str, max_new_tokens: int = 80) -> str:
        """Greedy generation (like your current code)"""
        return self.generate(context, max_new_tokens, temperature=1.0, top_k=1, top_p=1.0)
    
    def creative(self, context: str, max_new_tokens: int = 80) -> str:
        """Creative generation"""
        return self.generate(context, max_new_tokens, temperature=1.0, top_k=100, top_p=0.95, repetition_penalty=1.3)
    
    def balanced(self, context: str, max_new_tokens: int = 80) -> str:
        """Balanced generation"""
        return self.generate(context, max_new_tokens, temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.2)
    
    def chat(self):
        """Interactive chat mode"""
        print("🎬 Michael Scott Chatbot - Type 'quit' to exit")
        while True:
            context = input("\nYou: ").strip()
            if context.lower() in ['quit', 'exit']:
                break
            response = self.balanced(context)
            print(f"Michael: {response}")