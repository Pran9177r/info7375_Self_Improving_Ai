import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional
# Assuming your rewards.py is importable
from tinyzero.rewards import compute_reward 
import math # For isnan

class APOTrainer:
    """A*PO Trainer - More Stable Advantage Weighting"""

    def __init__(self, policy_model, reference_model, config: Dict):
        self.policy = policy_model
        self.ref_model = reference_model
        self.config = config

        # APO hyperparams
        apo_cfg = config.get('apo', {})
        self.beta = apo_cfg.get('beta', 0.5)
        self.v_star_samples = apo_cfg.get('v_star_samples', 5) # Increased default
        self.learning_rate = apo_cfg.get('learning_rate', 5e-7) # Lowered default
        self.kl_coef = apo_cfg.get('kl_coef', 0.02)
        self.use_exp_weights = apo_cfg.get('use_exp_weights', False) # Default to Advantage Weighting
        self.adv_clip = apo_cfg.get('adv_clip', 3.0)
        self.clip_grad_norm = apo_cfg.get('clip_grad_norm', 1.0) # Added grad clipping value
        self.weighting_scheme = apo_cfg.get('weighting_scheme', 'normalized_advantage') # 'exp', 'normalized_advantage', 'shifted_advantage'
        self.log_intermediate_values = apo_cfg.get('log_intermediate_values', False) # Flag for detailed logging


        # Generation / tokenization lengths
        model_cfg = config.get('model', {})
        self.gen_max_length = model_cfg.get('max_length', 128)
        self.sft_max_length = min(
            model_cfg.get('sft_max_length', 256),
            getattr(self.policy.tokenizer, "model_max_length", 4096)
        )

        # Sampling controls
        samp = config.get('sampling', {})
        self.temperature = samp.get('temperature', 0.8)
        self.top_p = samp.get('top_p', 0.9)
        self.top_k = samp.get('top_k', 0)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.learning_rate
        )

        self.step = 0

    @torch.no_grad()
    def compute_V_star(self, prompts: List[str], problems: Optional[List[Dict]] = None) -> np.ndarray:
        """Compute V* (soft value) per prompt using the reference model."""
        V_star_values = []
        self.ref_model.eval() # Ensure ref model is in eval mode

        # --- Batching for V* computation (more efficient if v_star_samples is large) ---
        # Note: This basic batching assumes all prompts fit; adjust if needed
        all_samples_flat = []
        prompt_indices = [] # Keep track of which prompt each sample belongs to

        print(f"  Generating {self.v_star_samples} samples per prompt for V* ({len(prompts)} prompts)...")
        # Generate samples (potentially could batch prompts here if ref_model supports it well)
        for i, prompt in enumerate(prompts):
            samples_per_prompt = self.ref_model.generate(
                [prompt],
                num_samples=self.v_star_samples,
                temperature=1.0, # Usually sample V* at temp 1.0
                max_length=self.gen_max_length
            )[0]
            all_samples_flat.extend(samples_per_prompt)
            prompt_indices.extend([i] * len(samples_per_prompt))

        # Compute rewards for all samples
        print(f"  Computing rewards for {len(all_samples_flat)} V* samples...")
        all_rewards = []
        for idx, sample in enumerate(all_samples_flat):
            prompt_idx = prompt_indices[idx]
            problem = problems[prompt_idx] if problems else {'prompt': prompts[prompt_idx], 'task': 'unknown'}
            r = compute_reward(sample, problem, require_cot=False) # Don't require CoT for V* calculation
            all_rewards.append(r)
            if idx % 10 == 0:
                 print(f"    Reward computed for sample {idx+1}/{len(all_samples_flat)}...", end="\r")
        print() # Newline after reward computation

        all_rewards = np.array(all_rewards, dtype=np.float32)

        # Calculate V* per original prompt
        print("  Calculating V* values...")
        for i in range(len(prompts)):
            # Get rewards corresponding to this prompt
            indices = [k for k, p_idx in enumerate(prompt_indices) if p_idx == i]
            rewards = all_rewards[indices]

            if rewards.size == 0:
                V_star = 0.0
            else:
                if self.beta > 0:
                    # numerically stable softmax-style expectation
                    max_r = rewards.max()
                    exp_terms = np.exp((rewards - max_r) / self.beta)
                    V_star = float(max_r + self.beta * np.log(np.mean(exp_terms)))
                else: # beta=0 means V* = max reward observed
                    V_star = float(rewards.max())

            V_star_values.append(V_star)

        print("  V* computation complete.")
        return np.array(V_star_values, dtype=np.float32)


    def _build_concat_with_labels(self, prompt_ids: torch.Tensor, comp_ids: torch.Tensor, pad_id: int):
        """Construct input_ids and labels with prompt masking."""
        device = prompt_ids.device
        input_ids = torch.cat([prompt_ids, comp_ids], dim=1)
        attention_mask = (input_ids != pad_id).long()
        labels = input_ids.clone()
        labels[:] = -100 # Mask all initially

        prompt_lens = (prompt_ids != pad_id).sum(dim=1)
        B, T = labels.size()
        for i in range(B):
            start = int(prompt_lens[i].item())
            # Only unmask if start index is within sequence length
            if start < T:
                labels[i, start:] = input_ids[i, start:]

        return input_ids, attention_mask, labels

    def _per_example_ce_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute mean CE per example over unmasked tokens."""
        B, T, V = logits.shape
        # Shift logits and labels for next token prediction loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        flat_logits = shift_logits.view(-1, V)
        flat_labels = shift_labels.view(-1)

        # Calculate loss per token, ignore pad index
        token_losses = F.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=-100,
            reduction='none'
        ).view(B, T - 1) # Reshape back to [B, T-1]

        # Mask based on shifted labels
        token_mask = (shift_labels != -100).float()
        # Calculate mean loss per example
        per_ex_loss = (token_losses.sum(dim=1) / token_mask.sum(dim=1).clamp_min(1.0))

        # Handle cases where an example has no valid labels (e.g., prompt filled max_length)
        per_ex_loss = torch.nan_to_num(per_ex_loss, nan=0.0) # Replace NaN with 0 if no valid tokens

        return per_ex_loss # [B]


    def _compute_kl_loss(self, logits_pi: torch.Tensor, logits_ref: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """ Computes KL divergence per example only on completion tokens """
        # Shift logits and labels like in CE loss
        logits_pi_shifted = logits_pi[..., :-1, :].contiguous()
        logits_ref_shifted = logits_ref[..., :-1, :].contiguous()
        labels_shifted = labels[..., 1:].contiguous()

        logp_pi = F.log_softmax(logits_pi_shifted, dim=-1)
        logp_ref = F.log_softmax(logits_ref_shifted, dim=-1)

        # Mask for completion tokens (labels != -100)
        token_mask = (labels_shifted != -100).float()

        # KL divergence per token: sum_vocab p_pi * (logp_pi - logp_ref)
        # Use formula: E_{token ~ pi} [log p_pi(token) - log p_ref(token)]
        # We approximate this with the sampled token's contribution
        # Need predicted probabilities, not just logprobs at true labels
        kl_div_tokens = F.kl_div(logp_ref, logp_pi, log_target=True, reduction='none').sum(-1) # sum over vocab
        kl_div_tokens = kl_div_tokens * token_mask # Apply mask

        # Average KL per example over valid tokens
        kl_per_ex = kl_div_tokens.sum(dim=1) / token_mask.sum(dim=1).clamp_min(1.0)
        kl_per_ex = torch.nan_to_num(kl_per_ex, nan=0.0) # Handle NaN

        return kl_per_ex # [B]


    def train_step(self, batch: List[Dict]) -> tuple:
        """A*PO training step with stable advantage weighting"""
        prompts = [item['prompt'] for item in batch]
        device = self.policy.model.device

        try:
            # Step 1: Compute V*
            V_star_np = self.compute_V_star(prompts, problems=batch)
            V_star_t = torch.tensor(V_star_np, dtype=torch.float32, device=device)

            # Step 2: Generate from policy
            self.policy.train() # Ensure policy is in train mode for dropout etc. if used
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            generated_texts = self.policy.generate(
                prompts, max_length=self.gen_max_length, temperature=self.temperature,
                do_sample=True, top_p=self.top_p, top_k=self.top_k
            )

            # Step 3: Compute rewards
            rewards = [compute_reward(text, problem, require_cot=False) for text, problem in zip(generated_texts, batch)]
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)

            # Step 4: Compute advantages & weights (STABLE VERSION)
            advantages = rewards_t - V_star_t # [B]

            # Normalize advantages across the batch for stability
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-6 # Avoid division by zero
            adv_norm = (advantages - adv_mean) / adv_std
            adv_norm = adv_norm.clamp(-self.adv_clip, self.adv_clip).detach() # Clip and detach

            # --- CHOOSE WEIGHTING SCHEME ---
            if self.weighting_scheme == 'exp':
                # Original A* weighting (can be unstable)
                weights = torch.exp(advantages / (self.beta + 1e-8)).detach()
                # Normalize weights to mean 1.0 (helps stabilize LR)
                weights = weights / weights.mean().clamp_min(1e-6)
            elif self.weighting_scheme == 'shifted_advantage':
                # Shift normalized advantages to be non-negative (often more stable)
                weights = (adv_norm + self.adv_clip).detach() # Shifts range to [0, 2*adv_clip]
                # Optional: Normalize to mean 1? Might not be necessary if shifted.
                # weights = weights / weights.mean().clamp_min(1e-6)
            else: # Default: 'normalized_advantage'
                # Use clipped normalized advantages directly (Simplest, often stable)
                # Adding 1.0 shifts the center from 0 to 1, range becomes approx [-2, 4] if clip=3
                # Clamping > 0 avoids negative loss contributions if CE is always positive
                weights = (adv_norm + 1.0).clamp(min=0.0).detach() # Shift and ensure non-negative

            # --- Logging Intermediate Values ---
            if self.log_intermediate_values and self.step % self.config.get('logging', {}).get('log_every', 5) == 0:
                print("\n--- Intermediate Values ---")
                for i in range(len(prompts)):
                    print(f"  Ex {i}: Reward={rewards_t[i]:.3f}, V*={V_star_t[i]:.3f}, Adv={advantages[i]:.3f}, AdvNorm={adv_norm[i]:.3f}, Weight={weights[i]:.3f}")
                print(f"  Advantage Stats: Mean={adv_mean:.3f}, Std={adv_std:.3f}")
                print(f"  Weight Stats: Mean={weights.mean():.3f}, Std={weights.std():.3f}, Min={weights.min():.3f}, Max={weights.max():.3f}")
                print("-------------------------\n")


            # Step 5: Build teacher-forced training batch
            enc_prompts = self.policy.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.sft_max_length)
            # Tokenize completions *without* special tokens if they match prompt start/end issues
            enc_comps = self.policy.tokenizer(generated_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.sft_max_length, add_special_tokens=False) # Try disabling special tokens for completion
            enc_prompts = {k: v.to(device) for k, v in enc_prompts.items()}
            enc_comps = {k: v.to(device) for k, v in enc_comps.items()}

            pad_id = self.policy.tokenizer.pad_token_id or getattr(self.policy.tokenizer, "eos_token_id", 0)

            input_ids, attention_mask, labels = self._build_concat_with_labels(
                enc_prompts["input_ids"], enc_comps["input_ids"], pad_id
            )

            # Truncate to sft_max_length
            input_ids = input_ids[:, :self.sft_max_length]
            attention_mask = attention_mask[:, :self.sft_max_length]
            labels = labels[:, :self.sft_max_length]

            # Step 6: Forward pass (policy)
            outputs = self.policy.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None # Compute custom loss
            )
            policy_logits = outputs.logits
            per_ex_ce_loss = self._per_example_ce_loss(policy_logits, labels) # [B]

            # Step 7: KL divergence term (if enabled)
            kl_term = torch.zeros_like(per_ex_ce_loss) # [B]
            if self.kl_coef and self.kl_coef > 0.0 and hasattr(self.ref_model, "model"):
                with torch.no_grad():
                    self.ref_model.eval() # Ensure ref model is in eval mode
                    ref_outputs = self.ref_model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                ref_logits = ref_outputs.logits.detach()
                kl_per_ex = self._compute_kl_loss(policy_logits, ref_logits, labels) # [B]
                kl_term = self.kl_coef * kl_per_ex

                # Add KL per-example to loss
                per_ex_loss_with_kl = per_ex_ce_loss + kl_term
            else:
                per_ex_loss_with_kl = per_ex_ce_loss

            # Step 8: Weight per-example loss and reduce
            # Ensure weights don't have NaNs/Infs
            if torch.isnan(weights).any() or torch.isinf(weights).any():
                 print("Warning: NaN or Inf detected in weights, using uniform weights for this step.")
                 weights = torch.ones_like(weights)

            # Apply weights (already detached)
            weighted_losses = per_ex_loss_with_kl * weights
            loss = weighted_losses.mean() # Average over batch

            # Check for NaN/Inf loss BEFORE backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                print("Error: NaN or Inf loss detected before backward pass. Skipping step.")
                print(f"Loss: {loss.item()}, Weights: {weights}, PerExLoss: {per_ex_loss_with_kl}")
                # Potentially log more details here
                raise ValueError("NaN/Inf loss detected") # Stop training

            # Step 9: Backprop & update
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.clip_grad_norm)
            self.optimizer.step()

            if torch.cuda.is_available(): torch.cuda.empty_cache()

            # Metrics
            self.step += 1
            loss_value = float(loss.item())
            avg_reward = float(rewards_t.mean().item())
            avg_advantage = float(advantages.mean().item()) # Raw advantage
            avg_v_star = float(V_star_t.mean().item())
            avg_kl = float(kl_term.mean().item()) # Average KL penalty per example


            if self.step % self.config.get('logging', {}).get('log_every', 5) == 0:
                print(
                    f"Step {self.step}: "
                    f"Loss={loss_value:.4f} (CE={per_ex_ce_loss.mean().item():.4f}, KL={avg_kl:.4f}), "
                    f"Reward={avg_reward:.3f}, "
                    f"Advantage={avg_advantage:.3f}, "
                    f"V*={avg_v_star:.3f}"
                )

            stats = {
                'loss': loss_value,
                'avg_reward': avg_reward,
                'avg_advantage': avg_advantage,
                'avg_v_star': avg_v_star,
                'avg_kl_penalty': avg_kl,
                'adv_norm_mean': float(adv_norm.mean().item()), # Should be near 0
                'adv_norm_std': float(adv_norm.std().item()),   # Should be near 1 before clipping
                'weight_mean': float(weights.mean().item()),
                'weight_std': float(weights.std().item()),
            }

            return loss_value, stats

        except Exception as e:
            print(f"\n--- Error in train_step ---")
            print(f"Step: {self.step}")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {e}")
            import traceback
            traceback.print_exc() # Print full traceback
            print("---------------------------\n")

            if torch.cuda.is_available(): torch.cuda.empty_cache()
            # Return safe default values
            return 0.0, {
                'loss': 0.0, 'avg_reward': 0.0, 'avg_advantage': 0.0,
                'avg_v_star': 0.0, 'avg_kl_penalty': 0.0, 'adv_norm_mean': 0.0,
                'adv_norm_std': 0.0, 'weight_mean': 1.0, 'weight_std': 0.0,
            }