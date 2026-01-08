import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.distributions import Categorical
from typing import Tuple, List
from torch.amp import autocast, GradScaler

class PPOLM:
    """
    PPO for language generation (preserved from example, improved).
    - Uses GPT-2.
    - Clipped surrogate with rewards (e.g., length + keywords).
    - Simple advantage (reward - baseline).
    - Citation: @schulman2017proximal (adapted for LM)
    """
    def __init__(self, model_name: str = 'distilgpt2', lr: float = 1e-5, epsilon: float = 0.2,
                 use_amp: bool = True, temperature: float = 0.9, top_k: int = 50, top_p: float = 0.95,
                 do_sample: bool = True):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epsilon = epsilon
        self.use_amp = use_amp
        self.scaler = GradScaler(device=self.device, enabled=use_amp)
        self.value_head = nn.Linear(self.model.config.n_embd, 1).to(self.device)
        # sampling defaults (can be overridden per-call in generate)
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample

    def generate(self, inputs, max_new_tokens: int = 30, temperature: float = None,
                 top_k: int = None, top_p: float = None, do_sample: bool = None) -> Tuple[torch.Tensor, List[float], List[float]]:
        """Generate from the LM.
        `inputs` can be either an `input_ids` tensor or a dict from the tokenizer
        (containing `input_ids` and `attention_mask`). Returns (sequence, log_probs, values_for_generated).
        """
        self.model.eval()
        # Normalize inputs: accept tokenizer outputs dict or tensor
        if isinstance(inputs, dict):
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
        else:
            input_ids = inputs.to(self.device)
            attention_mask = None

        with torch.no_grad():
            # Use stochastic sampling with safe defaults; provide reasonable top-k/top-p/temperature
            # resolve sampling parameters: call-level override -> instance default
            if temperature is None:
                temperature = self.temperature
            if top_k is None:
                top_k = self.top_k
            if top_p is None:
                top_p = self.top_p
            if do_sample is None:
                do_sample = self.do_sample

            gen_kwargs = dict(
                do_sample=do_sample,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            # Prefer max_new_tokens when available in transformers, otherwise fall back to max_length
            try:
                gen_kwargs['max_new_tokens'] = max_new_tokens
            except Exception:
                gen_kwargs['max_length'] = input_ids.shape[1] + max_new_tokens

            if attention_mask is not None:
                gen_kwargs['attention_mask'] = attention_mask

            outputs = self.model.generate(input_ids, **gen_kwargs)

        sequence = outputs.sequences  # (batch, seq_len)
        # `outputs.scores` is a list of logits for each generated token (len = gen_len)
        scores = torch.stack(outputs.scores, dim=1)  # (batch, gen_len, vocab)

        # compute log-probs for generated tokens
        log_probs = torch.log_softmax(scores, dim=-1)
        inputs_len = input_ids.shape[1]
        actions = sequence[:, inputs_len:]
        gathered_lp = log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1).tolist()

        # If nothing was generated (edge cases), retry with beam search (deterministic)
        gen_len = sequence.shape[1] - inputs_len
        if gen_len == 0:
            try:
                beams = self.model.generate(input_ids, num_beams=5, max_new_tokens=max_new_tokens, early_stopping=True)
            except TypeError:
                beams = self.model.generate(input_ids, num_beams=5, max_length=input_ids.shape[1] + max_new_tokens, early_stopping=True)
            sequence = beams
            # recompute generated tokens
            inputs_len = input_ids.shape[1]

        # Get hidden states by running full forward pass (so we can compute values)
        # We run the model on the full sequence to obtain hidden states.
        # If attention_mask was provided for the prompt, build a full mask for the sequence
        try:
            if attention_mask is not None:
                # Construct a full attention mask: ones for generated tokens
                gen_mask = torch.ones((sequence.shape[0], sequence.shape[1] - attention_mask.shape[1]), device=self.device, dtype=attention_mask.dtype)
                full_mask = torch.cat([attention_mask, gen_mask], dim=1)
                hidden_outputs = self.model(sequence, output_hidden_states=True, attention_mask=full_mask)
            else:
                hidden_outputs = self.model(sequence, output_hidden_states=True)
        except Exception:
            hidden_outputs = self.model(sequence, output_hidden_states=True)

        hidden_states = hidden_outputs.hidden_states[-1]
        # compute a value per timestep by averaging token embeddings up to that position
        values = [self.value_head(hidden_states[b, :i+1].mean(0, keepdim=True)).item()
                  for b in range(sequence.shape[0])
                  for i in range(sequence.shape[1])]
        # values is flattened per batch; reshape into list-of-lists then return only generated portion
        batch_values = []
        seq_len = sequence.shape[1]
        for b in range(sequence.shape[0]):
            start = b * seq_len
            batch_values.append(values[start:start + seq_len])

        return sequence, gathered_lp, [bv[inputs_len:] for bv in batch_values]

    def compute_reward(self, text: str) -> float:
        """Example reward: length + keywords."""
        reward = min(len(text.split()) / 25, 1.0)
        if 'good' in text.lower() or 'great' in text.lower():
            reward += 0.5
        return reward

    def update(self, prompts: List[str], max_new_tokens: int = 30, gradient_accumulation_steps: int = 1) -> float:
        self.model.train()
        losses = []
        accum_step = 0

        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True).to(self.device)['input_ids']
        old_sequences, old_log_probs, old_values = self.generate(inputs, max_new_tokens)

        # Rewards and advantages
        texts = [self.tokenizer.decode(seq[inputs.shape[1]:], skip_special_tokens=True) for seq in old_sequences]
        rewards = [self.compute_reward(t) for t in texts]
        advantages = [r - old_values[-1] for r in rewards]  # Simple baseline: last value

        # New forward
        with autocast(enabled=self.use_amp):
            outputs = self.model(old_sequences, output_hidden_states=True)
            new_logits = outputs.logits[:, :-1, :]
            new_log_probs = torch.log_softmax(new_logits, dim=-1).gather(2, old_sequences[:, 1:].unsqueeze(-1)).squeeze(-1)

            # Pad old_log_probs to match new_log_probs shape (batch_size, seq_len)
            # old_log_probs has length = generated tokens; pad with 0.0 at the beginning (input tokens)
            max_seq_len = new_log_probs.shape[1]
            padded_old_lp = []
            for lp_list in old_log_probs:
                # Pad at the beginning with zeros for input tokens
                pad_len = max_seq_len - len(lp_list)
                padded = [0.0] * pad_len + lp_list
                padded_old_lp.append(padded[:max_seq_len])  # Truncate if needed
            old_lp_tensor = torch.tensor(padded_old_lp, dtype=torch.float32, device=self.device)

            ratio = torch.exp(new_log_probs - old_lp_tensor)
            adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device).unsqueeze(-1)
            surr1 = ratio * adv_tensor
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv_tensor
            policy_loss = -torch.min(surr1, surr2).mean()

            new_values = self.value_head(outputs.hidden_states[-1].mean(1))
            value_loss = nn.MSELoss()(new_values, torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1))

            loss = policy_loss + 0.5 * value_loss

        loss = loss / gradient_accumulation_steps
        self.scaler.scale(loss).backward()

        accum_step += 1
        if accum_step % gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        losses.append(loss.item())
        return sum(losses) / len(losses)