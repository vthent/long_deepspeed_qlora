import math
import warnings
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)

import transformers
from transformers import logging
logger = logging.get_logger(__name__)


class MLPWrapper(nn.Module):
    """Universal MLP wrapper for both Llama and Qwen3 models"""
    def __init__(
        self,
        module,
        mini_s: int = 8,
        chunk_size: int = 4096,
        chunk_mode: bool = True
    ):
        super().__init__()
        self.module = module
        self.mini_s = mini_s
        self.chunk_size = chunk_size
        self.chunk_mode = chunk_mode
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = x.size()

        # Determine chunk size
        if self.chunk_mode:
            chunk_size = min(max(self.chunk_size // max(bsz, 1), 1024), q_len)
        else:
            chunk_size = max(math.ceil(q_len / max(self.mini_s, 1)), 1)
            chunk_size = min(chunk_size, q_len)
        
        # No need to chunk if chunk size is larger than sequence length
        if chunk_size >= q_len:
            return self.module(x)
            
        # Process chunks
        chunks = x.split(chunk_size, dim=1)
        outputs = []
        
        for chunk in chunks:
            outputs.append(self.module(chunk))
            
        return torch.cat(outputs, dim=1)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)

#v0
'''
class _LMHeadFunction(torch.autograd.Function):
    """
    Custom autograd function for LM head with a corrected and stable
    backward pass for accurate gradient computation.
    """
    @staticmethod
    def forward(ctx, hidden_states, labels, weight):
        # Flatten inputs for cross_entropy which expects 2D logits and 1D labels
        logits_flat = F.linear(hidden_states.view(-1, hidden_states.size(-1)), weight).float()
        labels_flat = labels.view(-1)
        
        # Calculate loss using the built-in, optimized CrossEntropyLoss
        loss = F.cross_entropy(logits_flat, labels_flat, reduction="sum", ignore_index=-100)
        
        # Increment usage counter for tracking mini-sequences
        if hasattr(weight, 'count'):
            weight.count += 1
        else:
            weight.count = 1
        
        # Save necessary tensors for the backward pass
        ctx.save_for_backward(hidden_states, labels, weight)
        
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, labels, weight = ctx.saved_tensors
        
        if hasattr(weight, 'count'):
            weight.count -= 1

        # Reshape for calculation
        hidden_states_flat = hidden_states.view(-1, hidden_states.size(-1))
        labels_flat = labels.view(-1)
        
        # Recompute logits to save memory (activation recomputation)
        logits_flat = F.linear(hidden_states_flat, weight).float()
        
        # CORRECT GRADIENT CALCULATION:
        # 1. Compute softmax probabilities
        grad_logits = F.softmax(logits_flat, dim=-1)
        
        # 2. Use scatter_ to subtract 1 at the location of the correct labels (p - y)
        # This is a robust way to handle the one-hot subtraction
        ignore_index = -100
        valid_labels_mask = labels_flat != ignore_index
        
        if valid_labels_mask.any():
             # Get the indices of the valid labels
            valid_indices = labels_flat[valid_labels_mask].unsqueeze(1)
            # Create a tensor of -1s to subtract
            subtraction_tensor = torch.full_like(valid_indices, -1, dtype=grad_logits.dtype)
            # Apply subtraction only to the rows with valid labels
            grad_logits[valid_labels_mask] = grad_logits[valid_labels_mask].scatter_add(1, valid_indices, subtraction_tensor)
        
        # Scale by the incoming gradient
        grad_logits *= grad_output
        grad_logits = grad_logits.to(hidden_states.dtype)

        # Gradient for hidden_states
        grad_hidden_states = (grad_logits @ weight).view_as(hidden_states)
        
        # Gradient for weights (accumulated)
        grad_weight_chunk = grad_logits.T @ hidden_states_flat
        if not hasattr(weight, 'grad') or weight.grad is None:
            weight.grad = grad_weight_chunk
        else:
            weight.grad.add_(grad_weight_chunk)

        # Only return the final accumulated weight gradient on the last chunk
        if hasattr(weight, 'count') and weight.count == 0:
            return grad_hidden_states, None, weight.grad
        else:
            return grad_hidden_states, None, None
'''

#v1
'''
class _LMHeadFunction(torch.autograd.Function):
    """
    Custom autograd function for LM head with corrected gradient computation.
    """
    @staticmethod
    def forward(ctx, hidden_states, labels, weight):
        # Flatten inputs for cross_entropy
        logits_flat = F.linear(hidden_states.view(-1, hidden_states.size(-1)), weight).float()
        labels_flat = labels.view(-1)
        
        # Calculate loss using cross-entropy with sum reduction
        loss = F.cross_entropy(logits_flat, labels_flat, reduction="sum", ignore_index=-100)
        
        # Increment usage counter
        if hasattr(weight, 'count'):
            weight.count += 1
        else:
            weight.count = 1
        
        # Save for backward
        ctx.save_for_backward(hidden_states, labels, weight)
        
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, labels, weight = ctx.saved_tensors
        
        if hasattr(weight, 'count'):
            weight.count -= 1

        # Reshape for calculation
        hidden_states_flat = hidden_states.view(-1, hidden_states.size(-1))
        labels_flat = labels.view(-1)
        
        # Recompute logits
        logits_flat = F.linear(hidden_states_flat, weight).float()
        
        # CORRECTED GRADIENT CALCULATION:
        # Compute softmax probabilities
        grad_logits = F.softmax(logits_flat, dim=-1)
        
        # Create mask for valid labels (not -100)
        valid_mask = labels_flat != -100
        
        # Subtract 1 from the probability at the correct label position
        # This implements the derivative: p_i - y_i
        if valid_mask.any():
            # Use advanced indexing to subtract 1 at correct positions
            batch_indices = torch.arange(len(labels_flat), device=labels_flat.device)[valid_mask]
            label_indices = labels_flat[valid_mask]
            grad_logits[batch_indices, label_indices] -= 1.0
        
        # Scale by the incoming gradient (from the loss)
        grad_logits = grad_logits * grad_output
        grad_logits = grad_logits.to(hidden_states.dtype)

        # Gradient for hidden_states
        grad_hidden_states = (grad_logits @ weight).view_as(hidden_states)
        
        # Gradient for weights (accumulated)
        grad_weight_chunk = grad_logits.T @ hidden_states_flat
        
        # Accumulate weight gradients
        if not hasattr(weight, 'grad') or weight.grad is None:
            weight.grad = grad_weight_chunk
        else:
            weight.grad = weight.grad + grad_weight_chunk

        # Return gradients (only return accumulated weight grad on last chunk)
        if hasattr(weight, 'count') and weight.count == 0:
            return grad_hidden_states, None, weight.grad
        else:
            return grad_hidden_states, None, None
'''

#v2

class _LMHeadFunction(torch.autograd.Function):
    """
    Stable backward for LM head: correct masking + return grads instead of mutating weight.grad
    """
    @staticmethod
    def forward(ctx, hidden_states, labels, weight):
        # [B, T, H] -> [B*T, H]
        hs_flat = hidden_states.view(-1, hidden_states.size(-1))
        logits_flat = F.linear(hs_flat, weight).float()  # [B*T, V]
        labels_flat = labels.view(-1)                    # [B*T]

        loss = F.cross_entropy(
            logits_flat, labels_flat,
            reduction="sum", ignore_index=-100
        )

        # Save tensors for backward
        ctx.save_for_backward(hidden_states, labels, weight)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, labels, weight = ctx.saved_tensors

        hs_flat = hidden_states.view(-1, hidden_states.size(-1))  # [N, H]
        labels_flat = labels.view(-1)                              # [N]
        logits_flat = F.linear(hs_flat, weight).float()            # [N, V]

        probs = F.softmax(logits_flat, dim=-1)                     # [N, V]

        ignore_index = -100
        valid_mask = (labels_flat != ignore_index)                 # [N]
        if valid_mask.any():
            # (p - y) only on valid rows; zero elsewhere
            grad_logits = torch.zeros_like(probs)
            grad_logits[valid_mask] = probs[valid_mask]
            grad_logits[valid_mask, labels_flat[valid_mask]] -= 1.0
        else:
            grad_logits = torch.zeros_like(probs)

        # Scale by upstream grad (a scalar for summed loss)
        grad_logits = (grad_logits * grad_output).to(hidden_states.dtype)

        # dL/dH = grad_logits @ W
        grad_hidden = (grad_logits @ weight).view_as(hidden_states)
        # dL/dW = grad_logits^T @ H
        grad_weight = grad_logits.T @ hs_flat

        # labels has no gradient
        return grad_hidden, None, grad_weight


class LMHeadWrapper(nn.Module):
    """Wrapper for LM head with custom gradient computation"""
    def __init__(self, original_weight):
        super().__init__()
        self.weight = nn.Parameter(original_weight.clone() if isinstance(original_weight, torch.Tensor) else original_weight.weight.clone())
        self.weight.count = 0
        self.lm_head_fn = _LMHeadFunction.apply
        
    def forward(self, hidden_states, labels):
        return self.lm_head_fn(hidden_states, labels, self.weight)

class CausalLMWrapperBase(nn.Module):
    """Base class for CausalLM wrappers supporting both Llama and Qwen3"""
    def __init__(self, module, mini_s: int = 64):
        super().__init__()
        self.model = module.model
        self.config = module.config
        self.vocab_size = module.vocab_size
        self.mini_s = mini_s
        self.original_lm_head = module.lm_head
        self.lm_head_wrapper = LMHeadWrapper(module.lm_head.weight)
        
        # Store original methods if available
        if hasattr(module, 'generate'):
            self.original_generate = module.generate
        
    def mini_batch_processing(self, hidden_states, labels):
        
        bsz, q_len, hidden_size = hidden_states.size()
        
        if labels is None:
            logits = F.linear(hidden_states[:, -1:, :], self.original_lm_head.weight)
            # Return logits, a placeholder loss sum, and a token count of 1
            return logits.float(), torch.tensor(0.0, device=hidden_states.device), 1.0

        hidden_states = hidden_states[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        
        total_loss = 0.0
        total_valid_tokens = 0
        num_chunks = min(self.mini_s, hidden_states.size(1))
        
        if num_chunks > 0:
            chunks_hidden = torch.tensor_split(hidden_states, num_chunks, dim=1)
            chunks_labels = torch.tensor_split(labels, num_chunks, dim=1)

            for chunk_hidden, chunk_labels in zip(chunks_hidden, chunks_labels):
                if chunk_hidden.numel() == 0: continue
                chunk_loss = self.lm_head_wrapper(chunk_hidden, chunk_labels)
                valid_tokens = (chunk_labels != -100).sum()
                if valid_tokens > 0:
                    total_loss += chunk_loss
                    total_valid_tokens += valid_tokens
           
        # KEY CHANGE: Return all three values separately
        if total_valid_tokens > 0:
            return None, total_loss, total_valid_tokens
        else:
            return None, torch.tensor(0.0, device=hidden_states.device, requires_grad=True), 1.0


    def forward(self, input_ids: torch.LongTensor = None, labels: Optional[torch.LongTensor] = None, return_dict: Optional[bool] = None, **kwargs):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        model_kwargs = kwargs
        model_kwargs['input_ids'] = input_ids
        model_kwargs['return_dict'] = return_dict
        
        outputs = self.model(**model_kwargs)
        hidden_states = outputs[0]

        logits, summed_loss, num_tokens = self.mini_batch_processing(hidden_states, labels)

        # Create the standard output object
        final_outputs = CausalLMOutputWithPast(
            loss=summed_loss, # Put the SUMMED loss here, which DeepSpeed can handle
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        # --- THIS IS THE KEY CHANGE ---
        # Attach the token count as a new attribute that DeepSpeed will ignore
        final_outputs.num_tokens = num_tokens

        # The Trainer will now handle both tuple and dict-like outputs correctly
        if not return_dict:
            return (final_outputs.loss,) + (final_outputs.logits,) + tuple(v for v in (final_outputs.past_key_values, final_outputs.hidden_states, final_outputs.attentions) if v is not None)

        return final_outputs

    def generate(self, *args, **kwargs):
        """Forward to original generate method if available"""
        if hasattr(self, 'original_generate'):
            return self.original_generate(*args, **kwargs)
        return super().generate(*args, **kwargs)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # Get model state dict
        state_dict = self.model.state_dict(destination, prefix, keep_vars)
        
        # Add lm_head parameters
        lm_head_state = self.original_lm_head.state_dict()
        for k, v in lm_head_state.items():
            state_dict[prefix + 'lm_head.' + k] = v
            
        return state_dict
        
    def save_pretrained(self, *args, **kwargs):
        # Save the original model structure
        self.model.save_pretrained(*args, **kwargs)

class MiniSequence(nn.Module):
    """Main wrapper class supporting both Llama and Qwen3 models"""
    def __init__(self, module, model_type: str = "auto", mini_s: int = 64):
        super().__init__()
        self.module = module
        self.mini_s = mini_s
        
        # Auto-detect model type if not specified
        if model_type == "auto":
            self.model_type = self._detect_model_type(module)
        else:
            self.model_type = model_type.lower()
            
        logger.info(f"Initializing MiniSequence with model_type: {self.model_type}")
        
        # Apply recursive wrapping
        self._recursive_visit('module', self.module, self)
        
        # Enable gradient checkpointing
        if hasattr(self.module, 'gradient_checkpointing_enable'):
            self.gradient_checkpointing_enable()

    def _detect_model_type(self, module):
        """Auto-detect model type from module structure"""
        module_str = str(type(module))
        
        if 'llama' in module_str.lower():
            return 'llama'
        elif 'qwen3' in module_str.lower() or 'qwen' in module_str.lower():
            return 'qwen3'
        else:
            # Check for specific model attributes
            if hasattr(module, 'model') and hasattr(module.model, 'layers'):
                layer_type = str(type(module.model.layers[0]))
                if 'llama' in layer_type.lower():
                    return 'llama'
                elif 'qwen' in layer_type.lower():
                    return 'qwen3'
        
        logger.warning("Could not auto-detect model type, defaulting to 'llama'")
        return 'llama'

    def _recursive_visit(self, name, module, parent_module):
        """Recursively visit and wrap appropriate modules"""
        
        # Check for Llama modules
        if self.model_type == 'llama':
            try:
                from transformers.models.llama.modeling_llama import LlamaMLP, LlamaForCausalLM
                
                if isinstance(module, LlamaMLP):
                    logger.info(f"Wrapping LlamaMLP: {name}")
                    wrapped = MLPWrapper(module, mini_s=self.mini_s)
                    setattr(parent_module, name, wrapped)
                    return
                    
                if isinstance(module, LlamaForCausalLM):
                    logger.info(f"Wrapping LlamaForCausalLM: {name}")
                    wrapped = CausalLMWrapperBase(module, mini_s=self.mini_s)
                    setattr(parent_module, name, wrapped)
                    return
                    
            except ImportError:
                logger.warning("Could not import Llama modules")
        
        # Check for Qwen3 modules
        elif self.model_type == 'qwen3':
            try:
                from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3ForCausalLM
                
                if isinstance(module, Qwen3MLP):
                    logger.info(f"Wrapping Qwen3MLP: {name}")
                    wrapped = MLPWrapper(module, mini_s=self.mini_s)
                    setattr(parent_module, name, wrapped)
                    return
                    
                if isinstance(module, Qwen3ForCausalLM):
                    logger.info(f"Wrapping Qwen3ForCausalLM: {name}")
                    wrapped = CausalLMWrapperBase(module, mini_s=self.mini_s)
                    setattr(parent_module, name, wrapped)
                    return
                    
            except ImportError:
                logger.warning("Could not import Qwen3 modules")
                
                # Fallback to string matching
                class_name = module.__class__.__name__
                if 'MLP' in class_name and hasattr(module, 'forward'):
                    logger.info(f"Wrapping MLP module by name: {name}")
                    wrapped = MLPWrapper(module, mini_s=self.mini_s)
                    setattr(parent_module, name, wrapped)
                    return
                    
                if 'ForCausalLM' in class_name and hasattr(module, 'model'):
                    logger.info(f"Wrapping CausalLM module by name: {name}")
                    wrapped = CausalLMWrapperBase(module, mini_s=self.mini_s)
                    setattr(parent_module, name, wrapped)
                    return
        
        # Skip if module is a wrapper already
        if isinstance(module, (MLPWrapper, CausalLMWrapperBase)):
            return
            
        # Recurse into child modules
        for child_name, child_module in module.named_children():
            if child_module is not module:  # Avoid infinite recursion
                self._recursive_visit(child_name, child_module, module)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.module, 'gradient_checkpointing_enable'):
            self.module.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def generate(self, *args, **kwargs):
        if hasattr(self.module, 'generate'):
            return self.module.generate(*args, **kwargs)
        else:
            raise AttributeError("The wrapped module does not have a generate method")

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)

    def save_pretrained(self, *args, **kwargs):
        if hasattr(self.module, 'save_pretrained'):
            self.module.save_pretrained(*args, **kwargs)
        else:
            raise AttributeError("The wrapped module does not have a save_pretrained method")
    
    def __getattr__(self, name):
        """Forward attribute access to wrapped module if not found in MiniSequence"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if hasattr(self.module, name):
                return getattr(self.module, name)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # Add or override the method to forward it to the original model
    def prepare_inputs_for_generation(self, *args, **kwargs):
        if hasattr(self.module, 'prepare_inputs_for_generation'):
            return self.module.prepare_inputs_for_generation(*args, **kwargs)
        else:
            raise AttributeError("The underlying model does not have 'prepare_inputs_for_generation'")

#display loss info
class EnhancedLossLoggingCallback(TrainerCallback):
    """
    - Mirrors your true per-token loss into logs["loss_true_mean"] (and optionally overrides logs["loss"])
    - Ensures learning_rate is present (pulls from optimizer if HF didn't add it)
    - Keeps/forwards grad_norm if HF added it (safe for FSDP/DeepSpeed)
    - Optional fallback to compute a local grad norm if HF didn't provide one
    """
    def __init__(self, override_console_loss=True, fallback_grad_norm=False):
        self.override_console_loss = override_console_loss
        self.fallback_grad_norm = fallback_grad_norm
        self._last_lr = None
        self._last_grad_norm = None

    # Grab LR each optimizer step (works for any optimizer / scheduler setup)
    def on_optimizer_step(self, args, state, control, **kwargs):
        opt = kwargs.get("optimizer", None)
        if opt is not None and len(opt.param_groups) > 0:
            lrs = [pg.get("lr", None) for pg in opt.param_groups if "lr" in pg]
            try:
                self._last_lr = float(sum(lrs) / len(lrs))
            except Exception:
                self._last_lr = None

    # Optionally compute a *local* grad norm only if HF didn't log one
    def _compute_local_grad_norm(self, model):
        try:
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not grads:
                return None
            # NOTE: this is LOCAL norm; with ZeRO-3/FSDP sharding it's not the global norm.
            stacked = torch.stack([g.detach().data.float().norm(2) for g in grads])
            return float(stacked.norm(2).item())
        except Exception:
            return None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        # 1) True per-token loss from your Trainer.compute_loss
        if "loss_mean_dbg" in logs:
            logs["loss_true_mean"] = logs["loss_mean_dbg"]
            if self.override_console_loss:
                logs["loss"] = logs["loss_mean_dbg"]  # makes console show the real mean

        # 2) Learning rate
        if "learning_rate" not in logs and self._last_lr is not None:
            logs["learning_rate"] = self._last_lr

        # 3) Grad norm
        if "grad_norm" in logs:
            self._last_grad_norm = logs["grad_norm"]  # forward the one HF computed (global, clip-aware)
        else:
            # Optional fallback (local only; not recommended with ZeRO-3/FSDP)
            if self.fallback_grad_norm and "model" in kwargs and kwargs["model"] is not None:
                gn = self._compute_local_grad_norm(kwargs["model"])
                if gn is not None:
                    logs["grad_norm"] = gn
                    self._last_grad_norm = gn
            elif self._last_grad_norm is not None:
                # carry forward the last seen value so dashboards stay continuous
                logs["grad_norm"] = self._last_grad_norm

#display grad_norm info
class ClipAuditCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        # Add max_grad_norm so it's visible
        if getattr(args, "max_grad_norm", None) is not None:
            logs["max_grad_norm"] = float(args.max_grad_norm)

        # If HF/Accelerate provided grad_norm (pre-clip), derive clip info
        gn = logs.get("grad_norm", None)
        mg = getattr(args, "max_grad_norm", None)
        if gn is not None and mg is not None and mg > 0:
            clipped = gn > mg
            logs["grad_clipped"] = bool(clipped)
            # factor applied to grads to satisfy the threshold (1.0 means no clipping)
            logs["grad_clip_factor"] = float(min(1.0, mg / max(gn, 1e-12)))

        # If you also log the true per-token loss elsewhere, keep it:
        if "loss_mean_dbg" in logs:
            logs["loss_true_mean"] = logs["loss_mean_dbg"]
            logs["loss"] = logs["loss_mean_dbg"]  # optional: make console show true mean

#use modified trainer 
class MST_Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        summed_loss = outputs.loss
        num_tokens = outputs.num_tokens
        if not torch.is_tensor(num_tokens):
            num_tokens = torch.tensor(num_tokens, device=summed_loss.device, dtype=summed_loss.dtype)
        else:
            num_tokens = num_tokens.to(device=summed_loss.device, dtype=summed_loss.dtype).clamp(min=1)

        mean_loss = summed_loss / num_tokens

        # one-line scalar logs (HF Trainer will pick these up)
        if hasattr(self, "log"):
            self.log({
                "loss_mean_dbg": mean_loss.detach().float().item(),
                "loss_sum_dbg":  summed_loss.detach().float().item(),
                "num_tokens_dbg": num_tokens.detach().float().item(),
            })

        return (mean_loss, outputs) if return_outputs else mean_loss



# Usage example:
"""
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-7B-Instruct",  # or "meta-llama/Llama-3-8B-Instruct"
    torch_dtype="auto",
    device_map="auto"
)

# Wrap with MiniSequence
wrapped_model = MiniSequence(model, model_type="auto", mini_s=64)

# Use as normal
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-7B-Instruct")
inputs = tokenizer("Hello, how are you?", return_tensors="pt")

# Training
outputs = wrapped_model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
loss.backward()

# Generation
generated = wrapped_model.generate(**inputs, max_new_tokens=50)







"""
