"""
Fractal Wave Text Diffusion
Hierarchical Two-Axis Discrete Diffusion (Macro-Syntactic & Micro-Morphological) in PyTorch.

Author: Arvin Golshahi (Independent researcher)
Paper DOI: https://doi.org/10.5281/zenodo.24209687
Zenodo Record: https://zenodo.org/records/24209687
Repository: https://github.com/BerryFlavor101/fractal-wave-diffusion

Description:
    Implements a 2D discrete diffusion model where text is generated coarse-to-fine
    simultaneously across the Sentence Axis (Macro-Diffusion: Pillars -> Modifiers -> Connectives)
    and the Word Axis (Micro-Diffusion: Lexical Root -> Thematic Affix -> Agreement Ending).

AI Assistance & Transparency Notice:
    This software and accompanying paper were organized and structured with the assistance
    of artificial intelligence tools; however, all intellectual content, theoretical logic,
    architectural design, analyses, and conclusions are the original work of the author
    (Arvin Golshahi).

License (GNU AGPLv3):
    Copyright (C) 2026 Arvin Golshahi

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple


# ==============================================================================
# 1. MORPHOSYNTACTIC DATA STRUCTURES & VOCABULARY
# ==============================================================================

@dataclass
class WordSpec:
    """
    Defines the structural hierarchy of a word along both diffusion axes.
    - macro_tier: 1 = Core Pillar (Step 1), 2 = Modifier (Step 2), 3 = Connective (Step 3)
    - micro_tiers: Per-character morphological level:
        1 = Lexical Root (emerges at local word age >= 1)
        2 = Internal Thematic/Derivational Affix (emerges at local word age >= 2)
        3 = Relational Inflectional Ending (snaps globally at Step 3b via Cross-Word Agreement)
    """
    text: str
    macro_tier: int
    micro_tiers: List[int]


class CharVocabulary:
    """Character-level vocabulary with dedicated Macro and Micro mask tokens."""
    def __init__(self):
        self.pad_token = "<PAD>"
        self.char_mask = "<CHAR_MASK>"  # Rendered as '_' (unformed character slot)
        self.word_mask = "<WORD_MASK>"  # Rendered as '[...]' (unborn word slot)

        alphabet = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.itos = [self.pad_token, self.char_mask, self.word_mask] + alphabet
        self.stoi = {ch: idx for idx, ch in enumerate(self.itos)}

        self.pad_id = 0
        self.char_mask_id = 1
        self.word_mask_id = 2

    def __len__(self) -> int:
        return len(self.itos)

    def encode_word(self, word: str, max_len: int) -> List[int]:
        token_ids = [self.stoi[ch] for ch in word]
        return token_ids + [self.pad_id] * (max_len - len(token_ids))


# ==============================================================================
# 2. FRACTAL WAVE DIFFUSION SCHEDULER
# ==============================================================================

class FractalWaveScheduler:
    """
    Governs the 2D forward masking and reverse unmasking schedule.
    Combines Local Word Age (for roots and internal affixes) with
    Global Phase Synchronization (for real-time inflectional agreement).
    """
    def __init__(self, vocab: CharVocabulary):
        self.vocab = vocab
        # Schedule tuples: (step_index, macro_time_t, snap_global_agreement, description)
        self.schedule = [
            (0, 0, False, "Step 0  | Initial Noise          "),
            (1, 1, False, "Step 1  | Core Pillars (Roots)   "),
            (2, 2, False, "Step 2  | Modifiers & Stems      "),
            (3, 3, False, "Step 3a | Connective Nuclei      "),
            (4, 3, True,  "Step 3b | Magnetic Agreement Snap"),
        ]

    def compute_canvas_template(
        self,
        macro_tiers: torch.Tensor,  # (B, W)
        micro_tiers: torch.Tensor,  # (B, W, C)
        macro_time: int,            # t in {0, 1, 2, 3}
        snap_agreement: bool        # True only at final agreement synchronization (Step 3b)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the structural mask of the canvas WITHOUT accessing ground-truth characters.
        Returns:
            template_ids: (B, W, C) filled with <WORD_MASK>, <CHAR_MASK>, or <PAD>
            char_visible: (B, W, C) boolean mask of characters unlocked at this step
        """
        B, W, C = micro_tiers.shape
        device = micro_tiers.device
        is_pad = (micro_tiers == 0)

        # 1. Horizontal Macro-Axis: Which words have entered the canvas by time t?
        word_active = (macro_time >= macro_tiers).unsqueeze(-1).expand(B, W, C)

        # 2. Vertical Micro-Axis: Local age of each word on the canvas: tau_i = t - M_i + 1
        word_age = (macro_time - macro_tiers + 1).unsqueeze(-1).expand(B, W, C)

        # Visibility laws:
        # - Local Morphology (micro_tiers 1 & 2): unlocked when word_age >= micro_tier
        # - Global Agreement (micro_tier 3): locked as '_' until snap_agreement is True
        local_morph_visible = ((micro_tiers == 1) & (word_age >= 1)) | (
            (micro_tiers == 2) & (word_age >= 2)
        )
        global_agree_visible = (micro_tiers == 3) & snap_agreement

        char_visible = word_active & (local_morph_visible | global_agree_visible) & (~is_pad)

        # Build structural canvas template
        template_ids = torch.full((B, W, C), self.vocab.pad_id, dtype=torch.long, device=device)
        template_ids = torch.where(
            word_active & (~is_pad),
            torch.tensor(self.vocab.char_mask_id, device=device),
            template_ids,
        )
        template_ids = torch.where(
            ~word_active,
            torch.tensor(self.vocab.word_mask_id, device=device),
            template_ids,
        )

        return template_ids, char_visible

    def forward_corrupt(
        self,
        clean_ids: torch.Tensor,
        macro_tiers: torch.Tensor,
        micro_tiers: torch.Tensor,
        macro_time: int,
        snap_agreement: bool,
    ) -> torch.Tensor:
        """Applies the forward diffusion mask to ground-truth characters for training."""
        template_ids, char_visible = self.compute_canvas_template(
            macro_tiers, micro_tiers, macro_time, snap_agreement
        )
        return torch.where(char_visible, clean_ids, template_ids)


# ==============================================================================
# 3. NEURAL ARCHITECTURE: AXIAL & MAGNETIC AGREEMENT TRANSFORMER
# ==============================================================================

class FractalWaveTransformer(nn.Module):
    """
    Two-Axis Transformer operating on the 2D (Batch, Words, Chars) text canvas:
      1. Vertical Micro-Transformer: Intra-word self-attention from roots to affixes.
      2. Horizontal Macro-Transformer: Cross-word bidirectional self-attention across
         the entire sentence to resolve real-time grammatical agreement.
    """
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        max_words: int,
        max_chars: int,
        num_steps: int = 5,
        d_model: int = 128,
        nhead: int = 4,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.max_words = max_words
        self.max_chars = max_chars

        # Token & Coordinate Embeddings
        self.char_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.word_pos_emb = nn.Embedding(max_words, d_model)
        self.char_pos_emb = nn.Embedding(max_chars, d_model)
        self.step_emb = nn.Embedding(num_steps, d_model)

        # Axis 1 (Vertical): Intra-Word Micro-Diffusion Transformer
        micro_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, dropout=0.0, batch_first=True
        )
        self.micro_transformer = nn.TransformerEncoder(micro_layer, num_layers=2)

        # Axis 2 (Horizontal): Cross-Word Macro-Diffusion & Agreement Transformer
        macro_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, dropout=0.0, batch_first=True
        )
        self.macro_transformer = nn.TransformerEncoder(macro_layer, num_layers=2)

        # Character projection head
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, canvas_ids: torch.Tensor, step_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            canvas_ids: (B, W, C) current noisy canvas state
            step_idx:   (B,) current discrete diffusion step index
        Returns:
            logits:     (B, W, C, vocab_size) predicted character distributions
        """
        B, W, C = canvas_ids.shape
        device = canvas_ids.device

        # Construct 2D positional + step embeddings
        w_pos = torch.arange(W, device=device).view(1, W, 1)
        c_pos = torch.arange(C, device=device).view(1, 1, C)
        s_emb = self.step_emb(step_idx).view(B, 1, 1, self.d_model)

        h = self.char_emb(canvas_ids) + self.word_pos_emb(w_pos) + self.char_pos_emb(c_pos) + s_emb

        # Padding masks (ensure <PAD> positions do not distort attention weights)
        pad_mask = (canvas_ids == self.pad_id)  # (B, W, C)

        # 1. Vertical Pass: Intra-word Micro-Attention
        h_micro = h.view(B * W, C, self.d_model)
        micro_pad_mask = pad_mask.view(B * W, C)
        # Prevent all-True padding rows if a word is completely masked
        safe_micro_mask = micro_pad_mask.clone()
        safe_micro_mask[:, 0] = False
        h_micro = self.micro_transformer(h_micro, src_key_padding_mask=safe_micro_mask)
        h = h_micro.view(B, W, C, self.d_model)

        # 2. Horizontal Pass: Cross-word Macro & Agreement Attention
        h_macro = h.view(B, W * C, self.d_model)
        macro_pad_mask = pad_mask.view(B, W * C)
        h_macro = self.macro_transformer(h_macro, src_key_padding_mask=macro_pad_mask)
        h = h_macro.view(B, W, C, self.d_model)

        return self.output_head(h)


# ==============================================================================
# 4. BENCHMARK SENTENCE SPECIFICATION & CANVAS RENDERER
# ==============================================================================

def build_showcase_sentence(
    vocab: CharVocabulary,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Encodes the showcase sentence:
    "I piccoli gatti dormivano profondamente sul tappeto"
    ("The small cats were-sleeping deeply on-the rug")
    """
    specs = [
        # Connective (Macro 3): Masc. Pl. definite article 'I' (pure agreement token -> Micro 3)
        WordSpec("I",             macro_tier=3, micro_tiers=[3]),
        # Modifier (Macro 2): Root 'piccol-' (1) + Masc. Pl. ending '-i' (3)
        WordSpec("piccoli",       macro_tier=2, micro_tiers=[1, 1, 1, 1, 1, 1, 3]),
        # Pillar (Macro 1): Root 'gatt-' (1) + Masc. Pl. ending '-i' (3)
        WordSpec("gatti",         macro_tier=1, micro_tiers=[1, 1, 1, 1, 3]),
        # Pillar (Macro 1): Root 'dorm-' (1) + Imperfect stem '-iva-' (2) + 3rd Pl. '-no' (3)
        WordSpec("dormivano",     macro_tier=1, micro_tiers=[1, 1, 1, 1, 2, 2, 2, 3, 3]),
        # Modifier (Macro 2): Root 'profond-' (1) + Adverbial suffix '-amente' (2)
        WordSpec("profondamente", macro_tier=2, micro_tiers=[1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]),
        # Connective (Macro 3): Spatial core 'su' (1) + fused Masc. Sg. article '-l' (3)
        WordSpec("sul",           macro_tier=3, micro_tiers=[1, 1, 3]),
        # Pillar (Macro 1): Root 'tappet-' (1) + Masc. Sg. ending '-o' (3)
        WordSpec("tappeto",       macro_tier=1, micro_tiers=[1, 1, 1, 1, 1, 1, 3]),
    ]

    max_len = max(len(s.text) for s in specs)
    clean_ids, macro_tiers, micro_tiers = [], [], []

    for s in specs:
        clean_ids.append(vocab.encode_word(s.text, max_len))
        macro_tiers.append(s.macro_tier)
        micro_tiers.append(s.micro_tiers + [0] * (max_len - len(s.micro_tiers)))

    return (
        torch.tensor([clean_ids], dtype=torch.long),
        torch.tensor([macro_tiers], dtype=torch.long),
        torch.tensor([micro_tiers], dtype=torch.long),
        max_len,
    )


def render_canvas(canvas_ids: torch.Tensor, vocab: CharVocabulary) -> str:
    """Formats a 2D (Words, Chars) tensor into the visual 'Text Construction Site' string."""
    words_out = []
    W, C = canvas_ids.shape
    for w in range(W):
        tokens = canvas_ids[w].tolist()
        if tokens[0] == vocab.word_mask_id:
            words_out.append("[...]")
        else:
            chars = []
            for tid in tokens:
                if tid == vocab.pad_id:
                    continue
                elif tid == vocab.char_mask_id:
                    chars.append("_")
                else:
                    chars.append(vocab.itos[tid])
            words_out.append(" ".join(chars))
    return "   ".join(words_out)


# ==============================================================================
# 5. TRAINING & PURE REVERSE-DIFFUSION GENERATION
# ==============================================================================

@torch.no_grad()
def generate_from_noise(
    model: FractalWaveTransformer,
    scheduler: FractalWaveScheduler,
    macro_tiers: torch.Tensor,
    micro_tiers: torch.Tensor,
    vocab: CharVocabulary,
) -> torch.Tensor:
    """
    Pure Reverse-Diffusion sampling loop starting from 100% noise ([...]).
    Note: Ground-truth `clean_ids` is NEVER passed to this function.
    """
    model.eval()
    device = macro_tiers.device

    # Initialize canvas at Step 0 (full noise: all words are '[...]')
    current_canvas, previously_visible = scheduler.compute_canvas_template(
        macro_tiers, micro_tiers, macro_time=0, snap_agreement=False
    )
    print(f"[{scheduler.schedule[0][3]}] -> {render_canvas(current_canvas[0], vocab)}")

    # Iterate through reverse diffusion transitions: Step 0 -> 1 -> 2 -> 3a -> 3b
    for idx in range(len(scheduler.schedule) - 1):
        curr_step_idx = scheduler.schedule[idx][0]
        _, next_macro_t, next_snap, next_label = scheduler.schedule[idx + 1]

        # 1. Neural network predicts character probabilities from current noisy canvas
        step_tensor = torch.tensor([curr_step_idx], device=device)
        logits = model(current_canvas, step_tensor)
        predicted_ids = logits.argmax(dim=-1)

        # 2. Query structural scheduler for which slots unlock at the next step
        next_template, next_visible = scheduler.compute_canvas_template(
            macro_tiers, micro_tiers, macro_time=next_macro_t, snap_agreement=next_snap
        )

        # 3. Keep previously crystallized characters fixed; fill newly unlocked slots with predictions
        newly_unlocked = next_visible & (~previously_visible)
        updated_canvas = next_template.clone()
        updated_canvas[previously_visible] = current_canvas[previously_visible]
        updated_canvas[newly_unlocked] = predicted_ids[newly_unlocked]

        current_canvas = updated_canvas
        previously_visible = next_visible

        print(f"[{next_label}] -> {render_canvas(current_canvas[0], vocab)}")

    return current_canvas


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = CharVocabulary()
    clean_ids, macro_tiers, micro_tiers, max_chars = build_showcase_sentence(vocab)
    clean_ids = clean_ids.to(device)
    macro_tiers = macro_tiers.to(device)
    micro_tiers = micro_tiers.to(device)

    scheduler = FractalWaveScheduler(vocab)
    model = FractalWaveTransformer(
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        max_words=clean_ids.shape[1],
        max_chars=max_chars,
        num_steps=len(scheduler.schedule),
        d_model=128,
        nhead=4,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    print("Training FractalWaveTransformer (2D Axial & Magnetic Diffusion)...")
    model.train()
    for epoch in range(250):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)

        # Train across all forward diffusion states (Step 0 -> Step 3a)
        for step_idx, macro_t, snap_agree, _ in scheduler.schedule[:-1]:
            noisy_canvas = scheduler.forward_corrupt(
                clean_ids, macro_tiers, micro_tiers, macro_t, snap_agree
            )
            step_tensor = torch.tensor([step_idx], device=device)
            logits = model(noisy_canvas, step_tensor)

            loss = F.cross_entropy(
                logits.view(-1, len(vocab)),
                clean_ids.view(-1),
                ignore_index=vocab.pad_id,
            )
            total_loss = total_loss + loss

        total_loss.backward()
        optimizer.step()

    print("Training complete.\n")
    print("=" * 100)
    print(" TEXT CONSTRUCTION SITE: FRACTAL WAVE REVERSE DIFFUSION (FROM PURE NOISE TO SENTENCE)")
    print("=" * 100)

    generated_canvas = generate_from_noise(model, scheduler, macro_tiers, micro_tiers, vocab)

    print("=" * 100)
    non_pad = clean_ids != vocab.pad_id
    accuracy = (generated_canvas[non_pad] == clean_ids[non_pad]).float().mean().item() * 100.0
    print(f"Reconstruction Accuracy on Unmasked Canvas: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
