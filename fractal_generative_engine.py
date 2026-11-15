"""
Fractal Wave Text Diffusion — Autonomous Character-Level Generative AI Engine
Progressive Priority Word Selection (Pillars -> Modifiers -> Connectives) &
100% Letter-by-Letter (Character-Level) Tokenization in PyTorch.

Author: Arvin Golshahi (Independent researcher)
Paper DOI: https://doi.org/10.5281/zenodo.24209687
Zenodo Record: https://zenodo.org/records/24209687
Repository: https://github.com/BerryFlavor101/fractal-wave-diffusion

Description:
    Implements the full generative AI version of Fractal Wave Text Diffusion:
      1. 100% Letter-by-Letter Tokenization (`CharVocabulary`): No word-level or
         subword lookup tables exist in the model. Every word on the 2D (Words, Chars)
         canvas and every semantic seed/prompt is tokenized strictly character-by-character.
      2. Progressive Semantic Anchor Generation (`generate_autonomous`):
         - Step 1: Chooses ONLY the most important words (Core Pillars, M_i=1) to
           establish the core meaning ("punti saldi") and generates their lexical
           roots (mu_{i,j}=1) letter-by-letter.
         - Step 2: Guided directly by the Pillar letters on the canvas, chooses the
           Modifiers (M_i=2), generating their roots (mu_{i,j}=1) and expanding the
           Pillars' thematic stems (mu_{i,j}=2) letter-by-letter.
         - Step 3a: Guided by the Pillars and Modifiers on the canvas, chooses the
           Connectives (M_i=3), generating their nuclei (mu_{i,j}=1) and completing
           the Modifiers' internal suffixes (mu_{i,j}=2) letter-by-letter.
         - Step 3b: Cross-word attention reads all roots/stems across the canvas and
           snaps all inflectional agreement endings (mu_{i,j}=3) letter-by-letter.
      3. Two-Regime AI Training Phase (`train_generative_engine`): Trains the 3-head
         Axial Transformer to both initiate Core Pillars at Step 1 and autonomously
         derive Modifiers, Connectives, and Morphological Agreements purely from the
         letters visible on the canvas.

License (GNU AGPLv3):
    Copyright (C) 2026 Arvin Golshahi
    Licensed under the GNU Affero General Public License v3.0 (see LICENSE).
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict


# ==============================================================================
# 1. STRICT LETTER-BY-LETTER VOCABULARY & AUTOMATED TRAINING ANNOTATOR
# ==============================================================================

class CharVocabulary:
    """
    100% Character-Level (Letter-by-Letter) Vocabulary.
    Every token in the neural network is a single character or a structural mask:
      - 0: <PAD>       (inactive/padded slot)
      - 1: <CHAR_MASK> ('_' unformed letter slot inside an active word)
      - 2: <WORD_MASK> ('[...]' unborn word slot not yet placed on the canvas)
      - 3: ' '         (space separator used only when encoding linear character prompts)
      - 4..N: Individual letters and punctuation marks.
    """
    def __init__(self):
        self.pad_token = "<PAD>"
        self.char_mask = "<CHAR_MASK>"
        self.word_mask = "<WORD_MASK>"
        self.space_token = " "

        alphabet = (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "àèéìòùÀÈÉÌÒÙ'.,!?;:-"
        )
        self.itos = [
            self.pad_token,
            self.char_mask,
            self.word_mask,
            self.space_token,
        ] + list(alphabet)
        self.stoi = {ch: idx for idx, ch in enumerate(self.itos)}

        self.pad_id = 0
        self.char_mask_id = 1
        self.word_mask_id = 2
        self.space_id = 3

    def __len__(self) -> int:
        return len(self.itos)

    def encode_word(self, word: str, max_chars: int) -> List[int]:
        """Tokenizes a single word strictly letter-by-letter up to max_chars."""
        word = word[:max_chars]
        token_ids = [self.stoi.get(ch, self.pad_id) for ch in word]
        return token_ids + [self.pad_id] * (max_chars - len(token_ids))

    def encode_char_stream(self, text: str, max_stream_len: int) -> List[int]:
        """Tokenizes any raw character sequence strictly letter-by-letter."""
        text = text.strip()[:max_stream_len]
        token_ids = [self.stoi.get(ch, self.pad_id) for ch in text]
        return token_ids + [self.pad_id] * (max_stream_len - len(token_ids))


@dataclass
class AnnotatedWord:
    text: str
    macro_tier: int        # 1 = Core Pillar, 2 = Modifier, 3 = Connective (0 = Empty slot)
    micro_tiers: List[int] # Per-letter: 1 = Root, 2 = Thematic/Deriv. Affix, 3 = Agreement Ending


class MorphosyntacticAnnotator:
    """
    Automated Morphosyntactic Decomposer used during the AI Training Phase.
    Parses raw sentences into:
      - Macro-Syntactic priority tiers M_i in {1 (Pillar), 2 (Modifier), 3 (Connective)}
      - Micro-Morphological letter tiers mu_{i,j} in {1 (Root), 2 (Affix), 3 (Ending)}
    """
    def __init__(self):
        # Articulated prepositions: spatial root letters (Micro 1) + agreement letters (Micro 3)
        self.articulated_preps: Dict[str, List[int]] = {
            "sul":   [1, 1, 3],       "sullo": [1, 1, 3, 3, 3], "sulla": [1, 1, 3, 3, 3],
            "sui":   [1, 1, 3],       "sugli": [1, 1, 3, 3, 3], "sulle": [1, 1, 3, 3, 3],
            "nel":   [1, 1, 3],       "nello": [1, 1, 3, 3, 3], "nella": [1, 1, 3, 3, 3],
            "nei":   [1, 1, 3],       "negli": [1, 1, 3, 3, 3], "nelle": [1, 1, 3, 3, 3],
            "dal":   [1, 1, 3],       "dallo": [1, 1, 3, 3, 3], "dalla": [1, 1, 3, 3, 3],
            "dai":   [1, 1, 3],       "dagli": [1, 1, 3, 3, 3], "dalle": [1, 1, 3, 3, 3],
            "del":   [1, 3, 3],       "dello": [1, 3, 3, 3, 3], "della": [1, 3, 3, 3, 3],
            "dei":   [1, 3, 3],       "degli": [1, 3, 3, 3, 3], "delle": [1, 3, 3, 3, 3],
            "al":    [1, 3],          "allo":  [1, 3, 3, 3],    "alla":  [1, 3, 3, 3],
            "ai":    [1, 3],          "agli":  [1, 3, 3, 3],    "alle":  [1, 3, 3, 3],
            "col":   [1, 1, 3],       "coi":   [1, 1, 3],
        }

        # Pure agreement articles (Macro 3, Micro 3)
        self.pure_articles = {"il", "lo", "la", "i", "gli", "le", "un", "uno", "una"}

        # Invariant functional connectives (Macro 3, Micro 1)
        self.invariant_connectives = {
            "di", "a", "da", "in", "con", "su", "per", "tra", "fra",
            "e", "ed", "o", "ma", "che", "non", "si",
            "the", "an", "on", "at", "by", "for", "with", "from", "to", "of", "and", "or", "but",
        }

        # Common lexical modifiers (adjectives / adverbs) -> Macro Tier 2
        self.known_modifiers = {
            "piccolo", "piccola", "piccoli", "piccole",
            "grande", "grandi", "veloce", "veloci", "lento", "lenta", "lenti", "lente",
            "antico", "antica", "antichi", "antiche", "nuovo", "nuova", "nuovi", "nuove",
            "bianco", "bianca", "bianchi", "bianche", "nero", "nera", "neri", "nere",
            "rosso", "rossa", "rossi", "rosse", "azzurro", "azzurra", "azzurri", "azzurre",
            "caldo", "calda", "caldi", "calde", "freddo", "fredda", "freddi", "fredde",
            "luminoso", "luminosa", "luminosi", "luminose", "silenzioso", "silenziosa",
            "silenziosi", "silenziose", "profondamente", "velocemente", "lentamente",
            "dolcemente", "silenziosamente", "sempre", "spesso", "molto", "troppo",
            "small", "little", "big", "fast", "slow", "bright", "silent", "deeply", "quickly",
        }

        self.adverb_suffixes = ("amente", "mente", "oni")
        self.verb_thematic_and_endings = [
            ("ava", "no"), ("eva", "no"), ("iva", "no"),
            ("ava", "mo"), ("eva", "mo"), ("iva", "mo"),
            ("av", "o"), ("av", "i"), ("av", "a"),
            ("ev", "o"), ("ev", "i"), ("ev", "a"),
            ("iv", "o"), ("iv", "i"), ("iv", "a"),
            ("er", "à"), ("er", "anno"), ("ir", "anno"),
            ("a", "no"), ("o", "no"),
        ]
        self.derivational_infixes = (
            "zion", "ment", "abil", "ibil", "os", "al", "ar", "ic", "izz", "eggi"
        )

    def annotate_word(self, word: str) -> AnnotatedWord:
        low = word.lower()
        n = len(word)

        # 1. Connectives (Macro Tier 3)
        if low in self.articulated_preps:
            micro = self.articulated_preps[low]
            assert len(micro) == n
            return AnnotatedWord(word, macro_tier=3, micro_tiers=micro)

        if low in self.pure_articles:
            return AnnotatedWord(word, macro_tier=3, micro_tiers=[3] * n)

        if low in self.invariant_connectives:
            return AnnotatedWord(word, macro_tier=3, micro_tiers=[1] * n)

        # 2. Adverbs in '-amente' / '-mente' (Macro Tier 2)
        for adv_suf in self.adverb_suffixes:
            if low.endswith(adv_suf) and n >= len(adv_suf) + 2:
                root_len = n - len(adv_suf)
                micro = [1] * root_len + [2] * len(adv_suf)
                assert len(micro) == n
                return AnnotatedWord(word, macro_tier=2, micro_tiers=micro)

        # 3. Verbs with Thematic Stems + Agreement Endings (Macro Tier 1)
        for them, end in self.verb_thematic_and_endings:
            combined = them + end
            if low.endswith(combined) and n >= len(combined) + 2:
                root_len = n - len(combined)
                micro = [1] * root_len + [2] * len(them) + [3] * len(end)
                assert len(micro) == n
                return AnnotatedWord(word, macro_tier=1, micro_tiers=micro)

        # 4. Nouns (Macro Tier 1) vs. Adjectives/Modifiers (Macro Tier 2)
        if low in self.known_modifiers or low.endswith(("oso", "osa", "osi", "ose", "revole", "revoli")):
            macro_tier = 2
        else:
            macro_tier = 1

        # 5. Letter-by-letter Micro Tiers for Nouns / Adjectives
        if n <= 2:
            return AnnotatedWord(word, macro_tier=macro_tier, micro_tiers=[1] * n)

        has_inflection = low[-1] in "oaieèàòù"
        end_len = 1 if has_inflection else 0
        stem = low[: n - end_len]

        infix_len = 0
        for inf in self.derivational_infixes:
            if stem.endswith(inf) and len(stem) >= len(inf) + 2:
                infix_len = len(inf)
                break

        root_len = n - infix_len - end_len
        micro = [1] * root_len + [2] * infix_len + [3] * end_len
        assert len(micro) == n
        return AnnotatedWord(word, macro_tier=macro_tier, micro_tiers=micro)

    def annotate_sentence(self, sentence: str) -> List[AnnotatedWord]:
        words = re.findall(r"[A-Za-zÀ-ÿ']+", sentence.strip())
        return [self.annotate_word(w) for w in words]

    def extract_pillar_seed_string(self, annotated_words: List[AnnotatedWord]) -> str:
        """
        Extracts the character-level stream of Core Pillar words (Macro Tier 1)
        to serve as a 100% letter-tokenized semantic intent during Step 0 -> Step 1.
        """
        pillars = [w.text.lower() for w in annotated_words if w.macro_tier == 1]
        return " ".join(pillars)

    def encode_corpus(
        self,
        sentences: List[str],
        vocab: CharVocabulary,
        max_words: Optional[int] = None,
        max_chars: Optional[int] = None,
        max_prompt_chars: int = 48,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """
        Converts raw sentences into 2D letter-by-letter tensors + letter-tokenized
        semantic seeds (Zero word-level or sentence-ID lookup tables).
        """
        parsed_corpus = [self.annotate_sentence(s) for s in sentences]
        if max_words is None:
            max_words = max(len(s) for s in parsed_corpus)
        if max_chars is None:
            max_chars = max(len(w.text) for s in parsed_corpus for w in s)

        batch_clean, batch_macro, batch_micro, batch_char_prompts = [], [], [], []

        for sent in parsed_corpus:
            sent = sent[:max_words]
            s_clean, s_macro, s_micro = [], [], []

            for w_spec in sent:
                s_clean.append(vocab.encode_word(w_spec.text, max_chars))
                s_macro.append(w_spec.macro_tier)
                m_tiers = w_spec.micro_tiers[:max_chars]
                s_micro.append(m_tiers + [0] * (max_chars - len(m_tiers)))

            while len(s_clean) < max_words:
                s_clean.append([vocab.pad_id] * max_chars)
                s_macro.append(0)
                s_micro.append([0] * max_chars)

            pillar_stream = self.extract_pillar_seed_string(sent)
            batch_char_prompts.append(vocab.encode_char_stream(pillar_stream, max_prompt_chars))
            batch_clean.append(s_clean)
            batch_macro.append(s_macro)
            batch_micro.append(s_micro)

        return (
            torch.tensor(batch_clean, dtype=torch.long),
            torch.tensor(batch_macro, dtype=torch.long),
            torch.tensor(batch_micro, dtype=torch.long),
            torch.tensor(batch_char_prompts, dtype=torch.long),
            max_words,
            max_chars,
        )


# ==============================================================================
# 2. FRACTAL WAVE SCHEDULER (PROGRESSIVE PRIORITY & LETTER UNMASKING)
# ==============================================================================

class AutonomousFractalScheduler:
    """
    Governs the 2D (Words x Letters) forward masking and progressive reverse
    unmasking schedule across Macro-Syntactic and Micro-Morphological axes.
    """
    def __init__(self, vocab: CharVocabulary):
        self.vocab = vocab
        self.schedule = [
            (0, 0, False, "Step 0  | Initial Noise          "),
            (1, 1, False, "Step 1  | Core Pillars (Roots)   "),
            (2, 2, False, "Step 2  | Modifiers & Stems      "),
            (3, 3, False, "Step 3a | Connective Nuclei      "),
            (4, 3, True,  "Step 3b | Magnetic Agreement Snap"),
        ]

    def compute_canvas_template(
        self,
        macro_tiers: torch.Tensor,  # (B, W): 1=Pillar, 2=Modifier, 3=Connective, 0=Empty, -1=Unborn
        micro_tiers: torch.Tensor,  # (B, W, C): 1=Root, 2=Affix, 3=Agreement, 0=Pad
        macro_time: int,            # t in {0, 1, 2, 3}
        snap_agreement: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, W, C = micro_tiers.shape
        device = micro_tiers.device

        if macro_time == 0:
            template_ids = torch.full(
                (B, W, C), self.vocab.word_mask_id, dtype=torch.long, device=device
            )
            char_visible = torch.zeros((B, W, C), dtype=torch.bool, device=device)
            return template_ids, char_visible

        # Words with macro_tier == -1 are still unborn ('[...]') during progressive inference
        unborn_slot = (macro_tiers == -1).unsqueeze(-1).expand(B, W, C)
        valid_committed_word = (macro_tiers > 0).unsqueeze(-1).expand(B, W, C)
        is_pad_char = (micro_tiers == 0) | (~valid_committed_word)

        # 1. Horizontal Macro-Axis: Which committed words are active by macro_time t?
        word_active = valid_committed_word & (macro_time >= macro_tiers).unsqueeze(-1).expand(B, W, C)

        # 2. Vertical Micro-Axis: Local word age tau_i(t) = t - M_i + 1
        word_age = (macro_time - macro_tiers + 1).unsqueeze(-1).expand(B, W, C)

        local_morph_visible = ((micro_tiers == 1) & (word_age >= 1)) | (
            (micro_tiers == 2) & (word_age >= 2)
        )
        global_agree_visible = (micro_tiers == 3) & snap_agreement

        char_visible = word_active & (local_morph_visible | global_agree_visible) & (~is_pad_char)

        # Build 2D structural template:
        # - Active words, locked character slots -> <CHAR_MASK> ('_')
        # - Unborn words (macro_tier == -1 or macro_tier > macro_time) -> <WORD_MASK> ('[...]')
        # - Unused word slots (macro_tier == 0) or pad chars -> <PAD>
        template_ids = torch.full((B, W, C), self.vocab.pad_id, dtype=torch.long, device=device)
        template_ids = torch.where(
            word_active & (~is_pad_char),
            torch.tensor(self.vocab.char_mask_id, device=device),
            template_ids,
        )
        template_ids = torch.where(
            unborn_slot | (valid_committed_word & (~word_active)),
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
        template_ids, char_visible = self.compute_canvas_template(
            macro_tiers, micro_tiers, macro_time, snap_agreement
        )
        return torch.where(char_visible, clean_ids, template_ids)


# ==============================================================================
# 3. 100% CHARACTER-TOKENIZED 3-HEAD AXIAL TRANSFORMER
# ==============================================================================

class AutonomousFractalWaveModel(nn.Module):
    """
    Two-Axis Transformer with Endogenous Structural Planning and 100% Letter-by-Letter
    Tokenization. Uses a single shared character embedding (`self.char_emb`) across both
    the 2D sentence canvas (B, W, C) and any optional character-level prompt stream (B, L).
    """
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        max_words: int,
        max_chars: int,
        max_prompt_chars: int = 48,
        num_steps: int = 5,
        d_model: int = 192,
        nhead: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.max_words = max_words
        self.max_chars = max_chars
        self.max_prompt_chars = max_prompt_chars

        # Shared 100% Character-Level Embedding Table
        self.char_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.word_pos_emb = nn.Embedding(max_words, d_model)
        self.char_pos_emb = nn.Embedding(max_chars, d_model)
        self.step_emb = nn.Embedding(num_steps, d_model)

        # Character-Level Prompt Encoder (encodes letter-by-letter streams without word tokens)
        self.prompt_pos_emb = nn.Embedding(max_prompt_chars, d_model)
        prompt_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, dropout=0.0, batch_first=True
        )
        self.char_prompt_encoder = nn.TransformerEncoder(prompt_layer, num_layers=1)
        self.latent_proj = nn.Linear(d_model, d_model)

        # Axis 1 (Vertical): Intra-Word Letter-by-Letter Micro-Transformer
        micro_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.micro_transformer = nn.TransformerEncoder(micro_layer, num_layers=num_layers)

        # Axis 2 (Horizontal): Cross-Word Macro-Syntactic & Agreement Transformer
        macro_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.macro_transformer = nn.TransformerEncoder(macro_layer, num_layers=num_layers)

        # 3 Synchronized Prediction Heads
        self.char_head = nn.Linear(d_model, vocab_size)
        self.micro_tier_head = nn.Linear(d_model, 4)  # {0: Pad, 1: Root, 2: Affix, 3: Agreement}
        self.macro_tier_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 4),  # {0: Empty Slot, 1: Pillar, 2: Modifier, 3: Connective}
        )

    def encode_letter_prompt(self, prompt_char_ids: torch.Tensor) -> torch.Tensor:
        """Encodes a 1D stream of character tokens (B, L) into a semantic vector (B, 1, 1, d_model)."""
        B, L = prompt_char_ids.shape
        device = prompt_char_ids.device
        p_pos = torch.arange(L, device=device).unsqueeze(0)
        p_h = self.char_emb(prompt_char_ids) + self.prompt_pos_emb(p_pos)

        p_pad_mask = (prompt_char_ids == self.pad_id)
        safe_p_mask = p_pad_mask.clone()
        safe_p_mask[:, 0] = False
        p_encoded = self.char_prompt_encoder(p_h, src_key_padding_mask=safe_p_mask)

        # Mean-pool over non-padding letters
        active_letters = (~p_pad_mask).float().unsqueeze(-1)
        pooled = (p_encoded * active_letters).sum(dim=1) / active_letters.sum(dim=1).clamp(min=1.0)
        return pooled.view(B, 1, 1, self.d_model)

    def forward(
        self,
        canvas_ids: torch.Tensor,
        step_idx: torch.Tensor,
        prompt_char_ids: Optional[torch.Tensor] = None,
        latent_z: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, W, C = canvas_ids.shape
        device = canvas_ids.device

        w_pos = torch.arange(W, device=device).view(1, W, 1)
        c_pos = torch.arange(C, device=device).view(1, 1, C)
        s_emb = self.step_emb(step_idx).view(B, 1, 1, self.d_model)

        context = torch.zeros((B, 1, 1, self.d_model), device=device)
        if prompt_char_ids is not None:
            context = context + self.encode_letter_prompt(prompt_char_ids)
        if latent_z is not None:
            context = context + self.latent_proj(latent_z).view(B, 1, 1, self.d_model)

        h = (
            self.char_emb(canvas_ids)
            + self.word_pos_emb(w_pos)
            + self.char_pos_emb(c_pos)
            + s_emb
            + context
        )

        pad_mask = (canvas_ids == self.pad_id)

        # 1. Vertical Pass: Intra-word letter-by-letter Micro-Attention
        h_micro = h.view(B * W, C, self.d_model)
        safe_micro_mask = pad_mask.view(B * W, C).clone()
        safe_micro_mask[:, 0] = False
        h_micro = self.micro_transformer(h_micro, src_key_padding_mask=safe_micro_mask)
        h = h_micro.view(B, W, C, self.d_model)

        # 2. Horizontal Pass: Cross-word letter-by-letter Macro & Agreement Attention
        h_macro = h.view(B, W * C, self.d_model)
        safe_macro_mask = pad_mask.view(B, W * C).clone()
        safe_macro_mask[:, 0] = False
        h_macro = self.macro_transformer(h_macro, src_key_padding_mask=safe_macro_mask)
        h = h_macro.view(B, W, C, self.d_model)

        # 3. Structural & Letter Prediction Heads
        char_logits = self.char_head(h)
        micro_logits = self.micro_tier_head(h)
        word_repr = h.mean(dim=2)
        macro_logits = self.macro_tier_head(word_repr)

        return char_logits, macro_logits, micro_logits


# ==============================================================================
# 4. PROGRESSIVE PRIORITY & LETTER-BY-LETTER REVERSE DIFFUSION GENERATION
# ==============================================================================

def sample_letter_tokens(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_k: int = 0,
    disallow_ids: Tuple[int, ...] = (0, 1, 2, 3),
) -> torch.Tensor:
    """Samples single-letter IDs from character logits, blocking mask/pad/space IDs."""
    filtered_logits = logits.clone()
    for tid in disallow_ids:
        if tid < filtered_logits.size(-1):
            filtered_logits[..., tid] = -1e9

    if temperature <= 0.0:
        return filtered_logits.argmax(dim=-1)

    scaled_logits = filtered_logits / max(temperature, 1e-5)
    if top_k > 0:
        top_vals, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)), dim=-1)
        min_acceptable = top_vals[..., -1].unsqueeze(-1)
        scaled_logits = torch.where(
            scaled_logits < min_acceptable,
            torch.full_like(scaled_logits, -1e9),
            scaled_logits,
        )

    probs = F.softmax(scaled_logits, dim=-1)
    flat_probs = probs.view(-1, probs.size(-1))
    return torch.multinomial(flat_probs, num_samples=1).view(logits.shape[:-1])


def render_canvas(canvas_ids: torch.Tensor, vocab: CharVocabulary) -> str:
    """Renders a 2D (W, C) letter tensor into the visual 'Text Construction Site' string."""
    words_out = []
    W, C = canvas_ids.shape
    for w in range(W):
        tokens = canvas_ids[w].tolist()
        if all(t == vocab.pad_id for t in tokens):
            continue
        if tokens[0] == vocab.word_mask_id:
            words_out.append("[...]")
        else:
            chars = []
            for tid in tokens:
                if tid == vocab.pad_id:
                    continue
                elif tid == vocab.char_mask_id:
                    chars.append("_")
                elif tid == vocab.word_mask_id:
                    chars.append("?")
                else:
                    chars.append(vocab.itos[tid])
            if chars:
                words_out.append(" ".join(chars))
    return "   ".join(words_out)


def canvas_to_plain_text(canvas_ids: torch.Tensor, vocab: CharVocabulary) -> str:
    """Converts a crystallized 2D (W, C) letter canvas into a clean text string."""
    words = []
    W, C = canvas_ids.shape
    for w in range(W):
        chars = [
            vocab.itos[tid]
            for tid in canvas_ids[w].tolist()
            if tid not in (vocab.pad_id, vocab.char_mask_id, vocab.word_mask_id, vocab.space_id)
        ]
        if chars:
            words.append("".join(chars))
    return " ".join(words)


@torch.no_grad()
def generate_autonomous(
    model: AutonomousFractalWaveModel,
    scheduler: AutonomousFractalScheduler,
    vocab: CharVocabulary,
    prompt_text: Optional[str] = None,
    latent_z: Optional[torch.Tensor] = None,
    temperature: float = 0.0,
    top_k: int = 5,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PROGRESSIVE PRIORITY & LETTER-BY-LETTER REVERSE DIFFUSION:
    Builds the sentence without any oracle masks by first selecting the most important
    words (Core Pillars at Step 1) to anchor the meaning, and then progressively
    selecting Modifiers (Step 2), Connectives (Step 3a), and Agreement Endings (Step 3b)
    purely from the letters crystallized on the canvas.
    """
    model.eval()
    device = next(model.parameters()).device
    B = 1
    W, C = model.max_words, model.max_chars

    prompt_char_ids = None
    if prompt_text is not None:
        encoded_prompt = vocab.encode_char_stream(prompt_text, model.max_prompt_chars)
        prompt_char_ids = torch.tensor([encoded_prompt], dtype=torch.long, device=device)

    # Step 0: Canvas starts as 100% unborn word slots '[...]' (<WORD_MASK>)
    current_canvas = torch.full((B, W, C), vocab.word_mask_id, dtype=torch.long, device=device)
    previously_visible = torch.zeros((B, W, C), dtype=torch.bool, device=device)

    # Track progressive word selection: -1 = Unborn ('[...]'), 1 = Pillar, 2 = Modifier, 3 = Connective, 0 = Empty
    committed_macro = torch.full((B, W), -1, dtype=torch.long, device=device)
    committed_micro = torch.zeros((B, W, C), dtype=torch.long, device=device)

    if verbose:
        print(f"[{scheduler.schedule[0][3]}] -> {render_canvas(current_canvas[0], vocab)}")

    # Progressive Reverse Wave: Step 0 -> Step 1 -> Step 2 -> Step 3a -> Step 3b
    for idx in range(len(scheduler.schedule) - 1):
        curr_step_idx = scheduler.schedule[idx][0]
        _, next_macro_t, next_snap, next_label = scheduler.schedule[idx + 1]

        step_tensor = torch.full((B,), curr_step_idx, dtype=torch.long, device=device)

        # At Step 0 -> Step 1, the AI uses the character-level intent to choose the Core Pillars ("punti saldi").
        # Once the Core Pillars are on the canvas (curr_step_idx >= 1), the AI drops the external prompt
        # (active_prompt = None) and generates Modifiers, Connectives, and Agreements purely from the Pillar letters!
        active_prompt = prompt_char_ids if curr_step_idx == 0 else None

        char_logits, macro_logits, micro_logits = model(
            current_canvas, step_tensor, prompt_char_ids=active_prompt, latent_z=latent_z
        )

        step_pred_macro = macro_logits.argmax(dim=-1)  # (B, W)
        step_pred_micro = micro_logits.argmax(dim=-1)  # (B, W, C)

        # PROGRESSIVE WORD SELECTION BY INFORMATIONAL PRIORITY:
        if next_macro_t == 1:
            # Step 1: Select ONLY Core Pillars (Macro Tier 1) & unused padding (0); others stay Unborn (-1)
            is_pillar = (step_pred_macro == 1)
            is_empty = (step_pred_macro == 0)
            committed_macro = torch.where(is_pillar, torch.tensor(1, device=device), committed_macro)
            committed_macro = torch.where(is_empty, torch.tensor(0, device=device), committed_macro)
            committed_micro = torch.where(is_pillar.unsqueeze(-1), step_pred_micro, committed_micro)

        elif next_macro_t == 2:
            # Step 2: Anchored on the Pillars, select Modifiers (Macro Tier 2) from remaining Unborn (-1) slots
            unborn_mask = (committed_macro == -1)
            is_modifier = unborn_mask & (step_pred_macro == 2)
            is_empty = unborn_mask & (step_pred_macro == 0)
            committed_macro = torch.where(is_modifier, torch.tensor(2, device=device), committed_macro)
            committed_macro = torch.where(is_empty, torch.tensor(0, device=device), committed_macro)
            committed_micro = torch.where(is_modifier.unsqueeze(-1), step_pred_micro, committed_micro)

        elif next_macro_t == 3 and not next_snap:
            # Step 3a: Anchored on Pillars + Modifiers, resolve remaining Unborn (-1) slots into Connectives (3) or Empty (0)
            unborn_mask = (committed_macro == -1)
            new_tier = torch.where(
                step_pred_macro == 0,
                torch.tensor(0, device=device),
                torch.tensor(3, device=device),
            )
            committed_macro = torch.where(unborn_mask, new_tier, committed_macro)
            committed_micro = torch.where(unborn_mask.unsqueeze(-1), step_pred_micro, committed_micro)
            committed_micro = torch.where(
                (committed_macro == 0).unsqueeze(-1),
                torch.zeros_like(committed_micro),
                committed_micro,
            )

        # Sample individual letters from the character head
        predicted_char_ids = sample_letter_tokens(
            char_logits,
            temperature=temperature,
            top_k=top_k,
            disallow_ids=(vocab.pad_id, vocab.char_mask_id, vocab.word_mask_id, vocab.space_id),
        )

        # Unlock letter slots corresponding to the current wave phase
        next_template, next_visible = scheduler.compute_canvas_template(
            committed_macro, committed_micro, macro_time=next_macro_t, snap_agreement=next_snap
        )

        newly_unlocked = next_visible & (~previously_visible)
        updated_canvas = next_template.clone()
        updated_canvas[previously_visible] = current_canvas[previously_visible]
        updated_canvas[newly_unlocked] = predicted_char_ids[newly_unlocked]

        current_canvas = updated_canvas
        previously_visible = next_visible

        if verbose:
            print(f"[{next_label}] -> {render_canvas(current_canvas[0], vocab)}")

    return current_canvas, committed_macro, committed_micro


# ==============================================================================
# 5. AI TRAINING PHASE (TWO-REGIME LETTER-BY-LETTER CURRICULUM) & DEMO
# ==============================================================================

def train_generative_engine(
    model: AutonomousFractalWaveModel,
    scheduler: AutonomousFractalScheduler,
    clean_ids: torch.Tensor,
    macro_tiers: torch.Tensor,
    micro_tiers: torch.Tensor,
    char_prompts: torch.Tensor,
    vocab: CharVocabulary,
    epochs: int = 350,
    lr: float = 3e-3,
) -> None:
    """
    Trains the 100% character-tokenized AutonomousFractalWaveModel:
      - At Step 0 (full '[...]' noise): Trains with the letter-tokenized prompt stream so
        the AI learns to select and place the Step 1 Core Pillars ("i punti saldi").
      - At Steps 1, 2, 3a (once the Core Pillars are visible on the canvas): Trains BOTH
        with and without the prompt (`prompt_char_ids=None`), forcing the AI to derive
        Step 2 Modifiers, Step 3a Connectives, and Step 3b Morphological Agreements
        directly from the Pillar letters crystallized on the canvas.
    """
    device = clean_ids.device
    B = clean_ids.shape[0]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)

        for step_idx, macro_t, snap_agree, _ in scheduler.schedule[:-1]:
            noisy_canvas = scheduler.forward_corrupt(
                clean_ids, macro_tiers, micro_tiers, macro_t, snap_agree
            )
            step_tensor = torch.full((B,), step_idx, dtype=torch.long, device=device)

            # Regime 1: Character-prompted pass
            char_logits, macro_logits, micro_logits = model(
                noisy_canvas, step_tensor, prompt_char_ids=char_prompts
            )
            loss_char = F.cross_entropy(
                char_logits.view(-1, len(vocab)), clean_ids.view(-1), ignore_index=vocab.pad_id
            )
            loss_macro = F.cross_entropy(macro_logits.view(-1, 4), macro_tiers.view(-1))
            loss_micro = F.cross_entropy(micro_logits.view(-1, 4), micro_tiers.view(-1))
            total_loss = total_loss + (loss_char + 0.5 * loss_macro + 0.5 * loss_micro)

            # Regime 2: Pure Canvas-Driven pass (Steps >= 1 depend ONLY on the letters on the canvas)
            if step_idx >= 1:
                u_char_logits, u_macro_logits, u_micro_logits = model(
                    noisy_canvas, step_tensor, prompt_char_ids=None
                )
                u_loss_char = F.cross_entropy(
                    u_char_logits.view(-1, len(vocab)), clean_ids.view(-1), ignore_index=vocab.pad_id
                )
                u_loss_macro = F.cross_entropy(u_macro_logits.view(-1, 4), macro_tiers.view(-1))
                u_loss_micro = F.cross_entropy(u_micro_logits.view(-1, 4), micro_tiers.view(-1))
                total_loss = total_loss + (u_loss_char + 0.5 * u_loss_macro + 0.5 * u_loss_micro)

        total_loss.backward()
        optimizer.step()


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Multi-sentence corpus with diverse syntax, word counts, genders, and numbers
    raw_corpus = [
        "I piccoli gatti dormivano profondamente sul tappeto",
        "Le piccole gatte dormivano profondamente sulla poltrona",
        "Il piccolo gatto dormiva profondamente sul tappeto",
        "Il grande cane correva velocemente nel giardino",
        "I nuovi treni viaggiavano silenziosamente nella notte",
        "La fredda pioggia cadeva lentamente sulle strade",
    ]

    vocab = CharVocabulary()
    annotator = MorphosyntacticAnnotator()

    print("1. [LETTER-BY-LETTER PRE-PROCESSING] Decomposing raw sentences into 2D character tensors...")
    clean_ids, macro_tiers, micro_tiers, char_prompts, max_words, max_chars = annotator.encode_corpus(
        raw_corpus, vocab
    )
    clean_ids = clean_ids.to(device)
    macro_tiers = macro_tiers.to(device)
    micro_tiers = micro_tiers.to(device)
    char_prompts = char_prompts.to(device)

    scheduler = AutonomousFractalScheduler(vocab)
    model = AutonomousFractalWaveModel(
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        max_words=max_words,
        max_chars=max_chars,
        max_prompt_chars=char_prompts.shape[1],
        num_steps=len(scheduler.schedule),
        d_model=192,
        nhead=4,
        num_layers=2,
    ).to(device)

    print(
        f"2. [AI TRAINING PHASE] Training 100% Character-Level AutonomousFractalWaveModel "
        f"(Vocab size: {len(vocab)} single-char tokens, Canvas: {max_words}x{max_chars})..."
    )
    train_generative_engine(
        model,
        scheduler,
        clean_ids,
        macro_tiers,
        micro_tiers,
        char_prompts,
        vocab,
        epochs=350,
        lr=3e-3,
    )
    print("Training complete.\n")

    print("=" * 105)
    print(" PROGRESSIVE PRIORITY GENERATION (PILLARS -> MODIFIERS -> CONNECTIVES) — 100% LETTER-BY-LETTER")
    print("=" * 105)

    total_correct, total_tokens = 0, 0
    for idx, target_text in enumerate(raw_corpus):
        parsed = annotator.annotate_sentence(target_text)
        pillar_seed_str = annotator.extract_pillar_seed_string(parsed)

        print(f"\n--- Sentence #{idx + 1} | Character-Tokenized Pillar Seed: \"{pillar_seed_str}\" ---")
        gen_canvas, committed_macro, _ = generate_autonomous(
            model,
            scheduler,
            vocab,
            prompt_text=pillar_seed_str,
            temperature=0.0,
            verbose=True,
        )
        final_sentence = canvas_to_plain_text(gen_canvas[0], vocab)
        print(f"   -> Progressive Word Tiers Chosen (1=Pillar, 2=Mod, 3=Conn, 0=Pad): {committed_macro[0].tolist()}")
        print(f"   -> Final AI Generated Sentence: \"{final_sentence}\"")

        non_pad = clean_ids[idx] != vocab.pad_id
        total_correct += (gen_canvas[0][non_pad] == clean_ids[idx][non_pad]).sum().item()
        total_tokens += non_pad.sum().item()

    print("\n" + "=" * 105)
    print(f"Multi-Sentence Letter-by-Letter Autonomous Accuracy: {(total_correct / total_tokens) * 100:.2f}%")
    print("=" * 105)


if __name__ == "__main__":
    main()
