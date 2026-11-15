# 🌊 Fractal Wave Text Diffusion

**Hierarchical Two-Axis Discrete Text Diffusion (Macro-Syntactic & Micro-Morphological) with 100% Character-Level Tokenization in PyTorch.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24209687.svg)](https://doi.org/10.5281/zenodo.24209687)
[![Zenodo Record](https://img.shields.io/badge/Zenodo-Record_24209687-blue.svg)](https://zenodo.org/records/24209687)
[![Code License: AGPL v3](https://img.shields.io/badge/Code_License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Paper License: CC BY 4.0](https://img.shields.io/badge/Paper_License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Author:** Arvin Golshahi (Independent researcher)

**Zenodo Publication:** [https://zenodo.org/records/24209687](https://zenodo.org/records/24209687) | **DOI:** [10.5281/zenodo.24209687](https://doi.org/10.5281/zenodo.24209687)

Unlike traditional Autoregressive Language Models (which generate text linearly from left to right using subword/word tokens) or standard Discrete Diffusion Models (which unmask random tokens uniformly across a sequence), **Fractal Wave Text Diffusion** combines **100% letter-by-letter (character-level) tokenization** with **hierarchical word selection by informational priority**.

During generation, the AI first establishes the core meaning of the sentence by placing the **most important words (Core Pillars)**, and progressively expands toward less semantically heavy words (**Modifiers**, then **Connectives**), while every single word crystallizes **letter-by-letter** from its lexical root to its final grammatical agreement ending.

---

## 📂 Repository Files

This repository provides two complementary PyTorch scripts:

1. **`fractal_diffusion.py` — Theoretical Benchmark & Paper Showcase:**
   Self-contained reference script reproducing the exact step-by-step Italian morphosyntactic testbed table from the paper (*"I piccoli gatti dormivano profondamente sul tappeto"*).
2. **`fractal_generative_engine.py` — Autonomous Character-Level Generative AI Engine:**
   Full end-to-end generative architecture for AI integration. It operates exclusively on single-letter tokens (`CharVocabulary`), includes an automated morphosyntactic decomposer (`MorphosyntacticAnnotator`) for training on arbitrary text corpora, and executes **progressive priority word selection** (first committing $`\hat{M}_i = 1`$ Core Pillars at Step 1, then deriving $`\hat{M}_i = 2`$ Modifiers at Step 2, $`\hat{M}_i = 3`$ Connectives at Step 3a, and $`\hat{\mu}_{i,j} = 3`$ Agreement Endings at Step 3b directly from the letters on the canvas).

---

## 🧠 Core Theory: Priority Word Selection + Letter-by-Letter Tokenization

In this framework, no word-level or BPE subword lookup table exists. Every token in the vocabulary is a **single letter** (plus structural masks `<PAD>`, `<CHAR_MASK>` (`_`), and `<WORD_MASK>` (`[...]`)), arranged on a 2D `(Words, Letters)` canvas:

1. **Horizontal Axis — Macro-Diffusion (Word Selection by Semantic Priority):** *Establishing semantic anchors first.*
   - **Step 1 (High Priority — Core Pillars, $`M_i = 1`$):** The AI first selects and places the core semantic anchors of the sentence (subject noun, main verb, object noun), leaving all other word positions as unborn `[...]` slots.
   - **Step 2 (Medium Priority — Modifiers, $`M_i = 2`$):** Guided directly by the Pillar letters now visible on the canvas, the AI selects where to insert adjectives and adverbs around the pillars.
   - **Step 3 (Low Priority — Connectives, $`M_i = 3`$):** Finally, functional glue words (articles, prepositions, conjunctions) emerge in the remaining slots.

2. **Vertical Axis — Micro-Diffusion (Letter-by-Letter Word Crystallization):** *From lexical root to inflectional endings.*
   - Within every selected word slot, characters are generated **letter-by-letter** from the inside out: first the **lexical root letters** ($`\mu_{i,j}=1`$), then the **internal thematic/derivational suffix letters** ($`\mu_{i,j}=2`$), and finally the **relational inflectional ending letters** ($`\mu_{i,j}=3`$).

---

## ⏳ The "Wave Effect" & Real-Time Morphological Agreement

Combining these two axes creates a **temporal wave of crystallization**: words do not finish forming at the same time. While Step 1 pillars are already growing their internal thematic suffix letters at Step 2, Step 2 modifiers are just beginning to expose their root letters.

### Showcase Example (Italian Morphosyntax Testbed)

To clearly observe letter-by-letter morphological agreement across words, we use the Italian sentence:

> **"I piccoli gatti dormivano profondamente sul tappeto"**
> *(Literal: "The [masc.pl] small [masc.pl] cats [masc.pl] slept/were-sleeping [3rd.pl] deeply on-the [masc.sg] rug [masc.sg]")*

| Diffusion Step | Macro-Diffusion (Word Priority Selection) | Micro-Diffusion (Letter-by-Letter Axis) | Visible Output in the "Text Construction Site" (Canvas) |
| :--- | :--- | :--- | :--- |
| **Step 0** | Empty syntactic canvas. | Full noise. | `[...]   [...]   [...]   [...]   [...]   [...]   [...]` |
| **Step 1** | Selects **Core Pillars** ($`M_i=1`$). | Generates lexical root letters ($`\mu=1`$) of core pillars. | `[...]   [...]   g a t t _   d o r m _ _ _ _ _   [...]   [...]   t a p p e t _` |
| **Step 2** | Selects **Modifiers** ($`M_i=2`$) anchored on Pillars. | Pillars expand thematic letters (`-iva-`); modifiers reveal root letters. | `[...]   p i c c o l _   g a t t _   d o r m i v a _ _   p r o f o n d _ _ _ _ _ _   [...]   t a p p e t _` |
| **Step 3a** | Selects **Connectives** ($`M_i=3`$). | Modifiers complete suffix letters (`-amente`); connective root letters (`su_`) appear; agreement letters wait. | `_   p i c c o l _   g a t t _   d o r m i v a _ _   p r o f o n d a m e n t e   s u _   t a p p e t _` |
| **Step 3b** | **Global Agreement Snap.** | All inflectional ending letters ($`\mu=3`$) crystallize simultaneously. | `I   p i c c o l i   g a t t i   d o r m i v a n o   p r o f o n d a m e n t e   s u l   t a p p e t o` |

### 🧲 The Key Insight: Real-Time "Magnetic" Agreement

Look at the core noun `gatt_` (selected at Step 1): it holds its final inflectional letter (`_`: will it be singular `-o` or plural `-i`?) in suspension through Steps 1, 2, and 3a while the rest of the syntactic skeleton forms around it.

Once every syntactic slot is active on the canvas, **Cross-Word Bidirectional Character Attention** acts like a magnetic field at Step 3b, snapping all inflectional ending letters into place simultaneously:

- **Masculine Plural Agreement Chain:** `I` $`\leftrightarrow`$ `piccol-i` $`\leftrightarrow`$ `gatt-i` $`\leftrightarrow`$ `dormiva-no`
- **Prepositional Fusion (Masculine Singular):** The articulated preposition `sul` (*su + il*, "on + the") first emerges as spatial root letters (`s u _`) at Step 3a, and an instant later fuses with the masculine singular noun `tappet-o` to become `s u l` at Step 3b.

---

## 📐 Mathematical Formulation

Let a sentence canvas be represented as a 2D discrete character tensor $`\mathbf{X}_t \in \mathcal{V}_{\text{char}}^{W \times C}`$, where $`\mathcal{V}_{\text{char}}`$ is the single-letter vocabulary, $`W`$ is the maximum number of words, and $`C`$ is the maximum number of letters per word.

Each word slot $`i \in \{1, \dots, W\}`$ has a **Macro Priority Tier** $`M_i \in \{0, 1, 2, 3\}`$ (where $`0`$ denotes an inactive slot for shorter sentences), and each letter slot $`(i, j)`$ has a **Micro Morphological Tier** $`\mu_{i,j} \in \{0, 1, 2, 3\}`$:

- $`\mu_{i,j} = 1`$: Lexical Root letter (e.g., `g-a-t-t-`, `d-o-r-m-`, `p-i-c-c-o-l-`, `s-u-`)
- $`\mu_{i,j} = 2`$: Internal Thematic / Derivational Affix letter (e.g., imperfect `-i-v-a-`, adverbial `-a-m-e-n-t-e`)
- $`\mu_{i,j} = 3`$: Relational Inflectional Ending / Agreement letter (e.g., `I`, `-i`, `-n-o`, `-l`, `-o`)

At discrete macro-time $`t \in \{1, 2, 3\}`$, the **local age** of word $`i`$ on the canvas is:

```math
\tau_i(t) = \max(0,\, t - M_i + 1)
```

The visibility mask of any letter $`(i, j)`$ is governed by two distinct physical regimes:

- **Word Activation (Macro Axis):** Word $`i`$ is active on the canvas iff $`M_i > 0`$ and $`\tau_i(t) \ge 1`$ (otherwise masked as `<WORD_MASK>`, rendered as `[...]`).
- **Internal Morphology (Local Time Regime, $`\mu_{i,j} \in \{1, 2\}`$):** Letter $`(i, j)`$ is unmasked iff $`\tau_i(t) \ge \mu_{i,j}`$.
- **Morphosyntactic Agreement (Global Phase Synchronization, $`\mu_{i,j} = 3`$):** Inflectional ending letters remain masked as `<CHAR_MASK>` (`_`) until the full syntactic skeleton is active ($`t = T_{\max}`$) and the global agreement phase is triggered.

---

## 🎓 AI Training Phase & Progressive Priority Generation

To allow an AI to generate arbitrary sentences letter-by-letter without predefined oracle masks at inference time, `fractal_generative_engine.py` implements a three-part training and inference pipeline:

1. **Automated Letter-by-Letter Decomposition (`MorphosyntacticAnnotator`):**
   During training preparation, raw sentences of varying lengths are automatically tokenized character-by-character into ground-truth letter grids $`\mathbf{X}_{\text{clean}} \in \mathcal{V}_{\text{char}}^{W \times C}`$, word priority tiers $`\mathbf{M} \in \{0, 1, 2, 3\}^W`$, and per-letter morphological tiers $`\boldsymbol{\mu} \in \{0, 1, 2, 3\}^{W \times C}`$.

2. **Multi-Task Endogenous Planning Objective:**
   At each diffusion step $`s`$, `AutonomousFractalWaveModel` processes the noisy letter canvas $`\mathbf{X}_s`$ through its Vertical (`micro_transformer`) and Horizontal (`macro_transformer`) attention stacks, jointly optimizing three prediction heads:

   ```math
   \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}\!\left(\hat{\mathbf{X}}, \mathbf{X}_{\text{clean}}\right) + \lambda_{M}\,\mathcal{L}_{\text{CE}}\!\left(\hat{\mathbf{M}}, \mathbf{M}\right) + \lambda_{\mu}\,\mathcal{L}_{\text{CE}}\!\left(\hat{\boldsymbol{\mu}}, \boldsymbol{\mu}\right)
   ```

   where $`\hat{\mathbf{X}}`$ predicts the probability distribution over single letters, $`\hat{\mathbf{M}}`$ predicts the word priority tier $`\hat{M}_i \in \{0, 1, 2, 3\}`$, and $`\hat{\boldsymbol{\mu}}`$ predicts the per-letter morphological tier $`\hat{\mu}_{i,j} \in \{0, 1, 2, 3\}`$.

3. **Progressive Anchor-First Inference (`generate_autonomous`):**
   - **Step 0 $`\to`$ Step 1:** The AI commits only the Core Pillars ($`\hat{M}_i = 1`$) and generates their root letters ($`\hat{\mu}_{i,j} = 1`$), leaving all other slots as unborn `[...]` tokens.
   - **Step 1 $`\to`$ Step 2 $`\to`$ Step 3a $`\to`$ Step 3b:** Trained with a Canvas-Driven unconditioned pass (`prompt_char_ids=None` for $`s \ge 1`$), the AI reads the Pillar root letters already on the canvas to progressively select the Modifiers ($`\hat{M}_i = 2`$) at Step 2, the Connectives ($`\hat{M}_i = 3`$) at Step 3a, and snap all inflectional agreement letters ($`\hat{\mu}_{i,j} = 3`$) at Step 3b.

---

## 🚀 Quickstart

### 1. Clone the repository and install dependencies

```bash
git clone https://github.com/BerryFlavor101/fractal-wave-diffusion.git
cd fractal-wave-diffusion
pip install -r requirements.txt
```

### 2A. Run the Paper Benchmark Showcase (`fractal_diffusion.py`)

```bash
python fractal_diffusion.py
```

### 2B. Run the Autonomous Letter-by-Letter Generative AI Engine (`fractal_generative_engine.py`)

```bash
python fractal_generative_engine.py
```

---

## 📖 Citation

If you find this work useful in your research, please cite the accompanying paper:

```bibtex
@article{golshahi2026fractal,
  title={Fractal Wave Text Diffusion: Hierarchical Two-Axis Coarse-to-Fine Generation via Macro-Syntactic and Micro-Morphological Synchronization},
  author={Golshahi, Arvin},
  year={2026},
  month={11},
  publisher={Zenodo},
  doi={10.5281/zenodo.24209687},
  url={https://zenodo.org/records/24209687}
}
```

---

## 🤖 AI Assistance & EU Transparency Disclosure

In accordance with European Union transparency guidelines on Artificial Intelligence (EU AI Act) and academic publishing standards:

> This paper and its accompanying software implementation were organized, formatted, and structured with the assistance of artificial intelligence tools; however, the core theoretical framework, underlying logic, architectural reasoning, analyses, and scientific conclusions are the original intellectual work of the author (Arvin Golshahi).

---

## 📜 License & Copyright

Copyright (c) 2026 Arvin Golshahi. All intellectual rights are retained by the author.

- **Source Code** (`fractal_diffusion.py`, `fractal_generative_engine.py`): Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) — see the [LICENSE](LICENSE) file for details.
- **Research Paper:** Licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) License ([Zenodo Record](https://zenodo.org/records/24209687)).
