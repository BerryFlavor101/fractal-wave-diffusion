# 🌊 Fractal Wave Text Diffusion
**Hierarchical Two-Axis Discrete Text Diffusion (Macro-Syntactic & Micro-Morphological) in PyTorch.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24209687.svg)](https://doi.org/10.5281/zenodo.24209687)
[![Zenodo Record](https://img.shields.io/badge/Zenodo-Record_24209687-blue.svg)](https://zenodo.org/records/24209687)
[![Code License: AGPL v3](https://img.shields.io/badge/Code_License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Paper License: CC BY 4.0](https://img.shields.io/badge/Paper_License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Author:** Arvin Golshahi (Independent researcher)  
**Zenodo Publication:** [https://zenodo.org/records/24209687](https://zenodo.org/records/24209687) | **DOI:** [10.5281/zenodo.24209687](https://doi.org/10.5281/zenodo.24209687)

Unlike traditional Autoregressive Language Models (which generate text linearly from left to right) or standard Discrete Diffusion Models (which unmask random tokens uniformly across a sequence), **Fractal Wave Text Diffusion** generates natural language by **order of informational priority**.

It applies a **fractal coarse-to-fine logic** simultaneously across two orthogonal axes: the **Sentence Axis (Macro-Diffusion)** and the **Word Axis (Micro-Diffusion)**.

---

## 🧠 Core Theory: The Two Orthogonal Axes

Language in this framework does not grow from left to right; it crystallizes from the semantic core outward toward the grammatical periphery on two levels at once:

1. **Horizontal Axis — Macro-Diffusion (Sentence Level):** *From "Caveman" telegraphic pillars to full syntax.*
   - **Step 1 (High Priority — Pillars):** Core semantic concepts appear first (subject noun, main verb, object noun).
   - **Step 2 (Medium Priority — Modifiers):** Adjectives and adverbs are inserted around the pillars.
   - **Step 3 (Low Priority — Connectives):** Functional glue words emerge last (articles, prepositions, conjunctions).

2. **Vertical Axis — Micro-Diffusion (Word Level):** *From lexical root to inflectional endings.*
   - Every word—whether a core pillar born at Step 1 or a functional preposition born at Step 3—crystallizes from the inside out: first its **lexical root** ($\mu=1$), then its **internal thematic/derivational affixes** ($\mu=2$), and finally its **relational inflectional endings** ($\mu=3$).

---

## ⏳ The "Wave Effect" & Real-Time Morphological Agreement

Combining these two axes creates a **temporal wave of crystallization**: words do not finish forming at the same time. While Step 1 pillars are already growing their internal thematic suffixes at Step 2, Step 2 modifiers are just beginning to expose their lexical roots.

### Showcase Example (Italian Morphosyntax Testbed)
To clearly observe morphological agreement across words, we use the Italian sentence:  
> **"I piccoli gatti dormivano profondamente sul tappeto"**  
> *(Literal: "The [masc.pl] small [masc.pl] cats [masc.pl] slept/were-sleeping [3rd.pl] deeply on-the [masc.sg] rug [masc.sg]")*

*(Note: Richly inflected languages like Italian make the micro-diffusion wave visually explicit in ways English cannot, since English adjectives and articles like "the small" do not inflect for gender or number).*

| Diffusion Step | Macro-Diffusion (Sentence Axis) | Micro-Diffusion (Character Axis) | Visible Output in the "Text Construction Site" (Canvas) |
| :--- | :--- | :--- | :--- |
| **Step 0** | Empty syntactic canvas. | Full noise. | `[...]   [...]   [...]   [...]   [...]   [...]   [...]` |
| **Step 1** | Places **Core Pillars** (High Priority). | Sculpts lexical roots of core pillars. | `[...]   [...]   g a t t _   d o r m _ _ _ _ _   [...]   [...]   t a p p e t _` |
| **Step 2** | Inserts **Modifiers** (Medium Priority: adjectives/adverbs). | Pillars expand thematic affixes (`-iva-`); modifiers reveal their roots. | `[...]   p i c c o l _   g a t t _   d o r m i v a _ _   p r o f o n d _ _ _ _ _ _   [...]   t a p p e t _` |
| **Step 3a** | Inserts **Connectives** (Low Priority: articles/prepositions). | Modifiers complete internal suffixes (`-amente`); connective roots (`su_`) appear; all agreement endings wait. | `_   p i c c o l _   g a t t _   d o r m i v a _ _   p r o f o n d a m e n t e   s u _   t a p p e t _` |
| **Step 3b** | **Global Agreement Snap.** | All inflectional endings crystallize simultaneously across the sentence. | `I   p i c c o l i   g a t t i   d o r m i v a n o   p r o f o n d a m e n t e   s u l   t a p p e t o` |

### 🧲 The Key Insight: Real-Time "Magnetic" Agreement
Look at the core noun `gatt_` (born at Step 1): it holds its final inflectional character (`_`: will it be singular `-o` or plural `-i`?) in suspension through Steps 1, 2, and 3a while the rest of the syntactic skeleton forms.

Once every syntactic slot is active on the canvas, **Cross-Word Bidirectional Attention** acts like a magnetic field at Step 3b, snapping all inflectional endings into place simultaneously:
- **Masculine Plural Agreement Chain:** `I` $\leftrightarrow$ `piccol-i` $\leftrightarrow$ `gatt-i` $\leftrightarrow$ `dormiva-no`
- **Prepositional Fusion (Masculine Singular):** The articulated preposition `sul` (*su + il*, "on + the") first emerges as a spatial root (`s u _`) at Step 3a, and an instant later fuses with the masculine singular noun `tappet-o` to become `s u l` at Step 3b.

---

## 📐 Mathematical Formulation

Let a sentence canvas be represented as a 2D discrete tensor $\mathbf{X}_t \in \mathcal{V}^{W \times C}$, where $W$ is the maximum number of words and $C$ is the maximum character length per word.
Each word $i \in \{1, \dots, W\}$ has a **Macro Priority Tier** $M_i \in \{1, 2, 3\}$, and each character $(i, j)$ has a **Micro Morphological Tier** $\mu_{i,j} \in \{1, 2, 3\}$:
- $\mu_{i,j} = 1$: Lexical Root (e.g., `gatt-`, `dorm-`, `piccol-`, `su-`)
- $\mu_{i,j} = 2$: Internal Thematic / Derivational Affix (e.g., imperfect tense `-iva-`, adverbial `-amente`)
- $\mu_{i,j} = 3$: Relational Inflectional Ending / Agreement Marker (e.g., `I`, `-i`, `-no`, `-l`, `-o`)

At discrete macro-time $t \in \{1, 2, 3\}$, the **local age** of word $i$ on the canvas is:

$$\tau_i(t) = \max(0,\, t - M_i + 1)$$

The visibility mask of any character $(i, j)$ is governed by two distinct physical regimes:
1. **Word Activation (Macro Axis):** Word $i$ is active on the canvas iff $\tau_i(t) \ge 1$ (otherwise masked as `<WORD_MASK>`, rendered as `[...]`).
2. **Internal Morphology (Local Time Regime, $\mu_{i,j} \in \{1, 2\}$):** Character $(i, j)$ is unmasked iff $\tau_i(t) \ge \mu_{i,j}$.
3. **Morphosyntactic Agreement (Global Phase Synchronization, $\mu_{i,j} = 3$):** Inflectional endings remain masked as `<CHAR_MASK>` (`_`) until the full syntactic skeleton is active ($t = T_{\max}$) and the global agreement phase is triggered.

---

## 🏗️ Neural Architecture (`FractalWaveTransformer`)

The PyTorch model processes the 2D `(Words, Chars)` canvas using a two-stage **Axial & Magnetic Transformer**:
1. **Vertical Micro-Transformer (Intra-Word Attention):** Operates along the character axis $C$ within each independent word to expand lexical roots into thematic stems.
2. **Horizontal Macro-Transformer (Cross-Word Agreement Attention):** Operates across the flattened $(W \times C)$ sequence so that active roots and inflectional `_` slots exchange gender, number, and person features across the entire sentence.

---

## 🚀 Quickstart

### 1. Clone the repository and install dependencies
```bash
git clone https://github.com/BerryFlavor101/fractal-wave-diffusion.git
cd fractal-wave-diffusion
pip install -r requirements.txt
```

### 2. Run the training & reverse-diffusion generation demo
```bash
python fractal_diffusion.py
```

### Expected Terminal Output
```text
====================================================================================================
 TEXT CONSTRUCTION SITE: FRACTAL WAVE REVERSE DIFFUSION (FROM PURE NOISE TO SENTENCE)
====================================================================================================
[Step 0  | Initial Noise          ] -> [...]   [...]   [...]   [...]   [...]   [...]   [...]
[Step 1  | Core Pillars (Roots)   ] -> [...]   [...]   g a t t _   d o r m _ _ _ _ _   [...]   [...]   t a p p e t _
[Step 2  | Modifiers & Stems      ] -> [...]   p i c c o l _   g a t t _   d o r m i v a _ _   p r o f o n d _ _ _ _ _ _   [...]   t a p p e t _
[Step 3a | Connective Nuclei      ] -> _   p i c c o l _   g a t t _   d o r m i v a _ _   p r o f o n d a m e n t e   s u _   t a p p e t _
[Step 3b | Magnetic Agreement Snap] -> I   p i c c o l i   g a t t i   d o r m i v a n o   p r o f o n d a m e n t e   s u l   t a p p e t o
====================================================================================================
Reconstruction Accuracy on Unmasked Canvas: 100.00%
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
> This paper and its accompanying software implementation were organized, formatted, and structured with the assistance of artificial intelligence tools; however, the core theoretical framework, underlying logic, architectural reasoning, analyses, and scientific conclusions are the original intellectual work of the author (**Arvin Golshahi**).

---

## 📜 License & Copyright
Copyright (c) 2026 **Arvin Golshahi**. All intellectual rights are retained by the author.
- **Source Code (`fractal_diffusion.py`):** Licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** — see the [LICENSE](./LICENSE) file for details.
- **Research Paper:** Licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** License ([Zenodo Record](https://zenodo.org/records/24209687)).
