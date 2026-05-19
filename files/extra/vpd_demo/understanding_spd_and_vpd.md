# Understanding Parameter Decomposition: SPD → VPD

A step-by-step guide with tensor shapes and intuition for understanding
Stochastic Parameter Decomposition (SPD) and adVersarial Parameter Decomposition (VPD).

---

## Part 1: Stochastic Parameter Decomposition (SPD)

### The Big Picture

We have a **trained, frozen** neural network. We want to understand *what mechanisms* its weight
matrices implement. The idea: break each weight matrix into simple rank-1 pieces ("subcomponents"),
figure out which pieces are actually used for each input, and verify that unused pieces can be safely removed.

SPD does not retrain or improve the model. It *reverse-engineers* it.

### Running Example

```
Model:  1-layer residual MLP
d_in    = 768    (residual stream / input dimension)
d_mlp   = 3072   (MLP hidden dimension)
C       = 9728   (subcomponents per matrix — intentionally more than rank)
B       = 4      (batch size)
```

The model computes:
```
y = x + MLP(x)
  = x + GELU(x @ W_in.T) @ W_out.T
```

The weight matrices we want to decompose:
```
W_in:   (3072, 768)    — MLP up-projection
W_out:  (768, 3072)    — MLP down-projection
```

These weights are **frozen**. SPD never changes them.

---

### Step 1: Decompose Each Weight Matrix into Rank-1 Subcomponents

Each weight matrix is written as a sum of outer products:

```
W_in ≈ Σ_c  u_c ⊗ v_c^T       for c = 1..C
```

We store these as two matrices:

```
U_in:  (3072, 9728)     — each column u_c is a vector on the output side
V_in:  (9728, 768)      — each row v_c is a vector on the input side

Reconstruction:  U_in @ V_in = (3072, 9728) @ (9728, 768) = (3072, 768) ≈ W_in
```

Similarly for W_out:
```
U_out: (768, 9728)
V_out: (9728, 3072)

Reconstruction:  U_out @ V_out = (768, 9728) @ (9728, 3072) = (768, 3072) ≈ W_out
```

Each **individual subcomponent c** is the rank-1 matrix:
```
subcomponent_c = u_c ⊗ v_c^T = U[:, c] @ V[c, :].unsqueeze(0)
Shape: (3072, 1) @ (1, 768) = (3072, 768)   — same shape as W_in, but rank 1
```

**Intuition:** Think of v_c as "what this subcomponent reads from the input" and u_c as "what it
writes to the output." Each subcomponent is one simple read-write channel through the weight matrix.

**Why C > rank of the matrix?** The matrix rank is at most 768, but we allow 9728 subcomponents.
This is because the model may encode more mechanisms than it has dimensions — this is called
**superposition**. More subcomponents than dimensions lets us find those mechanisms.

---

### Step 2: Compute Causal Importance Values

For each input, we need to know: which of the 9728 subcomponents are actually needed?

**Inner activation** — how much does the input project onto each subcomponent's reading direction:

```
x:    (B, 768)         — input activation (what enters this weight matrix)
V_in: (9728, 768)      — each row is a reading direction

h_in = x @ V_in.T      — shape: (B, 9728)
```

`h_in[b, c]` is a scalar: "how much does input b align with subcomponent c's reading direction."

**Causal importance prediction** — a tiny MLP per subcomponent turns h into a gate:

```
For each subcomponent c:
  MLP_c: scalar → scalar     (hidden dim ≈ 16, GELU activation, hard sigmoid at end)

g_in[b, c] = hard_sigmoid(MLP_c(h_in[b, c]))
```

```
g_in: (B, 9728)         — one importance value per subcomponent per batch element
                            each value ∈ [0, 1]
```

Similarly for W_out using its own V_out and the post-ReLU activations:
```
g_out: (B, 9728)
```

**What the values mean:**
```
g[b, c] = 0.0  →  subcomponent c is not needed for input b, ablate freely
g[b, c] = 1.0  →  subcomponent c is critical for input b, don't touch it
g[b, c] = 0.3  →  partially needed, can reduce but not below 30%
```

**Each matrix has its own independent set of MLPs.** Subcomponent 5 of W_in has nothing to do
with subcomponent 5 of W_out — they are separate decompositions linked only by the shared output.

---

### Step 3: Stochastic Masking — The Core SPD Trick

For each subcomponent, we sample a random mask:

```
r_in:  (B, 9728)     ~ Uniform(0, 1)                    — random scalars
m_in:  (B, 9728)     = g_in + (1 - g_in) * r_in         — lives in [g_in, 1]
```

Similarly for W_out:
```
r_out: (B, 9728)     ~ Uniform(0, 1)
m_out: (B, 9728)     = g_out + (1 - g_out) * r_out
```

**The mask range [g, 1] is the key design choice:**

| g_c value | mask m_c range | what happens |
|-----------|----------------|-------------|
| g = 1 (important) | m = 1 always | fully protected, never ablated |
| g = 0 (unimportant) | m ~ Uniform(0, 1) | randomly scaled anywhere from 0 to full |
| g = 0.3 (partial) | m ~ Uniform(0.3, 1) | partially protected, still wiggled |

**The fundamental insight:** If a subcomponent is truly unneeded, the output shouldn't change no
matter what random scaling we apply. By randomly wiggling unimportant subcomponents and demanding
the output stays the same, we are probabilistically checking many ablation combinations.

**Why this helps gradients:** Every subcomponent gets gradients on every training step, regardless of
its g value. Even with g = 0, the mask m is a random nonzero value, so the subcomponent contributes
something to the forward pass and gradients flow through. If the causal importance function is wrong
about something being unimportant, the reconstruction loss will signal the error, and gradients will flow
back to fix both the subcomponent parameters and the gate MLP.

This is a major advantage over APD (the predecessor), which used hard top-k selection — if a
subcomponent wasn't in the top-k, it got zero gradients and could never self-correct.

---

### Step 4: Build Masked Weights and Run Forward Pass

For each batch element b, build the masked weight matrix:

```
m_in[b]:  (9728,)
U_in:     (3072, 9728)

U_scaled = U_in * m_in[b]               — broadcast: (3072, 9728) * (9728,)
W'_in[b] = U_scaled @ V_in              — (3072, 9728) @ (9728, 768) = (3072, 768)
```

This is equivalent to `U_in @ diag(m_in[b]) @ V_in` but computed without forming the diagonal matrix.

**What it does to each subcomponent:**
```
subcomponent c contributes:  m_in[b, c] * (u_c @ v_c^T)

m = 1.0  →  full contribution
m = 0.0  →  completely ablated
m = 0.6  →  60% of its contribution
```

Run the full forward pass with masked weights:
```
hidden     = x @ W'_in[b].T              — (768,) @ (768, 3072) = (3072,)
hidden_act = GELU(hidden)                — (3072,)
mlp_out    = hidden_act @ W'_out[b].T    — (3072,) @ (3072, 768) = (768,)
y_masked   = x + mlp_out                 — (768,)     (residual connection)
```

Simultaneously, run the **original** model to get the target:
```
y_target from f(x, W_in, W_out)         — with original unmasked weights
```

---

### Step 5: Compute Losses and Update

**Loss 1 — Faithfulness:** Subcomponents should sum to the original weights.
```
L_faith = MSE(U_in @ V_in, W_in) + MSE(U_out @ V_out, W_out)     — scalar
```

**Loss 2 — Stochastic Reconstruction:** Masked output should match target.
```
L_stoch_recon = D(y_masked, y_target)        — KL divergence or MSE, scalar
                                                (repeat S times with fresh r, average)
```

**Loss 2b — Layerwise Stochastic Reconstruction:** Mask only one layer at a time, keep other
layers at original weights. This gives a less noisy training signal.
```
L_stoch_layerwise = (1/L) Σ_l  D(y_masked_layer_l, y_target)     — scalar
```

**Loss 3 — Importance Minimality:** Push all g values toward 0.
```
L_minimal = Σ_b Σ_c |g[b,c]|^p      — scalar, summed over batch and subcomponents
```

**Total SPD loss:**
```
L_SPD = L_faith + β1 * L_stoch_recon + β2 * L_stoch_layerwise + β3 * L_minimal
```

**The tension that makes SPD work:**

- L_stoch_recon wants g = 1 everywhere (protect everything → perfect reconstruction)
- L_minimal wants g = 0 everywhere (maximum sparsity)
- Equilibrium: g = 1 only for subcomponents that genuinely break output when ablated

**Trainable parameters (updated via gradient descent on L_SPD):**
```
U_in, V_in:    (3072, 9728) and (9728, 768)     — subcomponent vectors
U_out, V_out:  (768, 9728) and (9728, 3072)
MLP params:    one tiny MLP per subcomponent      — causal importance predictors
```

**Frozen (never updated):**
```
W_in, W_out:   original model weights
```

---

## Part 2: Why SPD Is Not Enough

SPD works well on toy models, but stochastic masking has a fundamental blind spot when applied
to real models. Let's see exactly why.

### The Problem: Correlated Cancellation

**Ground truth:** Subcomponent A adds +3 to some output logit. Subcomponent B adds -3 to the
same logit. They cancel out. Both are genuinely doing work — they're part of a real mechanism.
Their g values **should** be high (close to 1).

But the causal importance function has mistakenly set g_A = 0 and g_B = 0. It thinks both are
unimportant.

### What Stochastic Masking Does

We sample random masks:
```
m_A ~ Uniform(0, 1),  say we get m_A = 0.6
m_B ~ Uniform(0, 1),  say we get m_B = 0.4

Output contribution: 0.6 * (+3) + 0.4 * (-3) = 1.8 - 1.2 = +0.6
Target contribution: +3 + (-3) = 0
Error: 0.6   (small)
```

Next training step, different random draw:
```
m_A = 0.3,  m_B = 0.7

Output contribution: 0.3 * (+3) + 0.7 * (-3) = 0.9 - 2.1 = -1.2
Error: 1.2   (moderate)
```

On average across many samples, m_A and m_B both center around 0.5, and the errors tend to be
**moderate**. The reconstruction loss is elevated but not screaming. The gradient signal back to g_A and
g_B is **weak** — it says "maybe increase g a little" but not urgently.

Training might slowly fix it, or it might settle into a bad equilibrium where g stays near 0 and
the moderate noise is just tolerated.

### The Gap

The worst case would be m_A = 1.0 and m_B = 0.0, giving an error of +3. Or m_A = 0.0 and
m_B = 1.0, giving an error of -3. But Uniform(0, 1) is very unlikely to sample exactly these extremes.
It mostly samples things in the middle where A and B still partially cancel.

**The loss never spikes hard enough to create a strong gradient signal saying "these are important!"**

The decomposition settles into a state where it **looks** sparse (g ≈ 0 for both) and **looks** like it
reconstructs okay (moderate average loss), but it's wrong — it's hiding real mechanisms behind a
cancellation that random masking can't catch.

### What We Need

Instead of hoping random sampling finds the bad combination, we need to **actively search** for it.
That's what adversarial masking does.

---

## Part 3: adVersarial Parameter Decomposition (VPD)

VPD keeps the entire SPD framework (rank-1 subcomponents, causal importance function, stochastic
masking) and adds three things:

1. **Adversarial masking** — actively search for worst-case ablations
2. **Frequency-minimality loss** — penalize subcomponents that fire on too many inputs
3. **Δ-components** — explicit residual for guaranteed parameter faithfulness

### Running Example (Real Transformer)

```
Model:   4-layer decoder-only transformer
d_model  = 768
d_mlp    = 3072
n_heads  = 6
d_head   = 128
vocab    = 50277
T        = 512         (sequence length)
C        = 9728        (subcomponents per weight matrix)
B        = batch size

Weight matrices to decompose (per layer):
  Attention:  W_Q (768, 768), W_K (768, 768), W_V (768, 768), W_O (768, 768)
  MLP:        W_in (3072, 768), W_out (768, 3072)

  Total: 6 matrices × 4 layers = 24 matrices
```

All original weights are **frozen**. VPD reverse-engineers them.

---

### VPD Step 0: Target Forward Pass

Run the frozen original model to get the target output:

```
tokens:    (B, T)               — input token sequences
y_target:  (B, T, vocab)        — target model's output logits / probabilities
```

This is our ground truth for every reconstruction comparison.

---

### Critical: All 24 Matrices Are Masked Simultaneously

This is an important point that's easy to miss. When VPD does a masked forward pass (whether
stochastic or adversarial), it doesn't mask one matrix at a time. **All 24 weight matrices are
replaced with their masked versions in the same single forward pass.**

Each matrix is independently decomposed — it has its own U, V, Δ, its own 9728 subcomponents,
its own causal importance MLPs, and its own masks. Subcomponent 42 of W_Q in layer 0 has
nothing to do with subcomponent 42 of W_K in layer 0. But they're all plugged into the same
forward pass:

```
x = embed(tokens)

Layer 0:
  x = x + Attention(x, W'_Q0, W'_K0, W'_V0, W'_O0)   ← all 4 matrices masked
  x = x + MLP(x, W'_in0, W'_out0)                      ← both matrices masked

Layer 1:
  x = x + Attention(x, W'_Q1, W'_K1, W'_V1, W'_O1)   ← all 4 matrices masked
  x = x + MLP(x, W'_in1, W'_out1)                      ← both matrices masked

Layers 2, 3: same

logits = x @ W_unembed                                 ← NOT decomposed
y_masked: (B, T, vocab)
```

Then the reconstruction loss compares:
```
y_target:  original model with original weights    (B, T, vocab)
y_masked:  model with ALL 24 matrices masked       (B, T, vocab)

L_recon = KL(y_target, y_masked)
```

This makes VPD an **end-to-end** method — it checks whether the decomposition preserves the
model's **final behavior**, not just intermediate activations at one layer. That's a key advantage
over per-layer methods like transcoders, which only match activations layer by layer.

(Note: SPD had a layerwise loss variant where you mask only one layer at a time. VPD dropped
that and relies on full end-to-end masking instead.)

---

### VPD Step 1: Decompose Each Weight Matrix

Same as SPD, but with an explicit Δ residual. For each of the 24 weight matrices, e.g. W_in of
shape (3072, 768):

```
U:  (3072, 9728)      — output-side vectors (trainable)
V:  (9728, 768)       — input-side vectors (trainable)
Δ:  (3072, 768)       — residual: Δ = W_in - U @ V (derived, not separately trainable)
```

The effective weight matrix is always:
```
W_effective = U @ V + Δ = W_in     — exactly equals original, by construction
```

This means **parameter faithfulness is guaranteed**, not approximately optimized. Δ starts large
(before U @ V has learned anything) and shrinks during training as U @ V gets better. An L2
penalty pushes Δ toward zero.

Δ is always treated as fully ablatable: g = 0, no MLP, no learned importance.

---

### VPD Step 2: Compute Causal Importances

Now with a sequence dimension (the key difference from toy-model SPD):

```
x:    (B, T, 768)        — residual stream activations entering this matrix
V:    (9728, 768)

h = x @ V.T               — shape: (B, T, 9728)

For each subcomponent c:
  g[b,t,c] = hard_sigmoid(MLP_c(h[b,t,c]))

g: (B, T, 9728)            — per subcomponent, per position, per batch element
```

Subcomponent c might be important at position 5 ("the") but unimportant at position 12 ("and").

This means the masks, and therefore the effective weight matrices, will be **different for every
(batch, position) pair**:

```
W'[b=0, t=0]:  (3072, 768)    — masked weights for batch 0, token "The"
W'[b=0, t=1]:  (3072, 768)    — different masked weights for batch 0, token "princess"
W'[b=1, t=0]:  (3072, 768)    — different again for batch 1, first token
```

---

### VPD Step 3: Generate Both Mask Types

**Stochastic masks (same as SPD):**
```
r_stoch:  (B, T, 9728)     ~ Uniform(0, 1)
m_stoch:  (B, T, 9728)     = g + (1 - g) * r_stoch       — lives in [g, 1]
```

**Adversarial masks (new in VPD):**

Initialize m_adv randomly in [g, 1], then run gradient ascent **on the masks only**:

```
m_adv: (B, T, 9728)        — initialized in [g, 1], one per matrix (×24 matrices)

for step in range(20):      — typically ~20 adversarial steps
    (see "Adversarial Inner Loop Forward Pass" below for full details)
    
    loss = KL(y_target, y_adv)
    grad = d(loss) / d(m_adv)             — gradient w.r.t. masks only
    m_adv = m_adv + lr_adv * grad         — gradient ASCENT (maximize loss)
    m_adv = clamp(m_adv, min=g, max=1)    — stay in allowed range
```

**During this inner loop, ONLY m_adv changes.** U, V, Δ, all MLPs are frozen.

---

### VPD Step 3a: The Adversarial Inner Loop Forward Pass (detailed)

Inside each step of the adversarial search loop, a full model forward pass happens.
Here is exactly what occurs, with tensor shapes, for one step of the inner loop.

**We have the current m_adv values for all 24 matrices.** Let's trace through one layer.

**Building masked weights for MLP W_in (layer 0):**
```
m_adv_in:  (B, T, 9728)         — current adversarial mask for this matrix
U_in:      (3072, 9728)          — frozen during inner loop
V_in:      (9728, 768)           — frozen during inner loop
Δ_in:      (3072, 768)           — Δ = W_in - U_in @ V_in, frozen

For Δ, since g = 0 always:
  m_delta_adv: (B, T)            — also being adversarially optimized, in [0, 1]

For each batch b, position t:
  U_scaled       = U_in * m_adv_in[b, t]          — (3072, 9728) * (9728,) broadcast
  W'_in[b, t]    = U_scaled @ V_in                 — (3072, 9728) @ (9728, 768) = (3072, 768)
                   + Δ_in * m_delta_adv[b, t]       — (3072, 768) * scalar
```

**Repeat for all 24 matrices:** W_Q, W_K, W_V, W_O, W_in, W_out for each of 4 layers.

**Run the full transformer forward pass with masked weights:**
```
tokens:      (B, T)

Embedding:   (B, T, 768)         — from token embedding (not decomposed)

Layer 0 Attention:
  x:         (B, T, 768)         — residual stream input
  q:         (B, T, 768)         = x @ W'_Q[b,t].T       — per-position masked W_Q
  k:         (B, T, 768)         = x @ W'_K[b,t].T       — per-position masked W_K
  v:         (B, T, 768)         = x @ W'_V[b,t].T
  attn:      standard attention computation with q, k, v
  attn_out:  (B, T, 768)         = attn_result @ W'_O[b,t].T
  x:         (B, T, 768)         = x + attn_out           — residual

Layer 0 MLP:
  hidden:    (B, T, 3072)        = x @ W'_in[b,t].T
  activated: (B, T, 3072)        = GELU(hidden)
  mlp_out:   (B, T, 768)         = activated @ W'_out[b,t].T
  x:         (B, T, 768)         = x + mlp_out            — residual

Layers 1, 2, 3: same pattern

Unembedding: (B, T, vocab)       — x @ W_unembed (not decomposed)

y_adv:       (B, T, vocab)       — output with adversarially masked weights
```

**Compute loss and update masks:**
```
loss = KL(y_target, y_adv)                        — scalar

grad = d(loss) / d(m_adv for all 24 matrices)     — (B, T, 9728) per matrix

m_adv = m_adv + lr_adv * grad                     — gradient ASCENT
m_adv = clamp(m_adv, min=g, max=1)                — respect importance floor
```

Only m_adv values change. The adversary is searching for the worst ablation pattern.

After ~20 such steps, m_adv holds the worst-case masks for all 24 matrices.

---

### VPD Step 4: Outer Training Loop Forward Passes

Now we use both mask types for the actual training. Two separate forward passes, never mixed.

**Forward Pass A — Stochastic Masks:**

Build masked weights for all 24 matrices using m_stoch:
```
For each matrix l (e.g. W_in of layer 0):
  m_stoch_l:   (B, T, 9728)
  U_scaled     = U_l * m_stoch_l[b, t]            — (d_out, 9728) * (9728,)
  W'_stoch_l   = U_scaled @ V_l + Δ_l * m_delta_stoch

Run full transformer forward pass (same structure as inner loop above):
  tokens → embedding → 4 layers of masked attn + MLP → unembedding

y_stoch: (B, T, vocab)
```

**Forward Pass B — Adversarial Masks:**

Build masked weights for all 24 matrices using m_adv (from Step 3):
```
For each matrix l:
  m_adv_l:     (B, T, 9728)        — the worst-case masks found by adversarial search
  U_scaled     = U_l * m_adv_l[b, t]
  W'_adv_l     = U_scaled @ V_l + Δ_l * m_delta_adv

Run full transformer forward pass:

y_adv: (B, T, vocab)
```

Note: this forward pass with m_adv is the **same** as the final iteration of the inner loop.
It may be reused rather than recomputed.

---

### VPD Step 5: Compute All 5 Losses

**Loss 1 — Adversarial Reconstruction** (new in VPD)
```
L_adv_recon = KL(y_target, y_adv)                              — scalar
```
"Even under the worst-case ablation of unimportant subcomponents, output should match."

**Loss 2 — Stochastic Reconstruction** (same as SPD)
```
L_stoch_recon = KL(y_target, y_stoch)                           — scalar
```

**Loss 3 — Importance Minimality** (same as SPD)
```
g: (B, T, 9728) per matrix, across all 24 matrices

L_importance = (1/BT) Σ_b Σ_t Σ_l Σ_c |g[b,t,c]|^p            — scalar
```
"Mark as few subcomponents as important as possible."

**Loss 4 — Frequency Minimality** (new in VPD)
```
For each subcomponent c in matrix l:
  freq_c = Σ_b' Σ_t' |g[b',t',c]|^p       — how often c fires across the batch

L_freq = (1/BT) Σ_b Σ_t Σ_l Σ_c  |g[b,t,c]|^p * log2(1 + freq_c)    — scalar
```
"Don't just fire less — especially avoid being the subcomponent that fires on everything."
The log multiplier makes the per-activation penalty grow for high-frequency subcomponents,
encouraging them to split into more specialized, less frequent pieces.

**Loss 5 — Delta L2** (new in VPD)
```
L_delta = Σ_l ||W_l - U_l @ V_l||²                              — scalar
```
"Push subcomponent reconstruction closer to the original weights."

**Total VPD loss:**
```
L_VPD = β1 * L_adv_recon
      + β2 * L_stoch_recon
      + β3 * L_importance
      + β4 * L_freq
      + β5 * L_delta
```

---

### VPD Step 6: Backprop and Update (Actual Learning)

Compute gradients of L_VPD and update via gradient **descent**:

```
Updated (trainable):
  U_l, V_l for all 24 matrices              — subcomponent vectors
  MLP weights for all subcomponents          — causal importance predictors

Not updated (frozen):
  Original model weights W_l                 — the target we're reverse-engineering
  Δ_l = W_l - U_l @ V_l                     — derived, shrinks as U@V improves
  m_adv                                      — was only for the inner search loop
```

Gradients flow through:
```
L_VPD → y_stoch/y_adv → W'[b,t] → U, diag(m), V
                                  → m → g → MLP params
```

---

### The Tension That Drives VPD

```
RECONSTRUCTION PRESSURE (wants g HIGH):
  L_adv_recon    — "survive worst-case ablation"
  L_stoch_recon  — "survive random ablation"

SPARSITY PRESSURE (wants g LOW):
  L_importance   — "few subcomponents active per input"
  L_freq         — "each subcomponent active on few inputs"

FAITHFULNESS PRESSURE:
  L_delta        — "U @ V should approximate W"
```

The equilibrium: g ≈ 1 only for subcomponents that are genuinely necessary for a given input,
g ≈ 0 for everything else. This equilibrium IS the decomposition — it tells us which pieces
of each weight matrix implement which mechanisms for which inputs.

---

### Summary: SPD vs VPD

| Aspect | SPD | VPD |
|--------|-----|-----|
| Subcomponents | Rank-1 per matrix | Same |
| Causal importance | Learned MLP per subcomponent | Same (but per-position in sequence) |
| Stochastic masking | ✓ (the only mask type) | ✓ (one of two mask types) |
| Adversarial masking | ✗ | ✓ (catches correlated cancellations) |
| Faithfulness | MSE loss: U@V ≈ W | Δ = W - U@V (exact by construction) + L2 penalty |
| Importance penalty | Σ\|g\|^p | Same |
| Frequency penalty | ✗ | ✓: Σ\|g\|^p * log(1 + freq) |
| Tested on | Toy models | 67M parameter transformer |
| Feature splitting | Claimed absent | Empirically confirmed absent |
