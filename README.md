\# ROCA-AIAssistant: A Capsule Theory of Long-Lived Assistants with Orbital Salience

**Anonymous Authors**  
**Affiliation**  
**Contact**

## Abstract
We present a theory of AI assistants as long-lived systems that must maintain identity, preferences, and heterogeneous knowledge across months and years. We argue that **capsules**—explicit, typed, persistent memory objects—provide the right unit of assistant continuity (characters, styles, skills/tools, workflows, topics, and episodic memories). However, capsule-centric designs can become computationally expensive when they rely on high-dimensional neural embeddings for every update, global similarity search across growing memory, and frequent consolidation.

We frame an alternative in the ROCA style (Routed Orbital Capsule Architecture): (i) a routing/retrieval layer that activates a small subset of capsules per episode, (ii) replayable **salience dynamics** updated by usage and agreement signals, and (iii) a UI-only **orbital projection** that externalizes long-horizon salience as continuous inward/outward drift around an **Identity Nucleus**. A central constraint is that orbit geometry improves legibility but does not gate routing or retrieval. We show how orbital salience enables low-overhead genre adaptation and domain handoff: when an assistant transitions from metallurgy to chemistry, frequently used chemistry capsules drift inward while metallurgy naturally drifts outward into cold storage without being deleted.

## 1. Introduction
Many deployed assistants behave like short-context completion engines: effective locally, but brittle over time. When an assistant must persist across domains and genres, two challenges dominate:

1. **Selection at scale:** as memory grows, selecting the right context becomes expensive and noisy.
2. **Legibility and management:** users need to understand what is “core,” what is “stale,” and why behaviors recur.

This paper treats assistants as systems with explicit memory and control, not merely prompt templates. We argue for capsule-based memory and for ROCA-style orbital salience as an operating model that is both interpretable and compute-aware.

## 2. Contributions
- **Capsule theory for assistants:** defines capsules as explicit units of durable context with kind-specific metadata and routing representations.
- **Identity Nucleus + functional lanes:** separates always-on policy capsules from domain/genre capsules to prevent domain dominance and identity drift.
- **Agreement-driven gravity:** models long-horizon salience as a smoothed function of usage, recency, agreement, and connectivity.
- **UI-only orbital projection:** orbit geometry is a visualization contract and does not gate routing/retrieval.
- **Low-overhead adaptation:** formalizes how salience drift supports domain handoff (metallurgy → chemistry) and multi-genre behavior.
- **Reversible consolidation:** uses proxy capsules with shadow identities to reduce duplication without losing provenance.

## 3. Capsules as the unit of assistant memory
### 3.1 Definition
A **capsule** is a persistent object with:

- **Kind/type** (e.g., nucleus, character, style, skill/tool, workflow, memory, experimental)
- **Human-readable payload** (text, structured fields)
- **Metadata** (tags, assets, provenance)
- **Routing representation** (embedding, hash-vector, sparse tokens, or hybrid)
- **Usage history** (frequency, recency, outcome proxies)

Capsules need not be “true”; they need to be operationally useful and auditable.

### 3.2 Capsule taxonomy
We assume the ROCA taxonomy:

- **Identity Nucleus capsules:** communication defaults, reasoning preferences, user preferences, memory policy.
- **Character capsules:** identity bundles (traits, voice), references/assets, links to workflows/memories.
- **Style capsules:** aesthetic profiles (tone, palette), references.
- **Skill/tool capsules:** invocation templates and constraints.
- **Workflow capsules:** multi-step procedures and tool-chain signatures.
- **Memory capsules:** episodic summaries, entities, facts, provenance.
- **Experimental capsules:** sandboxed modules not active by default.

Table 1 gives one pragmatic representation target (illustrative, not a requirement).

| Capsule Type | Purpose | Example Representation |
|---|---|---|
| Character | Personas/characters | 32–64D routing vector + text + optional images |
| Style | Tone or aesthetic | 16–32D routing vector + references |
| Skill/Tool | Abilities/procedures | 16–32D vector + templates/constraints |
| Workflow | Multi-step pipelines | signature + examples + constraints |
| Memory/Fact | Episodic knowledge | text + optional embedding |
| Core/Identity | Always-on policy | compact “nucleus” capsule set |

### 3.3 Why capsules (and why not raw chat logs)
Capsules provide persistence, modularity, and controllable retrieval. Raw chat logs become long, redundant, and expensive to search, and they are difficult to curate without destructive summarization.

## 4. Why capsule assistants become computationally expensive
Capsules shift cost from “prompt only” to “memory operations.” The dominant cost drivers are:

### 4.1 Representation cost
Neural embeddings are expensive to compute and store, especially on CPU-only or offline deployments.

### 4.2 Global retrieval cost
Naïve routing is “embed query → score against all capsules → sort.” For $N$ capsules of dimension $d$, cosine scoring is $O(Nd)$ with $O(N \log N)$ sorting.

### 4.3 Consolidation cost
Continuous merging/summarization can be expensive and risky: it consumes model compute and can irreversibly remove nuance.

### 4.4 Hidden system costs
Long-lived systems also maintain agreement proxies, capsule graphs, and event logs for replay/audit; without incremental design, these add CPU and I/O overhead.

## 5. ROCA architecture: separation of concerns
We adopt the ROCA separation:

1. **Routing/Retrieval:** selects a subset of capsules per episode.
2. **Salience dynamics:** updates long-horizon salience from events (use, co-activation, correction).
3. **UI projection:** renders rings/orbits from salience; geometry does not gate selection.

This separation allows headless execution, deterministic replay, and UI swapping without changing routing correctness.

## 6. Dynamics: agreement, gravity, drift
### 6.1 Agreement signals
Agreement is an empirical proxy for coherence and stability:

- **Co-activation** frequency
- **Conflict penalties** (user corrections after activation)
- **Outcome proxies** (fewer retries, sustained continuation)

Agreement updates a capsule graph, where edge weights represent compatibility/co-use.

### 6.2 Gravity (salience)
Each capsule maintains gravity $g(c) \in [0,1]$:

$$
 g(c) = \sigma\Big(
 w_f \cdot \log(1 + \text{useCount}_c) +
 w_r \cdot \exp(-\Delta t_c / \tau) +
 w_a \cdot \text{agreement}_c +
 w_k \cdot \text{connectivity}_c
 \Big)
$$

As an MVP, maintain integer `orbitScore` with `+1` on use and `-1` on decay tick; then $g(c)=\sigma(\alpha \cdot \text{orbitScore}_c)$.

### 6.3 UI-only orbital projection
Each capsule kind occupies a lane $L$ with band $[r_{\min}(L), r_{\max}(L)]$. The target radius is:

$$
 r^*(c) = r_{\min}(L) + (1 - g(c)) \cdot (r_{\max}(L) - r_{\min}(L)).
$$

Displayed radius is smoothed:

$$
 r_{t+1}(c) = r_t(c) + \lambda (r^*(c) - r_t(c)).
$$

Angle stability preserves spatial memory:

$$
 	heta(c) = 2\pi \cdot \text{hash01}(id_c).
$$

**Non-goal:** using radius as a routing gate.

## 7. Compute-aware routing: why orbital salience can be cheaper
Orbit geometry alone does not reduce compute. Savings come from salience dynamics enabling an **active frontier**.

### 7.1 Active set routing
Maintain an active set $A$ (e.g., inner orbits / top-$k$ by gravity). For most queries, route primarily against $|A| \ll N$, reducing typical cost to $O(|A|d)$.

### 7.2 Hybrid representations
Routing can be multi-stage:

- Cheap: token overlap, tag boosts, hash-vectors
- Expensive: neural embeddings only when needed

This preserves offline operation and limits embedding compute.

### 7.3 UI-only distance prevents quality regressions
Explicitly separating UI projection from routing prevents the failure mode where visualization heuristics silently degrade retrieval correctness.

## 8. Capsules as genre controllers
Genres (technical writing, creative ideation, coaching, storyboarding, code generation) can be encoded as style/workflow/project capsules. Lane separation prevents any single genre from permanently dominating, while gravity drift brings the currently effective genre capsules closer to the nucleus.

## 9. Case study: metallurgy assistant handed to a chemist
An assistant used heavily for metallurgy will have metallurgy capsules with high gravity and inner-orbit placement. After a handoff to a chemist:

1. **Early transition:** chemistry questions repeatedly activate chemistry capsules (reaction mechanisms, solvent selection, analytical techniques). Their gravity increases and they drift inward.
2. **Re-centering:** metallurgy capsules experience decay and drift outward as use falls.
3. **No destructive forgetting:** metallurgy remains available as cold storage for rare queries and cross-domain work.

Identity remains stable because the nucleus is always-on and does not compete for topical salience.

## 10. Spawning and dedupe
ROCA-style systems propose new capsules via lane-specific detectors (characters, styles, workflows, memory clusters). Before spawning, dedupe against existing lane capsules; if similarity exceeds threshold $\delta$, reinforce instead of spawning. New capsules start in outer orbits with low initial gravity.

## 11. Consolidation without loss: shadow identities
To manage near-duplicates without irreversible merges:

- Create merged proxy capsule $M$ when $A$ and $B$ exhibit sustained similarity, co-activation, and low conflict.
- Keep $A$ and $B$ as **shadow identities** with provenance.
- If later divergence appears, reduce merge confidence and re-promote shadows.

## 12. Evaluation protocol
Because orbit geometry is UI-only, evaluation emphasizes interpretability and maintainability:

- **UI comparison:** orbital view vs list/grid with identical retrieval.
- **Tasks:** locate relevant capsules; diagnose behavior sources; resolve duplicates.
- **Measures:** time-to-find, interaction count, clarity/trust, correction rates, duplicate incidence.
- **Ablations:** remove agreement graph; remove smoothing; remove lane separation; remove coalescing.

## 13. Discussion and limitations
Limitations include reliance on agreement proxies, risk of over-spawning without good dedupe, and UI clutter at large capsule counts. These are primarily systems/product problems (rate limits, semantic zoom, curation tools), not fundamental obstacles to capsule theory.

## 14. Conclusion
Capsules provide a durable and legible substrate for long-lived assistants, but naïve capsule retrieval and consolidation can be computationally expensive. ROCA-style separation—routing, salience dynamics, and UI-only orbital projection—yields an assistant that is interpretable, stable across long horizons, and compute-aware in the common case. Orbital salience supports domain handoff without deletion: new domains drift inward through use, while older domains drift outward through decay.

## Appendix A: Minimal policy sketch
- **Create** capsules when stable concepts repeat.
- **Route** per episode using an active set plus selective expansion.
- **Update** gravity from usage and agreement.
- **Decay** on time ticks.
- **Coalesce** via proxy+shadows rather than destructive merges.

This makes the assistant adaptive, stable, and efficient.
