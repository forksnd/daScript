.. _tutorial_dasLLAMA_problem_statement:

====================================
dasLLAMA-00 — The Problem Statement
====================================

.. index::
    single: Tutorial; dasLLAMA
    single: Tutorial; Transformer
    single: Tutorial; Llama 2

Before using dasLLAMA, build the thing it has to become.

This tutorial is a complete, deliberately tiny Llama-2 *inference* engine —
inference is running a model, as opposed to training one — in one daslang
file. It maps a real checkpoint (the file of 15 million learned numbers, or
*parameters*, that training produced), runs every layer of the transformer,
encodes a prompt, chooses the most likely next token, and prints the
completion.
There is no ``require dasllama``: only the problem, expressed directly.

It tells a second story too. The first dasLLAMA commit was authored on
2026-06-27. It began with the same naive fp32 matrix multiplication used here;
about a day later, `PR #3297
<https://github.com/GaijinEntertainment/daScript/pull/3297>`_ contained
Llama-2 and Llama-3, GGUF loading, quantization, threading, batched prefill,
tests, and chat. The path from this readable program to native machine code
never had to leave one programming environment.

No prior machine-learning background is assumed. By the end of the page you
should be able to read ``forward`` top to bottom and know why every line is
there.

Get the model
=============

Download the two files used by Karpathy's `llama2.c
<https://github.com/karpathy/llama2.c>`_:

* `stories15M.bin
  <https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin>`_
  — the TinyStories Llama-2 checkpoint
* `tokenizer.bin
  <https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin>`_
  — its tokenizer vocabulary

Put them anywhere convenient, then run from the daScript project root::

   daslang.exe tutorials/dasLLAMA/00_problem_statement.das -- stories15M.bin tokenizer.bin "Once upon a time"

The interpreter is intentionally usable here. On a fast desktop it tells the
story at about six tokens per second, which makes every operation
inspectable. Add ``-jit`` to run the exact same source as native code — about
75 tokens per second, twelve times the interpreter::

   daslang.exe -jit tutorials/dasLLAMA/00_problem_statement.das -- stories15M.bin tokenizer.bin "Once upon a time"

Leave off the final argument to use the default prompt, ``Once upon a time``.

Either way, the run ends in the same place. This tutorial decodes greedily —
always the single most likely next token — so this checkpoint tells exactly
one story for this prompt, every run, token-identical in the interpreter and
under JIT::

   dim=288, layers=6, heads=6, parameters≈15204000

   Once upon a time, there was a little girl named Lily. She loved to play
   outside in the sunshine. One day, she saw a big, red ball in the sky. It
   was the sun! She thought it was so pretty.
   ...

The rest of Lily's story follows, token by token. By the end of this page you
should be able to read ``forward`` and say why every one of those tokens
happened.

Numbers, not words
==================

The model cannot read. The tokenizer splits the prompt into pieces from a
fixed 32,000-entry vocabulary and hands over their integer IDs. Everything
after that is arithmetic on *vectors* — fixed-length arrays of floats — and
all of it rests on one operation.

One word has to be settled first: **learned**. Every number this program
touches was chosen by *training*, a search that finished long before this
program runs: show the model text, nudge all 15 million numbers so that the
true next token scores slightly higher, repeat a few billion times. Training
is not part of this program and is not covered here — this page only
executes its result. But wherever a table or matrix below is called
*learned*, that is what it means: its values were selected, example by
example, to make the final prediction come out right.

First, each ID is traded for a vector: row ``token`` of a learned
32,000 × 288 table. Those 288 floats are coordinates. Training pushed tokens
that behave alike toward similar directions, so the row for ``dog`` ends up
far closer to ``cat`` than to ``Tuesday`` — not by decree, but because these
same rows double as the final classifier (visible at the end of the page):
contexts that could continue with either word pull both rows toward the same
directions, and rows pulled the same way end up neighbors. No individual
coordinate means
anything by itself; *direction* in this 288-dimensional space is what carries
meaning.

Second, vectors are compared with the **dot product**: multiply matching
coordinates and add everything up.

.. figure:: /_static/dasllama/00_dot_product.*
   :width: 100%
   :alt: Aligned vectors have a large positive dot product, unrelated vectors near zero, opposed vectors negative.

The result is a single number measuring agreement, and it is the only
comparison this program ever makes. One geometric fact about it matters
later: the dot product depends only on the two lengths and the angle between
the vectors — turn both by the same amount and nothing changes.

A matrix multiplication is many dot products at once: each of the ``d`` rows
of ``W`` is a learned pattern, and ``y[row] = dot(W[row], x)`` asks "how
strongly does ``x`` express this pattern?" Every matrix you will meet below —
the attention projections, the feed-forward, the classifier — is used
exactly this way, one question per row, which is why nearly all inference
time is spent in ``matmul``.

The transformer at a glance
===========================

.. figure:: /_static/dasllama/00_pipeline.*
   :width: 100%
   :alt: One token's journey: prompt to token IDs to embedding, through six layers of attention and feed-forward, to logits, with the chosen token fed back in.

``x`` is the **residual stream**: the model's evolving representation of the
current token in the context of everything before it — *residual* because
each layer contributes only a difference on top of it, never a replacement.
It starts as the token's embedding row. Each of the six layers then adds two
corrections:

* **attention** pulls in information from the tokens seen so far — the only
  step where positions communicate; and
* the **feed-forward** block transforms the token's own features in place.

"Adds" is literal — ``x = x + correction``, twice per layer. Erasing what
came before would take a correction that actively cancels it, so the default
is refinement: six layers work on one accumulating representation instead of
computing six unrelated ones.

After the last layer, the classifier compares ``x`` against every vocabulary
row — one final matrix multiplication — producing one score, or *logit*, per
possible next token; why this names the *next* token is taken up below. The
largest logit wins, and the winner is fed back in at ``position + 1`` for the
journey to repeat.

Prompt processing is called **prefill**. Generation is the loop at the
bottom: choose a token, display it, feed it back to ``forward``, and choose
again. Both paths use exactly the same transformer in this tutorial.

The five numerical operations
=============================

A Llama transformer needs only five numerical building blocks:

``matmul``
   Computes ``y = W x`` — ``d`` dot products against the input vector, one
   per matrix row. These learned projections create Q, K, and V, run the
   feed-forward network, and produce vocabulary logits. Nearly all inference
   time is spent here.

``rmsnorm``
   Scales a vector by
   ``1 / sqrt(mean(x[i] * x[i]) + epsilon)``, then applies a learned weight per
   coordinate. The residual stream is added to twice per layer, so its
   magnitude would otherwise drift over six layers; rmsnorm resets the scale
   before each use, and the learned weight restores whatever per-coordinate
   scale training preferred. The two steps are not a wash: the division is
   input-dependent — it cancels whatever magnitude this particular ``x``
   arrived with — while the learned weight is a fixed constant, so downstream
   matrices always see inputs at a predictable scale.

``softmax``
   Converts arbitrary attention scores ``s[i]`` into positive weights
   ``exp(s[i] - max(s)) / sum(exp(s[j] - max(s)))``. The weights sum to one, so
   attention becomes a weighted average of value vectors. Subtracting the
   largest score changes no ratio but prevents ``exp`` from overflowing.

``silu``
   Computes ``x * sigmoid(x)``, where ``sigmoid(x) = 1 / (1 + exp(-x))``
   slides smoothly from 0 to 1 — a soft switch. Without a non-linearity, stacked matrix
   multiplications would collapse into one linear transformation. Llama uses
   SiLU as the learned gate of its feed-forward block.

``rope``
   Rotary position embedding — how the model knows *where* a token is.
   Each pair of Q and K coordinates is treated as a 2D arrow and turned by
   ``position * frequency``; the rotations cancel in the dot product except
   for the difference between positions. Explained in full in the attention
   walkthrough below.

They are ordinary, strongly typed daslang functions (the ``#`` in a
signature marks a borrowed view — explained with the checkpoint below). For
example, the entire matrix multiplication is:

.. code-block:: das

   def matmul(var y : array<float>; w : array<float>#; x : array<float>; n, d : int) {
       for (row in range(d)) {
           var sum = 0.0
           let offset = row * n
           for (col in range(n)) {
               sum += w[offset + col] * x[col]
           }
           y[row] = sum
       }
   }

The small loop invariants are worth naming even in reference code: they make
the structure clearer and save repeated work in the interpreter. There is
still an easy performance improvement left for the reader to find — a hint:
daslang's ``for`` can walk several arrays in lockstep.

The checkpoint is already an array
==================================

A llama2.c checkpoint starts with seven integers describing the model. What
follows is one flat fp32 array containing embeddings, normalization weights,
attention matrices, feed-forward matrices, and the classifier — plus a
stretch of precomputed RoPE tables this program computes on the fly and
simply skips (the lone ``at +=`` in ``weight_layout``, and the reason the
printed parameter count carries a ``≈``).

``fmap`` memory-maps the file — its bytes appear as an array, nothing is
copied — and scoped ``array_view`` blocks present the header as
``array<int>#`` and the weights as ``array<float>#`` (the trailing ``#``
marks a borrowed view that cannot outlive the block it is handed to).
Further views select each matrix by element offset:

.. code-block:: das

   array_view(bytes, 28, (length(bytes) - 28) / 4, type<float>) $(model : array<float>#) {
       array_view(model, offset, n * d) $(matrix : array<float>#) {
           matmul(out, matrix, input, n, d)
       }
   }

The views are borrowed, checked, and confined to their blocks. Loading creates
no second 60 MB copy, and the transformer does no pointer arithmetic.
Alignment and lifetimes are properties of ``array_view``, not obligations
pushed onto ``forward``. There is no ``unsafe`` anywhere in the file: even
the tokenizer's floats — which sit at unaligned offsets, because its records
carry variable-length text — are read with ``pod_view``, the checked,
alignment-free counterpart of ``array_view``.

One token through the model
===========================

``forward`` takes one ``token`` at one ``position``. It does not return text or
even a token ID: it fills ``state.logits`` with the scores from which the
caller will choose the next token.

Embedding: an ID becomes a vector
---------------------------------

The embedding table has one learned row of ``dim`` floats per vocabulary
token. ``forward`` copies the selected row into ``s.x``. From this point on,
the transformer manipulates vectors, not token IDs. ``s.x`` remains the
residual stream throughout all layers.

Attention: retrieve relevant earlier information
------------------------------------------------

The embedding row for ``time`` is the same in "once upon a time", "the time
is now", and "time flies" — one context-free vector per vocabulary entry.
Attention is how the vector picks its context back up: it lets the current
position retrieve information from every position in the cache — everything
so far, itself included.

Each layer first normalizes ``x``, then uses three matrix multiplications to
make three views of the current token:

* the **query** (Q) — what this token is looking for;
* the **key** (K) — what this token can be found *by*; and
* the **value** (V) — what it hands over once found.

The names describe use, not essence. The three matrices are three ordinary
projections of the same normalized ``x``, and nothing intrinsic makes one a
"query": Q earns its name because the code only ever compares it against
keys, V because it only ever enters the final blend. Training exploits that
structure and shapes each matrix for its role.

One thing is still missing: nothing computed so far knows *where* anything
is. The dot product is blind to order — without position information, "the
dog bit the man" and "the man bit the dog" would attend identically. Position
has to be written into Q and K themselves, and RoPE — *rotary position
embedding* — writes it in as rotation.

Take two adjacent coordinates of a vector and treat them as one 2D arrow.
Rotating that arrow changes its direction but not its length, so whatever the
two coordinates encoded is preserved — but the arrow's orientation now also
says "position ``p``": each step forward in the sequence turns it by one more
increment of that pair's angle.

.. figure:: /_static/dasllama/00_rope.*
   :width: 100%
   :alt: Position rotates each coordinate pair; rotations on Q and K cancel in the dot product, leaving only the relative distance; different pairs rotate at different speeds.

The payoff appears in the dot product. First, note the 48-wide score splits
cleanly: group the coordinate products two at a time and the head's dot
product is just the sum of 24 independent 2D pair-dots, so a per-pair
argument covers the whole thing. Now recall the geometric reading: a dot
product depends only on the two lengths and the angle between the vectors.
Rotation
changes no lengths, so rotating Q by ``p * theta`` and K by ``q * theta``
changes the angle between them by exactly the difference,
``(p - q) * theta`` — the absolute positions cancel. Attention measures how
far back a key sits, not where the pair happens to fall in the sequence:
"three tokens ago" looks the same at position 10 as at position 210.

A single rotation speed would be ambiguous — turn far enough and the arrow
comes full circle. So each of a head's 24 pairs gets its own frequency (the
``i % head_size`` in ``rope`` restarts the ladder for every head — enough,
because scores never mix slices of different heads), from one full turn
every ~6 tokens (the fastest pair turns one radian per token,
and a full turn is 2π) down to a barely perceptible drift — the second,
minute, and hour hands of a clock. That spread is the
``1.0 / pow(10000.0, ...)`` line in ``rope``; the base 10000 is a convention
inherited from the earliest transformers, and any wide spread of speeds
works. V is never rotated: it is the payload, and only the Q–K match needs
to know distance.

The rotated K and the untouched V are then copied into the layer's
**KV cache** — the model's memory of every position it has processed, the
current one included: a token attends over its own K and V too. Without that
cache, generating every new token would recompute keys and values for the
entire prompt.

The retrieval itself happens in **heads**. Comparing all 288 coordinates as
one block would produce a single softmax — one way of looking back per layer —
and a token usually needs several at once: one eye on the previous word,
another on the subject, a third on the opening "Once". So Q, K, and V are
split into six slices of 48 coordinates, and each slice — each *head* — runs
the whole retrieval independently, with its own scores and its own weights.
"Head" is the standard name for one of these parallel readers; in the code it
is literally a slice, ``query_row = head * head_size``.

For one head and one cached position ``p``, the code computes:

.. code-block:: text

   score[p] = dot(query, key_cache[p]) / sqrt(head_size)

A high score means "this cached position holds what I am looking for" — the
query direction agrees with that key's direction. The ``sqrt(head_size)``
division keeps scores in the same range at any width: for typical,
uncorrelated coordinates the 48 products partially cancel like a random
walk, so the sum grows like ``sqrt(48)``, not 48. The range matters because
``softmax`` exponentiates — scores a few units apart already produce
near-winner-take-all weights, and unscaled scores would freeze every head
into hard single-position lookups. ``softmax`` turns all of the head's
scores into weights, and the weighted sum

.. code-block:: text

   attention_head = sum(weight[p] * value_cache[p])

retrieves a mixture of cached values: mostly those of high-scoring
positions, faint traces of the rest.

.. figure:: /_static/dasllama/00_attention.*
   :width: 100%
   :alt: One attention head: the query from the current token is scored against every cached key, softmax turns scores into weights, and the weights blend the cached values.

The score loop ends at ``position``, so future tokens are never considered;
that loop bound is the **causal mask** expressed without a separate mask
array. (The ``head / kv_multiple`` indexing supports **grouped-query
attention** — larger models cut cache memory by sharing one cached K/V head
among several query heads. stories15M has six of each, so the multiple is 1
and the indexing is invisible here.)

The six 48-float head outputs are concatenated in ``s.xb`` — 288 again; the
normalized copy that produced Q, K, and V is dead by then, so the buffer is
reused. Each head retrieved on its private 48-coordinate slice, so one more
learned projection, ``attention_output``, mixes the six answers into a
single 288-wide correction, which is added to ``x``. This **residual
connection** lets a layer contribute a correction while preserving the
representation it received:

.. code-block:: text

   x = x + attention(x)

Feed-forward: transform each token's features
---------------------------------------------

Attention moves information between token positions. The feed-forward block
then transforms the current token's features in place. After another RMSNorm,
two projections expand ``dim`` to ``hidden_dim`` — from 288 to 768 features.
One path passes through SiLU and gates the other element by element: where
the gate is near zero, that feature is switched off. A final projection
returns to ``dim``:

.. code-block:: text

   hidden = W_down (SiLU(W_gate x) * (W_up x))
   x      = x + hidden

This is Llama's gated feed-forward network, commonly called **SwiGLU**.

.. figure:: /_static/dasllama/00_silu.*
   :width: 100%
   :alt: The SiLU curve: near zero for negative inputs, close to the identity for positive inputs, smooth throughout.

The bend in that curve is what *non-linear* means — negative inputs are
squeezed to nearly nothing, positive inputs pass through almost unchanged —
and the bend is what makes depth count. Chains of purely linear operations
collapse: two matrix multiplications in a row are exactly equivalent to one.
Put a bend between them and they are not; each layer can now reshape what the
previous ones produced instead of merely restating it. The feed-forward
block also holds about two thirds of every layer's weights in these three
matrices — the expansion to 768 gives the gate more features to switch than
the stream itself carries. The second residual addition completes the layer.
Attention plus feed-forward repeats for every layer in the checkpoint.

Logits: a vector becomes the next-token choice
-----------------------------------------------

After the final layer, one last RMSNorm stabilizes ``x``. The classifier
matrix projects it from ``dim`` values to ``vocab`` logits — one dot product
of ``x`` against each of 32,000 rows. For stories15M the classifier *is* the
embedding table (``shared_classifier`` in the code; larger models often
train a separate matrix), so the question is literally: which token's row
does ``x`` now point along?

That question deserves a pause, because it is the whole trick. ``x``
*started* as the current token's own row — so why does the comparison name
the token that comes *after*? Nothing in the arithmetic makes it so. It is
what training selected every weight to achieve: layer by layer, ``x`` is
nudged away from "the token I am" toward "the token that follows here",
because that final comparison is exactly what training scored, billions of
times. The mechanics on this page are the pipeline; the checkpoint's numbers
are why it points the right way.

A logit is merely an unnormalized score: larger means the model considers
that token more likely. ``most_likely`` chooses the largest one, so this
tutorial is deterministic. Everything usually called *sampling* — the
*temperature* and *top-p* knobs of chat UIs — is a different policy applied
to this same vector: soften the logits into a probability distribution with
``softmax`` and draw from it instead.

Prompting is repeated ``forward``
=================================

``tokenizer.bin`` stores a vocabulary piece and a precomputed merge score for
every token. ``encode`` begins with token ``1`` — **BOS**, *beginning of
sequence*, a marker the model saw at the start of every training text — and,
before the first word, SentencePiece's dummy leading-space token, since
pieces mark word starts with a space. It
then splits the prompt into UTF-8 characters, falling back to raw bytes for
anything the vocabulary lacks (IDs 0–2 are reserved markers, so byte ``b``
lives at token ``b + 3``), and repeatedly joins the highest-scoring adjacent
pair found in the vocabulary — the merge scores decide which pairs deserve
to be one token. This is llama2.c's compact SentencePiece-style BPE encoder;
``token_text`` is the decoder half of the same scheme (pieces carry a
leading space marking a word start, which the first word after BOS drops,
and ``<0xNN>`` pieces decode back to raw bytes).

Prefill is deliberately unsurprising:

.. code-block:: das

   for (position, token in range(length(tokens)), tokens) {
       forward(c, w, model, s, token, position)
   }

After the last prompt token, ``state.logits`` already predicts the first
completion token. Generation chooses it, prints its vocabulary piece, calls
``forward`` with that token at the next position, and repeats. The KV cache
written during prefill makes the whole prompt available to every generated
token.

.. figure:: /_static/dasllama/00_generation.*
   :width: 100%
   :alt: Prefill runs forward once per prompt token and fills the KV cache; generation feeds each chosen token back in as the next input.

This tutorial's serial prefill makes the relationship completely visible.
Production dasLLAMA evaluates prompt tokens in batches, but computes the same
causal result. Generation stops when the model predicts token ``1`` again:
the training data was many stories laid end to end with the BOS marker
between them, so predicting it is the model saying "this story is complete —
a new one would start here." The checkpoint's sequence length is the other
stop: the hard context bound.

From seed to library
====================

This program deliberately fixes the dimensions, format, precision, execution
strategy, tokenizer direction, and sampling policy to one tiny model. dasLLAMA
is what happens when each of those constraints becomes an interface:

* checkpoint layout becomes GGUF metadata and an architecture registry;
* fp32 matrices become quantized tensor formats and tuned kernels;
* serial row loops become job-queue work;
* serial prompt prefill becomes a batched implementation;
* greedy choice becomes a sampling pipeline;
* one SentencePiece tokenizer becomes full model-specific tokenization; and
* scratch arrays become reusable sessions with managed KV caches.

The important part is visible before any of that machinery: a fully functional
transformer is small enough to read, run, interpret, JIT, and improve in one
source file. That is the problem dasLLAMA solves at production scale.

Credits and further reading
===========================

This tutorial reimplements, in daslang, the design of Andrej Karpathy's
`llama2.c <https://github.com/karpathy/llama2.c>`_: the checkpoint format,
the tokenizer format, and the shape of the forward pass all follow his
``run.c``. The checkpoint is his `tinyllamas
<https://huggingface.co/karpathy/tinyllamas>`_ training run on the
`TinyStories <https://arxiv.org/abs/2305.07759>`_ dataset (Eldan & Li) —
which is why a 15-million-parameter model can tell a coherent story at all.

If this page leaves you wanting the training half of the story, Karpathy
teaches it better than anyone:

* `Neural Networks: Zero to Hero <https://karpathy.ai/zero-to-hero.html>`_ —
  the full video course, from backpropagation to a working GPT;
* `Let's build GPT: from scratch, in code, spelled out
  <https://www.youtube.com/watch?v=kCc8FmEb1nY>`_ — the transformer built
  live, the natural companion to this page;
* `Let's build the GPT Tokenizer
  <https://www.youtube.com/watch?v=zduSFxRajkE>`_ — everything ``encode``
  glosses over.

The architecture itself comes from the papers: `Attention Is All You Need
<https://arxiv.org/abs/1706.03762>`_ (the transformer), `RoFormer
<https://arxiv.org/abs/2104.09864>`_ (rotary position embedding), `GLU
Variants Improve Transformer <https://arxiv.org/abs/2002.05202>`_ (SwiGLU),
and `Llama 2 <https://arxiv.org/abs/2307.09288>`_ (this exact assembly of
them).

.. seealso::

   Full source: :download:`tutorials/dasLLAMA/00_problem_statement.das <../../../../tutorials/dasLLAMA/00_problem_statement.das>`

   Next tutorial: :ref:`tutorial_dasLLAMA_hello_generate`
