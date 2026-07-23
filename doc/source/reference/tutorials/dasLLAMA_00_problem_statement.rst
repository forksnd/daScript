.. _tutorial_dasLLAMA_problem_statement:

====================================
dasLLAMA-00 — The Problem Statement
====================================

.. index::
    single: Tutorial; dasLLAMA
    single: Tutorial; Transformer
    single: Tutorial; Llama 2

Before using dasLLAMA, build the thing it has to become.

This tutorial is a complete, deliberately tiny Llama-2 inference engine in one
daslang file. It maps a real 15-million-parameter checkpoint, runs every layer
of the transformer, chooses the most likely next token, and prints a story.
There is no ``require dasllama``: only the problem, expressed directly.

It tells a second story too. The first dasLLAMA commit was authored on
2026-06-27. It began with the same naive fp32 matrix multiplication used here;
about a day later, `PR #3297
<https://github.com/GaijinEntertainment/daScript/pull/3297>`_ contained
Llama-2 and Llama-3, GGUF loading, quantization, threading, batched prefill,
tests, and chat. The path from this readable program to native machine code
never had to leave one programming environment.

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

   daslang.exe tutorials/dasLLAMA/00_problem_statement.das -- stories15M.bin tokenizer.bin

The interpreter is intentionally usable here. On a typical development
machine it produces several tokens per second, which makes every operation
inspectable. Add ``-jit`` to run the exact same source as native code::

   daslang.exe -jit tutorials/dasLLAMA/00_problem_statement.das -- stories15M.bin tokenizer.bin

The five operations
===================

A Llama transformer needs only five numerical building blocks:

``matmul``
   Projects a vector through a matrix. Nearly all inference time is spent
   here, and the tutorial begins with the obvious row-by-row implementation.

``rmsnorm``
   Keeps the residual stream at a useful scale.

``softmax``
   Turns attention scores into weights which sum to one.

``silu``
   Supplies the non-linearity in the feed-forward gate.

``rope``
   Rotates pairs of query and key coordinates to encode token position.

They are ordinary, strongly typed daslang functions. For example, the entire
matrix multiplication is:

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
still an easy performance improvement left for the reader to find.

The checkpoint is already an array
==================================

A llama2.c checkpoint starts with seven integers describing the model. What
follows is one flat fp32 array containing embeddings, normalization weights,
attention matrices, feed-forward matrices, and the classifier.

``fmap`` maps the file, and scoped ``array_view`` blocks present the header as
``array<int>#`` and the weights as ``array<float>#``. Further views select each
matrix by element offset:

.. code-block:: das

   array_view(bytes, 28, (length(bytes) - 28) / 4, type<float>) $(model : array<float>#) {
       array_view(model, offset, n * d) $(matrix : array<float>#) {
           matmul(out, matrix, input, n, d)
       }
   }

The views are borrowed, checked, and confined to their blocks. Loading creates
no second 60 MB copy and the tutorial needs no ``unsafe`` code. Alignment and
lifetimes are properties of ``array_view``, not obligations pushed onto the
transformer.

One token through the model
===========================

``forward`` is the whole transformer. For each token it:

1. copies the token embedding into the residual stream;
2. normalizes it and projects query, key, and value;
3. applies RoPE and appends key and value to the cache;
4. computes causal self-attention for every head;
5. projects the attention result and adds the residual;
6. runs the gated SiLU feed-forward network and adds its residual; and
7. applies the final normalization and classifier to produce vocabulary
   logits.

That sequence repeats for six layers. ``most_likely`` selects the largest
logit, making generation deterministic.

Why there is no prompt
======================

Token ``1`` is both the beginning-of-sequence token and the sequence delimiter
used by llama2.c generation. Giving it to the model is enough to begin the
TinyStories continuation; generation ends when the model emits ``1`` again.
The checkpoint's sequence length remains the hard safety bound.

Prompting would not change the transformer. It would require the other half of
the tokenizer—splitting input text and applying its BPE merge scores—which is
useful code but not part of this problem statement. The tutorial therefore
loads only the vocabulary pieces needed to turn generated token ids back into
text. Prompt encoding arrives through the real dasLLAMA API in
:ref:`tutorial 01 <tutorial_dasLLAMA_hello_generate>`.

From seed to library
====================

This program deliberately fixes the dimensions, format, precision, execution
strategy, tokenizer direction, and sampling policy to one tiny model. dasLLAMA
is what happens when each of those constraints becomes an interface:

* checkpoint layout becomes GGUF metadata and an architecture registry;
* fp32 matrices become quantized tensor formats and tuned kernels;
* serial row loops become job-queue work;
* one-token evaluation gains batched prompt prefill;
* greedy choice becomes a sampling pipeline;
* a decoder vocabulary becomes full model-specific tokenization; and
* scratch arrays become reusable sessions with managed KV caches.

The important part is visible before any of that machinery: a fully functional
transformer is small enough to read, run, interpret, JIT, and improve in one
source file. That is the problem dasLLAMA solves at production scale.

.. seealso::

   Full source: :download:`tutorials/dasLLAMA/00_problem_statement.das <../../../../tutorials/dasLLAMA/00_problem_statement.das>`

   Next tutorial: :ref:`tutorial_dasLLAMA_hello_generate`
