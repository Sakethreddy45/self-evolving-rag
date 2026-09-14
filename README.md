# Self-Evolving RAG

A RAG system that identifies retrieval failures and improves its retrieval strategies over time.

## Architecture

```text
                        ┌────────────────────────────────────┐
                        │           RUNTIME GRAPH            │
                        │         (per query, stable)        │
                        │                                    │
   question ──────────► │   plan ──► retrieve ──► grade      │
                        │    ▲          ▲           │        │
                        │    └──────────┼───────────┤        │
                        │               │           ▼        │
                        │  answer ◄─ verify ◄─── generate    │
                        │               │           ▲        │
                        │               └───────────┘        │
                        └────┬─────────────────────▲─────────┘
                             │                     │
                   every run │                     │ selects strategy
                      logged │                     │
                             ▼                     │
                  ┌────────────────────┐   ┌──────┴─────────────┐
                  │   TRAJECTORY LOG   │   │  STRATEGY REGISTRY │
                  └─────────┬──────────┘   │  live │ shadow │ 🔒 │
                            │              └────────▲────────────┘
                            ▼                       │
              ┌──────────────────────────┐         │
              │      LEARNING LOOP       │         │
              │                          │         │
              │  cluster failures        │         │
              │       ↓                  │         │
              │  diagnose cause          │         │
              │       ↓                  │         │
              │  write prompt artifacts  │         │
              │       ↓                  │         │
              │  evaluate against key    │         │
              │       ↓                  │         │
              │  keep if improved ───────┼─────────┼──► runtime
              └────────────┬─────────────┘         │
                           │                       │
                failure persists                  │
                after two rounds                  │
                           │                       │
                           ▼                       │
              ┌──────────────────────────┐         │
              │     EVOLUTION LOOP       │         │
              │                          │         │
              │  write retrieval spec    │         │
              │       ↓                  │         │
              │  human review            │         │
              │       ↓                  │         │
              │  generate Python         │         │
              │       ↓                  │         │
              │  static screening        │         │
              │       ↓                  │         │
              │  sandbox evaluation      │         │
              │       ↓                  │         │
              │  shadow run              │         │
              │       ↓                  │         │
              │  promote ────────────────┼─────────┘
              └──────────────────────────┘

                    Both loops evaluated against
                              │
                              ▼
              ┌────────────────────────────────────┐
              │             ANSWER KEY             │
              │       human-created · read-only    │
              │       outside generated code       │
              └────────────────────────────────────┘
```

## Runtime Graph

The runtime graph handles each query while keeping the execution flow fixed.

A question is first planned into one or more queries. Each query is sent to a retrieval strategy, and the returned documents are graded for relevance.

Relevant documents are passed to generation. The generated answer is then verified for:

* **Grounding:** Is the answer supported by the retrieved documents?
* **Usefulness:** Does the answer actually address the question?

The result determines the next step. An ungrounded answer can be regenerated, while a grounded but unhelpful answer can trigger another planning and retrieval cycle.

Every step is recorded in the trajectory log, including retrieval strategy, queries, documents, grading results, generation, verification, and failures.

The graph itself stays fixed. Retrieval strategies are selected through the strategy registry.

## Learning Loop

The learning loop analyzes trajectories and groups similar failures by their failure pattern rather than by topic.

It can produce changes such as:

* Query decomposition prompts
* Query rewrite prompts
* Few-shot examples
* Routing instructions

Each change is evaluated against the answer key. Changes are kept only when they improve the relevant failures without causing regressions.

If prompt-level changes continue to fail, the problem can be passed to the evolution loop.

## Evolution Loop

The evolution loop handles failures that require a new retrieval capability rather than another prompt change.

It generates a retrieval strategy that follows a fixed interface. Possible strategies include:

* Metadata filtering
* Hybrid search
* Reranking

Generated code goes through static screening and sandbox evaluation before being registered. A successful strategy first runs in shadow mode, where its results can be compared with the live strategy without affecting the user response.

Strategies that pass the evaluation and shadow stage can be promoted to the live registry.

## Strategy Registry

The strategy registry controls which retrieval strategies are available to the runtime graph.

Strategies can be:

* **Live** — available for production retrieval
* **Shadow** — evaluated alongside the live strategy
* **Quarantined** — removed from live use after a failure

This keeps generated retrieval code separate from the runtime graph itself. A new strategy can be added or removed without changing the graph.

## Evaluation

A fixed, human-created answer key provides ground truth for evaluating changes.

The answer key is read-only and kept outside generated code. It is used to determine whether a prompt change or new retrieval strategy actually improves the system rather than relying only on the system's own retrieval and grading results.
