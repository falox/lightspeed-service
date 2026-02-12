# Tuning Log

Logical fixes and improvements to the OLS codebase. Each entry describes the
problem, the root cause, and the fix so it can be re-applied even if the
surrounding code has changed significantly.

---

## 1. Preserve tool-call history in follow-up conversation turns

**Problem**: During a multi-turn troubleshooting session the LLM uses MCP tools
on the first reply but stops using them in subsequent replies unless the user
explicitly asks. The user has to say things like "use tools to check" to get
tool usage again.

**Root cause**: When conversation history is reconstructed from the cache for
follow-up turns, only `HumanMessage` and `AIMessage` (text) are kept. The
intermediate `AIMessage(tool_calls=...)` and `ToolMessage` (tool results) from
prior turns are discarded. The LLM therefore sees a history where it answered
directly without tools, and follows that pattern.

The method responsible is `CacheEntry.cache_entries_to_history` in
`ols/app/models/models.py`. It iterated over cached entries and appended only
`entry.query` and `entry.response`, ignoring the `tool_calls` and
`tool_results` fields that are already stored in each `CacheEntry`.

**Fix**: When a cache entry contains `tool_calls` and `tool_results`, reconstruct
the full LangChain message sequence before the final text response:

1. `HumanMessage` — the user query
2. `AIMessage(content="", tool_calls=[...])` — the LLM's tool-call request
3. `ToolMessage(content=..., tool_call_id=...)` for each tool result, matched
   by ID
4. `AIMessage(content=...)` — the final text response

Entries without tool calls are unchanged (just `HumanMessage` + `AIMessage`).

This way the LLM sees that it previously used tools and received results,
which naturally encourages it to continue using tools when the follow-up
warrants it.

**Files changed**:
- `ols/app/models/models.py` — `CacheEntry.cache_entries_to_history`, added
  `ToolMessage` import
- `tests/unit/app/models/test_models.py` — added tests for tool-call history
  reconstruction and mixed (tool + non-tool) conversation history

---

## 2. Increase tool output token budgets

**Problem**: MCP tool outputs (logs, metrics, resource listings) are frequently
truncated, causing the LLM to work with incomplete data and produce shallow
root cause analysis. With a 128K context window, the default budgets were
overly conservative.

**Root cause**: The defaults in `ols/constants.py` were:
- `DEFAULT_MAX_TOKENS_PER_TOOL_OUTPUT = 8000` (~6K words per tool)
- `DEFAULT_MAX_TOKENS_FOR_TOOLS = 32000` (total across all rounds)

The total budget is shared across all rounds and also includes tool definition
schemas. By round 3-4 the effective per-tool limit drops well below 8K,
making later tool calls nearly useless. For log or metrics output, even
the full 8K is often not enough for meaningful analysis.

**Fix**: Increase defaults to:
- `DEFAULT_MAX_TOKENS_PER_TOOL_OUTPUT = 16000`
- `DEFAULT_MAX_TOKENS_FOR_TOOLS = 48000`

These are still well within the 128K default context window (which reserves
4K for response), leaving plenty of room for the prompt, RAG context, and
conversation history. Users with smaller context windows can override via
model config (`max_tokens_per_tool_output` and `max_tokens_for_tools` in
the `parameters` section).

**Files changed**:
- `ols/constants.py` — updated `DEFAULT_MAX_TOKENS_PER_TOOL_OUTPUT` and
  `DEFAULT_MAX_TOKENS_FOR_TOOLS`

---

## 3. Increase MAX_ITERATIONS to allow more tool-calling rounds

**Problem**: The tool-calling agent loop in `iterate_with_tools` forces the LLM
to produce a text-only answer on the final round by unbinding tools
(`is_final_round = True` when `i == max_rounds`). With `MAX_ITERATIONS = 5`,
the LLM gets at most 4 rounds of actual tool use. For complex troubleshooting
that requires gathering logs, events, pod status, and metrics across multiple
resources, 4 rounds is not enough.

**Root cause**: `MAX_ITERATIONS` in `ols/constants.py` was set to 5. The
`iterate_with_tools` loop already has a natural exit condition: when the LLM
returns text without tool calls, `finish_reason == "stop"` triggers a return
at line ~347. So the forced final round is only a safety net against infinite
loops — it should be set high, not used as the primary exit condition.

**Fix**: Increase `MAX_ITERATIONS` from 5 to 15 in `ols/constants.py`. The
`is_final_round` mechanism is kept as a safety cap at the higher limit, but
the LLM is now expected to organically decide when to stop calling tools and
produce an answer in most cases, using the `finish_reason == "stop"` exit.

**Files changed**:
- `ols/constants.py` — changed `MAX_ITERATIONS` from 5 to 15

---

## 4. Improve agent system prompt for structured troubleshooting

**Problem**: `AGENT_INSTRUCTION_GENERIC` in `ols/customize/ols/prompts.py` was
a single vague line: "Given the user's query you must decide what to do with it
based on the list of tools provided to you." This gave the LLM no guidance on
investigation methodology. It tended to call one tool, get a result, and
immediately jump to a conclusion without cross-referencing related resources.

**Root cause**: The prompt lacked a structured reasoning framework. Without
explicit instructions to gather broad context first and synthesize after
collecting evidence, the LLM defaulted to a shallow single-tool-call pattern.

**Fix**: Expanded `AGENT_INSTRUCTION_GENERIC` with a lightweight structured
troubleshooting methodology:

1. Start broad — identify affected resources, namespace, scope.
2. Gather specifics — collect logs, events, status.
3. Cross-reference — check related resources (nodes, limits, recent changes).
4. Synthesize — provide root cause analysis only after sufficient evidence.

Added explicit instruction: "Do not jump to conclusions after a single tool
call. Use multiple tools to build a complete picture before answering."

`AGENT_INSTRUCTION_GRANITE` was left unchanged — it has its own format
tailored to Granite models.

**Files changed**:
- `ols/customize/ols/prompts.py` — expanded `AGENT_INSTRUCTION_GENERIC`

---

## 5. Improve tool output truncation to keep head and tail

**Problem**: When a tool output exceeded the token limit, `truncate_tool_output`
in `ols/utils/token_handler.py` truncated from the tail — keeping the beginning
and chopping the end. This is wrong for logs and events, where the most recent
(and usually most relevant) entries are at the end. The truncated output also
included a message saying "Please ask a more specific question," which is
unhelpful when the LLM should still attempt analysis with the available data.

**Root cause**: The method kept only a contiguous head portion of the output
and discarded everything after the token limit. For log-like output, this meant
the most recent and actionable entries were always lost.

**Fix**: Changed `truncate_tool_output` to keep both head (~40% of usable
tokens) and tail (~60%) with a gap marker in the middle:

```
[first ~40% of tokens]

[... OUTPUT TRUNCATED: {N} tokens removed from middle ...]

[last ~60% of tokens]
```

The 40/60 split favors the tail since recent data matters more for
troubleshooting. The gap marker tells the LLM how much was removed so it
knows the data is incomplete. The old "Please ask a more specific question"
message was removed — the LLM should still attempt analysis with what it has.

**Files changed**:
- `ols/utils/token_handler.py` — rewrote `truncate_tool_output` for
  head+tail truncation with gap marker
- `tests/unit/utils/test_token_handler.py` — updated truncation tests to
  verify both head and tail are preserved

---

## 6. Soften "be extremely concise" style guide in agent system prompt

**Problem**: The `AGENT_SYSTEM_INSTRUCTION` style guide in
`ols/customize/ols/prompts.py` told the LLM to "Be extremely concise" and
"Remove unnecessary words." This caused the LLM to strip out diagnostic
evidence and produce one-line answers that stated a root cause without showing
what was checked or what evidence supported the conclusion. Users had no way
to verify the answer or understand the reasoning.

**Root cause**: The style guide prioritized brevity over completeness. The
instruction "Remove unnecessary words" was interpreted by the LLM as license
to omit tool output references, checked resources, and supporting evidence —
all of which are necessary for a useful troubleshooting response.

**Fix**: Replaced the four-line style guide with three lines that preserve
conciseness as a goal while requiring diagnostic evidence:

- "Be concise but include all diagnostic evidence that supports your
  conclusion."
- "Show what was checked and what was found before stating the root cause."
- "Prioritize actionable details: root cause, affected resources, and fix."

This keeps responses focused (no filler) while ensuring the LLM includes
the reasoning chain from tool results to conclusion.

**Files changed**:
- `ols/customize/ols/prompts.py` — replaced style guide in
  `AGENT_SYSTEM_INSTRUCTION`

---

## 7. Truncate tool results in conversation history

**Problem**: After entry #1 (preserve tool-call history), follow-up turns
include full tool result content from prior turns in the conversation
history. Tool results can be 16K+ tokens each. After 2 turns with tool
calls, the prompt exceeds the context window (saw `Prompt length 76567
exceeds LLM available context window limit 75904 tokens`).

**Root cause**: `CacheEntry.cache_entries_to_history` in
`ols/app/models/models.py` reconstructs `ToolMessage(content=...)` with
the full tool output. For follow-up turns, the LLM already synthesized
these results into its previous answer — it doesn't need the full output
again. The token budget math: 128K context − 4K response − 48K tools =
76K available. Full tool results in history easily consume 30–40K of that.

**Fix**: Truncate each `ToolMessage` content to 200 characters when
reconstructing history. Content exceeding 200 chars gets a suffix
`... [truncated in history]`. This keeps the tool call structure visible
(the LLM knows it used tools and sees a preview of the result) without
burning 16K tokens per tool result on history.

**Files changed**:
- `ols/app/models/models.py` — added content truncation in
  `CacheEntry.cache_entries_to_history`
- `tests/unit/app/models/test_models.py` — added
  `test_cache_entries_to_history_tool_content_truncated`
