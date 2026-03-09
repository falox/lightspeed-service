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

Tool result content is truncated to 200 characters with a `[truncated in history]`
suffix to prevent full outputs (16K+ tokens each) from exceeding the context
window on follow-up turns.

**Files changed**:
- `ols/app/models/models.py` — `CacheEntry.cache_entries_to_history`, added
  `ToolMessage` import, truncation of tool result content
- `tests/unit/app/models/test_models.py` — added tests for tool-call history
  reconstruction, truncation, and mixed (tool + non-tool) conversation history

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
