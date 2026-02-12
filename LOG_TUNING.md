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
