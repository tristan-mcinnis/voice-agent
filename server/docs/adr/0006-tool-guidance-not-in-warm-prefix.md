# ADR-0006: Tool `guidance` is metadata, not auto-injected prompt

**Status:** accepted

**Date:** 2026-05-14

## Context

`BaseTool` carries an optional `guidance: str` field — a free-form prose
block intended to teach the LLM how to use that specific tool. Until now
`PromptBuilder._collect_tool_guidance()` concatenated every registered
tool's guidance into the frozen system prompt at session start.

This silently undermined the cache-warm prefix guarantee that the
`PromptBuilder` is built around (see ADR-0003 on voice latency). Two
concrete problems:

1. **Cost scales with tool count, not session need.** A session that only
   reads the clipboard pays for the full computer-use workflow protocol,
   the memory-tool's add/replace/list playbook, and every other tool's
   workflow notes — every turn, until the cache is invalidated.

2. **Tool authors had no signal that adding `guidance` was costly.** The
   field looked free; it was not.

The LLM's function-selection step already reads each tool's `description`
field from the tools schema, which providers cache independently.

## Decision

`PromptBuilder` no longer auto-injects `guidance` into the system prompt.
The `guidance` field stays on `BaseTool` as metadata — tests and future
opt-in surfaces (e.g. an on-first-use injection mechanism) can still read
it — but the warm prefix only carries: Soul → Memory → User → Project
Rules → Tools (the inventory) → Skills.

Load-bearing tool workflows belong in the tool's `description` field, where
the LLM picks them up at function-selection time.

## Consequences

- Cache-warm prefix size becomes independent of tool count.
- Voice latency stays bounded by the (smaller, stable) cognitive stack.
- Tools whose `description` does not already encode the protocol they
  expect (e.g. "call X before Y") must move that protocol into the
  description. As of this ADR, every existing tool's description already
  conveys its workflow.
- The `guidance` field is preserved on `BaseTool` so we can opt into
  lazy injection later (e.g. on first invocation of a tool, append its
  guidance to the next assistant turn's context) without re-introducing
  it to the warm prefix.
