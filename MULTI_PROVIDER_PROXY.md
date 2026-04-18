# Multi-Provider Proxy Behavior

This branch extends `active-memory` beyond the Anthropic `/v1/messages` path used by Claude Code.

The proxy now supports three request families:

- Anthropic: `/v1/messages`
- OpenAI / Codex: `/v1/responses` and `/v1/chat/completions`
- Gemini: routes ending in `:generateContent` or `:streamGenerateContent`

## How It Works

The core memory system still operates on a normalized conversation:

- `user` messages
- `assistant` messages
- optional system text

Each provider route is adapted into that common form before active-memory runs. After the context is rewritten, the proxy converts the optimized conversation back into the provider's native request shape and forwards it upstream.

Assistant responses are also parsed per provider so the memory tree can ingest newly generated content immediately.

## Current Support

### Anthropic / Claude Code

This remains the most complete path.

- The proxy intercepts `/v1/messages`
- It rewrites the full message list
- It supports immediate assistant-response ingestion
- It supports Anthropic streaming ingestion with content-block reconstruction

This is still the closest thing to a transparent drop-in proxy.

### OpenAI / Codex

The proxy supports:

- stateless `/v1/responses` requests that provide explicit `input`
- `/v1/chat/completions`
- non-streaming response ingestion
- best-effort streaming response ingestion

For `responses.create(...)`, the proxy extracts:

- `instructions` as system text
- `input` message items as the conversation history

It then rewrites the request with the optimized conversation before forwarding it.

### Gemini

The proxy supports:

- `:generateContent`
- `:streamGenerateContent`
- non-streaming response ingestion
- best-effort streaming response ingestion

The proxy extracts:

- `config.system_instruction` as system text
- `contents` entries as the conversation history

It rewrites `contents` with the optimized conversation and forwards the request upstream.

## Important Limitation

Anthropic-style clients generally send the full visible conversation on each request. That makes active-memory proxying straightforward: the proxy can replace the raw history with a curated one.

Some OpenAI/Codex workflows use server-side conversation state, especially via `previous_response_id`. In that mode, the upstream provider is carrying hidden context that the proxy cannot safely rewrite.

Because of that, this branch treats OpenAI Responses requests with `previous_response_id` as passthrough.

That means:

- stateless OpenAI/Codex requests work well
- stateful server-side conversation chaining is not yet fully memory-managed by the proxy

## Practical Summary

- Claude Code via Anthropic proxying: full support
- Codex via stateless OpenAI Responses requests: supported
- Codex via `previous_response_id` chained server-side state: passthrough
- Gemini generate-content requests: supported

## Recommended Use

Use the proxy when your client sends explicit conversation history in the request.

If your client depends heavily on provider-managed hidden state, use the provider-aware `active-memory-chat` interface instead, or expect some requests to fall back to passthrough until deeper provider-specific state handling is added.
