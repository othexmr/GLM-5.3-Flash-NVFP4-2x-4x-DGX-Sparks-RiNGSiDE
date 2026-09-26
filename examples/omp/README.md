# oh-my-pi (omp) client example

[`models.yml`](models.yml) is a provider entry for [oh-my-pi](https://github.com/can1357/oh-my-pi) (`omp`, written for
18.2.6) that talks to the OpenAI-compatible server of either profile.

1. Check the server: `curl -s http://HEAD_NODE:8888/v1/models` lists `nvidia/GLM-5.3-Flash-NVFP4` (`HEAD_NODE` is
   rank 0's address, as in [docs/operations.md](../../docs/operations.md)).
2. Merge the `dgx-sparks` provider into `~/.omp/agent/models.yml`, next to any providers already there, and replace
   `HEAD_NODE`. If the port differs from 8888, use `SWITCHLESS_API_PORT` from the site file.
3. Pick the `dgx-sparks` provider and the `GLM-5.3-Flash-NVFP4` model in omp.

Thinking levels become the request's `reasoning_effort`:

| omp level | `reasoning_effort` | on the server |
|---|---|---|
| `minimal` | `none` | the answer comes without a reasoning block |
| `low` (default) | `low` | the chat template asks for low reasoning effort |
| `high` | `high` | high reasoning effort |
| `max` | `max` | maximum effort, also the template's default when a request sends no effort |

The reasoning text arrives in each message's `reasoning` field (`reasoningContentField`). `replayReasoningContent: true`
sends it back with the history. omp leaves it off for servers that are not on localhost, and without it every tool step
starts without the reasoning of the steps before, so the model plans the task again at each step. The chat template keeps
the reasoning of the turns after the last user message and drops that of earlier turns (the server's default
`clear_thinking`), so the prompt grows by the reasoning of the current task only.

**Output budget.** `omitMaxOutputTokens: true` makes omp send no `max_tokens`, so the server lets each answer use
whatever is left of the 262,144-token window: long thinking at the `high` and `max` levels never stops at a fixed cap,
and the whole window stays available for the history (omp compacts it as the window fills). With a fixed `maxTokens`
instead, keep `contextWindow + maxTokens` at or below 262,144: the server rejects a request whose prompt plus
`max_tokens` exceeds the window, and omp sends `maxTokens` unchanged with every request. At `high`, one planning step
of a large coding task has thought for more than 32,768 tokens.

The server has no authentication and no TLS: reach it only over a trusted network or through an authenticating proxy
(docs/operations.md, "API trust boundary").
