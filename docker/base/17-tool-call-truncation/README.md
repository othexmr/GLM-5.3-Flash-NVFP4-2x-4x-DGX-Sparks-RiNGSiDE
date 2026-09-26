# Stage 17: truncated tool calls are delivered as valid JSON

On the shipped auto tool path, a tool call whose generation stops inside an `<arg_value>` was streamed
with `arguments` ending mid-string (not JSON) while the non-streaming parse returned `{}`, and both were
labelled complete. The repair (two parser engine files, exact preimage/after hashes in
`source-manifest.json`) makes `_final_arg_json` fall back to the partial conversion when the
non-partial conversion does not extend the streamed prefix, adds an escape-aware last-resort closer that
validates with `json.loads`, routes the non-streaming path through the same function, and exposes
`tool_call_truncated` / `tool_end_synthesized` as read-only signals. Untruncated calls are byte-identical
(516 combinations compared). Serving-side wiring of the truncation signal into `finish_reason` and the
Responses `status` is a separate change and is not included.

`test_cpu.py` (with `harness.py`) drives the installed engine modules through `parse_delta` and `parse`:
12 model outputs × delta modes × chunkings = 1,410 streamed calls. In the real image the harness's
'tool name' check reads a field the installed engine does not populate, so it fails 1,305 cases before
and after the install alike (0/1,410 on the packet's shimmed Mac run); `gate.py` therefore requires
the three truncation checks (streamed arguments parse as JSON; streaming and non-streaming agree; the
parser exposes the truncation signal) to fall from their pre-install counts (116/116/1 on the tested
image) to zero, the harness artifact to stay unchanged, and no new check to fail, then the build reruns
every stage-12 suite.
No GPU code, kernel, scheduler or cache path changes; determinism and throughput are untouched.
