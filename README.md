# libsygo

A pure-Go implementation of [Switchyard](https://github.com/NVIDIA-NeMo/Switchyard)'s
`libsy`: the algorithms that decide **which model a request should route to**.

No cgo, no Rust toolchain, no FFI, no subprocess. Standard library only.

## Why a port rather than a binding

Upstream libsy depends on exactly one Switchyard crate — `switchyard-protocol`,
the shared vocabulary. The provider parsing and transport live in *separate*
crates (`switchyard-translation`, `switchyard-llm-client`), so the algorithms are
a self-contained bundle with nothing peripheral tangled in. That is what makes a
port coherent, and it is why this repository exists rather than a C ABI.

What a port cannot copy is upstream's **tuning** — thresholds, hold turns, the
tool vocabulary. Those are measured against benchmarks, not derivable, so every
behaviour here is ported and diffed rather than invented. See *Conformance*.

## The seam

`libsygo` performs no I/O. The judge call is supplied by the host:

```go
type Decider interface {
    Call(ctx context.Context, request jev.DecisionRequest) (jev.DecisionResponse, error)
}
```

So a dataplane that owns its egress — Envoy, or anything with an HTTP stack —
keeps TLS, credentials, retries and timeouts host-side. `jev.Client` satisfies
`Decider`, so a host with no stack of its own can use it directly.

## Usage

```go
router, err := libsygo.New(
    libsygo.Profile{
        Algorithm: "llm_classifier",
        Mode:      "capability",
        Judge: &libsygo.Judge{
            Type:       "decision",
            Cutoff:     0.4,
            Candidates: map[string]string{"a": "smart", "b": "cheap"},
            Evidence:   json.RawMessage(`{"candidates":{"a":"capable tier","b":"efficient tier"}}`),
        },
        SessionAffinity: true,
    },
    libsygo.Models{
        "capable":   {"smart"},
        "efficient": {"cheap"},
        "judge":     {"jev-latest"},
        "any":       {"cheap", "smart"}, // required: see below
    },
)
if err != nil {
    return err
}

outcome := router.Route(ctx, libsygo.Turn{
    Request: libsygo.TextRequest(prompt), // or DecodeOpenAIChat(body)
    Session: tenantID + "/" + sessionID,  // must be tenant-namespaced
}, decider)

// outcome.Model is the pick; outcome.SelectedModelIDs is the fallback order;
// outcome.Evidence explains the decision.
```

A runnable version, with a stand-in judge and no cloud credentials:
`go run ./examples/judge`.

### The `any` invariant

`Models["any"]` must list **every** routable target, and every other category
must be a subset of it — upstream resolves the selected target against `any` and
otherwise fails mid-route with `TargetNotFound`. `New` rejects a violating pool at
construction instead. Order matters: `any` is the fallback order after the chosen
tier.

### Tenancy

One `Router` holds one profile's per-session latch, so cache it by
`(tenant, profile ID, revision)` and namespace session ids per tenant. Two tenants
sharing a session id would otherwise share retained routing state.

## Conformance

Verified against upstream libsy turn for turn by the harness in
[`github.com/dio/switchyard-capi`](https://github.com/dio/switchyard-capi): the
same corpus, the same recorded golden, diffed on tier, ordered candidates,
evidence, and the rewritten request. Upstream's own tests are the specification.

## Scope

Implemented: the capability classifier over a decision (System One / JEV) judge,
with the composition it needs — the cascade, the per-session latch, the cutoff,
fail-open, candidate ordering, evidence.

Not implemented, and refused at construction rather than approximated:

| Family | Status |
|---|---|
| Signal router (tool signals, stage) | not ported — the largest family, and it needs the normalized request's tools/messages |
| Escalation | not ported |
| Advisor gate, plan/execute, composite, sub-agent | not ported |
| Custom classifiers, LLM-text judges | not ported |

Several of those read tool calls and message content out of a normalized request,
so a host handing this library only a prompt can run the judge path and nothing
else.

## Licence

Apache-2.0. Portions derive from Switchyard (NVIDIA, Apache-2.0); see `NOTICE`.
