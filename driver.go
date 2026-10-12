package libsygo

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/dio/libsygo/jev"
)

// Driver is what a classifier reaches the host through: the runtime model pool,
// the judge, and the evidence attached to the outcome.
//
// Upstream's Driver also carries offloaded model calls. This port has none yet —
// the only host work a routing decision needs is the judge — so the pool, the
// judge and the evidence are what it carries.
type Driver struct {
	models   Models
	judge    Decider
	evidence json.RawMessage
	calls    int
}

// Models returns the models in a category, ordered best-first.
func (d *Driver) Models(category string) []string {
	if d.models == nil {
		return nil
	}
	return d.models[category]
}

// FirstModel returns the first model in a category, or "" when it is empty.
func (d *Driver) FirstModel(category string) string {
	if list := d.Models(category); len(list) > 0 {
		return list[0]
	}
	return ""
}

// CallDecision performs one judge call and counts it, so the outcome can report
// how many were made — which is what makes the session latch observable.
func (d *Driver) CallDecision(ctx context.Context, request jev.DecisionRequest) (jev.DecisionResponse, error) {
	d.calls++
	if d.judge == nil {
		return jev.DecisionResponse{}, errors.New("no judge decider was supplied for this call")
	}
	return d.judge.Call(ctx, request)
}

// Calls is how many judge calls this run has made.
func (d *Driver) Calls() int { return d.calls }

// SetEvidence replaces the outcome's evidence.
func (d *Driver) SetEvidence(value any) {
	d.evidence = mustJSON(value)
}

// SetEvidenceIfEmpty attaches evidence only when none is set, so a decision made
// earlier in a cascade is not overwritten by a later component.
func (d *Driver) SetEvidenceIfEmpty(value any) {
	if len(d.evidence) == 0 {
		d.SetEvidence(value)
	}
}

// Evidence is the JSON explaining the decision.
func (d *Driver) Evidence() json.RawMessage { return d.evidence }
