package fluxevolve

import (
	"math/rand"
	"testing"
	"time"
)

func init() {
	rand.Seed(42)
}

func TestNewEngineDefaults(t *testing.T) {
	e := NewEngine()
	if e.FitnessThreshold != 0.3 {
		t.Errorf("threshold = %v, want 0.3", e.FitnessThreshold)
	}
	if e.MutationProbability != 0.1 {
		t.Errorf("prob = %v, want 0.1", e.MutationProbability)
	}
	if e.EliteThreshold != 0.8 {
		t.Errorf("elite = %v, want 0.8", e.EliteThreshold)
	}
	if e.Behaviors == nil {
		t.Error("Behaviors should be initialized")
	}
	if e.Generation != 0 {
		t.Errorf("generation = %v, want 0", e.Generation)
	}
}

func TestAddBehavior(t *testing.T) {
	e := NewEngine()
	e.AddBehavior("x", 5.0, 0.0, 10.0, 0.1)
	b := e.Behaviors["x"]
	if b == nil {
		t.Fatal("behavior not found")
	}
	if b.Value != 5.0 || b.Min != 0.0 || b.Max != 10.0 || b.MutationRate != 0.1 {
		t.Errorf("got %+v", b)
	}
}

func TestAddBehaviorClampsValue(t *testing.T) {
	e := NewEngine()
	e.AddBehavior("x", 15.0, 0.0, 10.0, 0.1)
	if e.Behaviors["x"].Value != 10.0 {
		t.Errorf("value = %v, want 10.0", e.Behaviors["x"].Value)
	}
}

func TestFindBehavior(t *testing.T) {
	e := NewEngine()
	e.AddBehavior("a", 1, 0, 10, 0.1)
	if e.FindBehavior("a") == nil {
		t.Error("should find a")
	}
	if e.FindBehavior("missing") != nil {
		t.Error("should not find missing")
	}
}

func TestGetFound(t *testing.T) {
	e := NewEngine()
	e.AddBehavior("v", 7.5, 0, 10, 0.1)
	if v := e.Get("v"); v != 7.5 {
		t.Errorf("got %v", v)
	}
}

func TestGetMissing(t *testing.T) {
	e := NewEngine()
	if e.Get("nope") != -1 {
		t.Error("missing should return -1")
	}
}

func TestSetClamps(t *testing.T) {
	e := NewEngine()
	e.AddBehavior("s", 5, 0, 10, 0.1)
	e.Set("s", 100)
	if e.Behaviors["s"].Value != 10 {
		t.Errorf("got %v", e.Behaviors["s"].Value)
	}
	e.Set("s", -5)
	if e.Behaviors["s"].Value != 0 {
		t.Errorf("got %v", e.Behaviors["s"].Value)
	}
}

func TestSetMissing(t *testing.T) {
	e := NewEngine()
	e.Set("nope", 5) // should not panic
}

func TestScore(t *testing.T) {
	e := NewEngine()
	e.AddBehavior("a", 5, 0, 10, 0.1)
	e.Score("a", 1.0)
	e.Score("a", 3.0)
	b := e.Behaviors["a"]
	if b.Uses != 2 {
		t.Errorf("uses = %v", b.Uses)
	}
	if b.CumulativeScore != 4.0 {
		t.Errorf("score = %v", b.CumulativeScore)
	}
}

func TestScoreMissing(t *testing.T) {
	e := NewEngine()
	e.Score("nope", 5.0) // should not panic
}

func TestCycleIncrementsGeneration(t *testing.T) {
	e := NewEngine()
	e.Cycle(time.Now(), 0.5)
	if e.Generation != 1 {
		t.Errorf("gen = %v", e.Generation)
	}
}

func TestCycleEliteNoMutation(t *testing.T) {
	e := NewEngine()
	e.AddBehavior("x", 5, 0, 10, 0.1)
	e.AddBehavior("y", 5, 0, 10, 0.1)
	m := e.Cycle(time.Now(), 0.9)
	if m != 0 {
		t.Errorf("elite mutations = %v, want 0", m)
	}
}

func TestCycleLowFitness(t *testing.T) {
	e := NewEngine()
	for i := 0; i < 10; i++ {
		e.AddBehavior("b"+string(rune('A'+i)), 5, 0, 10, 0.5)
	}
	m := e.Cycle(time.Now(), 0.1)
	if m == 0 {
		t.Error("expected some mutations with low fitness")
	}
}

func TestCycleUpdatesFitnessScore(t *testing.T) {
	e := NewEngine()
	e.Cycle(time.Now(), 0.6)
	if e.FitnessScore != 0.6 {
		t.Errorf("fitness = %v", e.FitnessScore)
	}
}

func TestRevert(t *testing.T) {
	e := NewEngine()
	e.AddBehavior("x", 5, 0, 10, 1.0)
	e.Cycle(time.Now(), 0.1)
	if len(e.History) == 0 {
		t.Fatal("no history")
	}
	oldVal := e.History[0].OldValue
	e.Behaviors["x"].Value = 999 // force change to prove revert
	ok := e.Revert(0)
	if !ok {
		t.Error("revert failed")
	}
	if e.Behaviors["x"].Value != oldVal {
		t.Errorf("value = %v, want %v", e.Behaviors["x"].Value, oldVal)
	}
	if !e.History[0].Reverted {
		t.Error("should be marked reverted")
	}
}

func TestRevertAlreadyReverted(t *testing.T) {
	e := NewEngine()
	e.AddBehavior("x", 5, 0, 10, 0.1)
	e.Cycle(time.Now(), 0.1)
	e.Revert(0)
	if e.Revert(0) {
		t.Error("double revert should fail")
	}
}

func TestRevertInvalidIndex(t *testing.T) {
	e := NewEngine()
	if e.Revert(-1) || e.Revert(0) || e.Revert(99) {
		t.Error("invalid index should fail")
	}
}

func TestRollback(t *testing.T) {
	e := NewEngine()
	e.AddBehavior("x", 5, 0, 10, 1.0)
	e.Cycle(time.Now(), 0.1) // gen 1
	e.Cycle(time.Now(), 0.1) // gen 2
	e.Cycle(time.Now(), 0.1) // gen 3
	reverts := e.Rollback(1)
	if reverts == 0 {
		t.Error("expected reverts")
	}
	if e.Generation != 1 {
		t.Errorf("gen = %v, want 1", e.Generation)
	}
}

func TestBestAndWorst(t *testing.T) {
	e := NewEngine()
	e.AddBehavior("good", 5, 0, 10, 0.1)
	e.AddBehavior("bad", 5, 0, 10, 0.1)
	e.Score("good", 10.0)
	e.Score("good", 10.0)
	e.Score("bad", -5.0)
	e.Score("bad", -5.0)
	best := e.BestBehaviors(1)
	worst := e.WorstBehaviors(1)
	if best[0].Name != "good" {
		t.Errorf("best = %v", best[0].Name)
	}
	if worst[0].Name != "bad" {
		t.Errorf("worst = %v", worst[0].Name)
	}
}

func TestBestWorstEmpty(t *testing.T) {
	e := NewEngine()
	if len(e.BestBehaviors(5)) != 0 {
		t.Error("empty engine should return empty")
	}
}

func TestMutationsCounter(t *testing.T) {
	e := NewEngine()
	e.AddBehavior("x", 5, 0, 10, 1.0)
	e.Cycle(time.Now(), 0.1)
	if e.MutationsTotal == 0 {
		t.Error("should have mutations")
	}
}
