package main

import (
	"context"
	"fmt"
	"os"
	"strings"
)

// brain is the pluggable analysis LLM. Given one source file's path and content,
// it returns a structured reading of that file: a prose summary, the symbols it
// defines, the dependencies it imports, and any notable design decisions. It talks
// to Claude or Gemini; everything Jennah-facing is identical regardless of which
// brain answers — that's the point of the demo. The brain holds no session state:
// each file is analyzed independently, so the steward can fan over a tree freely.
type brain interface {
	analyzeFile(ctx context.Context, relpath, content string) (fileAnalysis, error)
	label() string // short "provider/model" string for the startup banner
}

// fileAnalysis is the structured output the model is forced to produce for each
// file. commitFile turns it into a vector chunk (Summary) plus graph nodes/edges
// (Symbols → DEFINES, Imports → IMPORTS, Decisions → NOTES).
type fileAnalysis struct {
	Summary   string     `json:"summary"`
	Symbols   []symbol   `json:"symbols"`
	Imports   []string   `json:"imports"`
	Decisions []decision `json:"decisions"`
}

type symbol struct {
	Name string `json:"name"`
	Kind string `json:"kind"`
}

type decision struct {
	Title     string `json:"title"`
	Rationale string `json:"rationale"`
}

// The analyze_file tool, described once and mapped into each SDK's own tool type
// so the two backends stay in lockstep. The model is FORCED to call it (Anthropic
// tool_choice, Gemini function-calling mode ANY), so every analysis comes back as
// one structured payload we parse straight into fileAnalysis.
const (
	toolName = "analyze_file"
	toolDesc = "Report a structured reading of ONE source file for a codebase knowledge graph. Summarize what the file does, list the top-level symbols it defines, the modules/packages it imports, and any notable architectural decisions or trade-offs evident in the code. Be precise and terse; prefer real identifiers over prose."

	sumDesc  = "one-paragraph summary of the file's purpose and responsibilities"
	symDesc  = "top-level symbols the file DEFINES (functions, types, methods, constants); omit imported/external names"
	nameDesc = "the symbol's identifier, e.g. 'commitFile', 'jennahClient'"
	kindDesc = "the symbol kind: function|method|type|struct|interface|const|var"
	impDesc  = "modules or packages this file IMPORTS (import paths or package names), e.g. 'net/http', 'google.golang.org/protobuf/proto'"
	decDesc  = "notable design decisions or trade-offs evident in the file; empty if none stand out"
	dtDesc   = "short title of the decision, e.g. 'protojson over the gateway'"
	drDesc   = "one sentence on why it was done this way / the trade-off"
)

// newBrain selects the analysis provider. "auto" prefers Anthropic when an
// Anthropic key is present, else Gemini — so someone with only one key set just
// runs `go run .`. anthropicKey, when non-empty, is the Anthropic API key from
// --anthropic-api-key (already defaulted to $ANTHROPIC_API_KEY); it overrides the
// SDK's own env lookup.
func newBrain(ctx context.Context, provider, anthropicKey string) (brain, error) {
	if provider == "auto" {
		switch {
		case anthropicKey != "":
			provider = "anthropic"
		case os.Getenv("GEMINI_API_KEY") != "" || os.Getenv("GOOGLE_API_KEY") != "" || useVertexAI():
			provider = "gemini"
		default:
			return nil, fmt.Errorf("no analysis credentials found: set GEMINI_API_KEY / Vertex AI env (Gemini) or pass --anthropic-api-key / set ANTHROPIC_API_KEY (Anthropic), or pass --provider")
		}
	}
	switch strings.ToLower(provider) {
	case "gemini":
		return newGeminiBrain(ctx)
	case "anthropic", "claude":
		return newAnthropicBrain(anthropicKey), nil
	default:
		return nil, fmt.Errorf("unknown --provider %q (want auto|gemini|anthropic)", provider)
	}
}
