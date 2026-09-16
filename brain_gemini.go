package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"google.golang.org/genai"
)

// geminiModel keeps the demo snappy and cheap; swap to "gemini-2.5-pro" for
// maximum capability. The same id works on both the AI Studio (API-key) and
// Vertex AI backends.
const geminiModel = "gemini-3.8-flash"

// geminiBrain is the Google Gemini backend. It talks to either AI Studio (an API
// key in GEMINI_API_KEY / GOOGLE_API_KEY) or Vertex AI (GCP project + location +
// Application Default Credentials).
type geminiBrain struct {
	client  *genai.Client
	backend string // for the startup banner, e.g. "vertex:my-proj/us-central1"
	config  *genai.GenerateContentConfig
}

// useVertexAI reports whether to route Gemini through Vertex AI rather than the
// AI Studio API-key path: either explicitly (GOOGLE_GENAI_USE_VERTEXAI=1|true),
// or implicitly when a GCP project is configured and no Studio key is set.
func useVertexAI() bool {
	switch strings.ToLower(os.Getenv("GOOGLE_GENAI_USE_VERTEXAI")) {
	case "1", "true":
		return true
	}
	return os.Getenv("GEMINI_API_KEY") == "" && os.Getenv("GOOGLE_API_KEY") == "" &&
		os.Getenv("GOOGLE_CLOUD_PROJECT") != ""
}

func newGeminiBrain(ctx context.Context) (*geminiBrain, error) {
	cc := &genai.ClientConfig{Backend: genai.BackendGeminiAPI}
	backend := "ai-studio"
	if useVertexAI() {
		// Vertex uses Application Default Credentials (run once:
		//   gcloud auth application-default login
		// or set GOOGLE_APPLICATION_CREDENTIALS to a service-account key) — no
		// API key. Project/location come from the standard GCP env vars.
		project := os.Getenv("GOOGLE_CLOUD_PROJECT")
		if project == "" {
			return nil, fmt.Errorf("Vertex AI selected but GOOGLE_CLOUD_PROJECT is not set (and run: gcloud auth application-default login)")
		}
		location := envOr("GOOGLE_CLOUD_LOCATION", os.Getenv("GOOGLE_CLOUD_REGION"))
		if location == "" {
			location = "global" // Gemini 2.5 is served on the global endpoint
		}
		cc.Backend = genai.BackendVertexAI
		cc.Project = project
		cc.Location = location
		backend = "vertex:" + project + "/" + location
	}
	client, err := genai.NewClient(ctx, cc)
	if err != nil {
		return nil, fmt.Errorf("gemini client: %w", err)
	}
	config := &genai.GenerateContentConfig{
		MaxOutputTokens:   2048,
		SystemInstruction: genai.NewContentFromText("You are a codebase steward that reads source files and reports precise, structured analysis for a knowledge graph. Always answer by calling the analyze_file tool.", genai.RoleUser),
		// Force the tool: mode ANY makes the model always emit a function call.
		ToolConfig: &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeAny},
		},
		Tools: []*genai.Tool{{
			FunctionDeclarations: []*genai.FunctionDeclaration{{
				Name:        toolName,
				Description: toolDesc,
				Parameters: &genai.Schema{
					Type: genai.TypeObject,
					Properties: map[string]*genai.Schema{
						"summary": {Type: genai.TypeString, Description: sumDesc},
						"symbols": {
							Type: genai.TypeArray, Description: symDesc,
							Items: &genai.Schema{
								Type: genai.TypeObject,
								Properties: map[string]*genai.Schema{
									"name": {Type: genai.TypeString, Description: nameDesc},
									"kind": {Type: genai.TypeString, Description: kindDesc},
								},
								Required: []string{"name", "kind"},
							},
						},
						"imports": {
							Type: genai.TypeArray, Description: impDesc,
							Items: &genai.Schema{Type: genai.TypeString},
						},
						"decisions": {
							Type: genai.TypeArray, Description: decDesc,
							Items: &genai.Schema{
								Type: genai.TypeObject,
								Properties: map[string]*genai.Schema{
									"title":     {Type: genai.TypeString, Description: dtDesc},
									"rationale": {Type: genai.TypeString, Description: drDesc},
								},
								Required: []string{"title", "rationale"},
							},
						},
					},
					Required: []string{"summary", "symbols", "imports", "decisions"},
				},
			}},
		}},
	}
	return &geminiBrain{client: client, backend: backend, config: config}, nil
}

func (b *geminiBrain) label() string { return "gemini/" + geminiModel + " (" + b.backend + ")" }

func (b *geminiBrain) analyzeFile(ctx context.Context, relpath, content string) (fileAnalysis, error) {
	msg := fmt.Sprintf("File path: %s\n\n```\n%s\n```", relpath, content)
	resp, err := b.client.Models.GenerateContent(ctx, geminiModel,
		[]*genai.Content{genai.NewContentFromText(msg, genai.RoleUser)}, b.config)
	if err != nil {
		return fileAnalysis{}, err
	}
	if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
		return fileAnalysis{}, fmt.Errorf("model returned no content")
	}
	for _, part := range resp.Candidates[0].Content.Parts {
		if part.FunctionCall == nil {
			continue
		}
		a := part.FunctionCall.Args
		return fileAnalysis{
			Summary:   asString(a["summary"]),
			Symbols:   toSymbols(a["symbols"]),
			Imports:   toStrings(a["imports"]),
			Decisions: toDecisions(a["decisions"]),
		}, nil
	}
	return fileAnalysis{}, fmt.Errorf("model returned no analyze_file function call")
}

// Gemini hands tool args back as generic map[string]any, so unpack by hand.
func asString(v any) string { s, _ := v.(string); return s }

func toStrings(v any) []string {
	arr, _ := v.([]any)
	out := make([]string, 0, len(arr))
	for _, e := range arr {
		if s := asString(e); s != "" {
			out = append(out, s)
		}
	}
	return out
}

func toSymbols(v any) []symbol {
	arr, _ := v.([]any)
	out := make([]symbol, 0, len(arr))
	for _, e := range arr {
		m, _ := e.(map[string]any)
		if m == nil {
			continue
		}
		if name := asString(m["name"]); name != "" {
			out = append(out, symbol{Name: name, Kind: asString(m["kind"])})
		}
	}
	return out
}

func toDecisions(v any) []decision {
	arr, _ := v.([]any)
	out := make([]decision, 0, len(arr))
	for _, e := range arr {
		m, _ := e.(map[string]any)
		if m == nil {
			continue
		}
		if t := asString(m["title"]); t != "" {
			out = append(out, decision{Title: t, Rationale: asString(m["rationale"])})
		}
	}
	return out
}
