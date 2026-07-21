package main

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// anthropicModel keeps the demo snappy and cheap; swap to
// anthropic.ModelClaudeOpus4_8 for maximum capability.
const anthropicModel = anthropic.Model("claude-sonnet-5")

// anthropicBrain is the Claude backend. Unlike a chatbot it keeps no transcript:
// each file is a fresh, independent forced-tool call.
type anthropicBrain struct {
	client anthropic.Client
}

// newAnthropicBrain builds the Claude backend. When apiKey is non-empty (from
// --anthropic-api-key) it's passed explicitly; otherwise the SDK falls back to
// its usual ANTHROPIC_API_KEY / ambient profile resolution.
func newAnthropicBrain(apiKey string) *anthropicBrain {
	var opts []option.RequestOption
	if apiKey != "" {
		opts = append(opts, option.WithAPIKey(apiKey))
	}
	return &anthropicBrain{client: anthropic.NewClient(opts...)}
}

func (b *anthropicBrain) label() string { return "anthropic/" + string(anthropicModel) }

func (b *anthropicBrain) analyzeFile(ctx context.Context, relpath, content string) (fileAnalysis, error) {
	tools := []anthropic.ToolUnionParam{{OfTool: &anthropic.ToolParam{
		Name:        toolName,
		Description: anthropic.String(toolDesc),
		InputSchema: anthropic.ToolInputSchemaParam{
			Properties: map[string]any{
				"summary": map[string]any{"type": "string", "description": sumDesc},
				"symbols": map[string]any{
					"type": "array", "description": symDesc,
					"items": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"name": map[string]any{"type": "string", "description": nameDesc},
							"kind": map[string]any{"type": "string", "description": kindDesc},
						},
						"required": []string{"name", "kind"},
					},
				},
				"imports": map[string]any{
					"type": "array", "description": impDesc,
					"items": map[string]any{"type": "string"},
				},
				"decisions": map[string]any{
					"type": "array", "description": decDesc,
					"items": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"title":     map[string]any{"type": "string", "description": dtDesc},
							"rationale": map[string]any{"type": "string", "description": drDesc},
						},
						"required": []string{"title", "rationale"},
					},
				},
			},
			Required: []string{"summary", "symbols", "imports", "decisions"},
		},
	}}}

	msg := fmt.Sprintf("File path: %s\n\n```\n%s\n```", relpath, content)
	resp, err := b.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:      anthropicModel,
		MaxTokens:  2048,
		System:     []anthropic.TextBlockParam{{Text: "You are a codebase steward that reads source files and reports precise, structured analysis for a knowledge graph. Always answer by calling the analyze_file tool."}},
		Messages:   []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock(msg))},
		Tools:      tools,
		ToolChoice: anthropic.ToolChoiceParamOfTool(toolName),
	})
	if err != nil {
		return fileAnalysis{}, err
	}
	for _, blk := range resp.Content {
		if v, ok := blk.AsAny().(anthropic.ToolUseBlock); ok {
			var fa fileAnalysis
			if err := json.Unmarshal([]byte(v.JSON.Input.Raw()), &fa); err != nil {
				return fileAnalysis{}, fmt.Errorf("decode analysis: %w", err)
			}
			return fa, nil
		}
	}
	return fileAnalysis{}, fmt.Errorf("model returned no analyze_file tool call")
}
