package main

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/bedrock"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/aws/aws-sdk-go-v2/config"
)

// anthropicModel keeps the demo snappy and cheap; swap to
// anthropic.ModelClaudeOpus4_8 for maximum capability.
const anthropicModel = anthropic.Model("claude-sonnet-5-5")

// bedrockModel is the same model served by Amazon Bedrock. It is a cross-region
// inference profile id, not a bare model id: Bedrock serves current Claude models
// only through a profile, and "global." routes to whichever region has capacity
// at no premium over a single-region profile.
const bedrockModel = anthropic.Model("global.anthropic.claude-sonnet-5-5")

// anthropicBrain is the Claude backend, reached either directly (ANTHROPIC_API_KEY
// or an ambient ant profile) or through Amazon Bedrock; both speak the same
// Messages API, so only the client and the model id differ. Unlike a chatbot it
// keeps no transcript: each file is a fresh, independent tool call.
type anthropicBrain struct {
	client anthropic.Client
	model  anthropic.Model
	via    string // "anthropic" or "bedrock", for the banner
}

// newAnthropicBrain builds the Claude backend. When apiKey is non-empty (from
// --anthropic-api-key) it's passed explicitly; otherwise the SDK falls back to
// its usual ANTHROPIC_API_KEY / ambient profile resolution.
func newAnthropicBrain(apiKey string) *anthropicBrain {
	var opts []option.RequestOption
	if apiKey != "" {
		opts = append(opts, option.WithAPIKey(apiKey))
	}
	return &anthropicBrain{client: anthropic.NewClient(opts...), model: anthropicModel, via: "anthropic"}
}

// newBedrockBrain builds the Claude backend on Amazon Bedrock, with requests signed
// by AWS credentials instead of an Anthropic key.
//
// profile is applied as an explicit option rather than left to AWS_PROFILE because
// an explicitly chosen profile outranks AWS_ACCESS_KEY_ID in the environment, and
// AWS_PROFILE does not: with bare keys exported, AWS_PROFILE is silently ignored
// and the calls run, without error, in whatever account those keys belong to.
func newBedrockBrain(ctx context.Context, region, profile string) *anthropicBrain {
	loadOpts := []func(*config.LoadOptions) error{config.WithRegion(region)}
	if profile != "" {
		loadOpts = append(loadOpts, config.WithSharedConfigProfile(profile))
	}
	client := anthropic.NewClient(bedrock.WithLoadDefaultConfig(ctx, loadOpts...))
	return &anthropicBrain{client: client, model: bedrockModel, via: "bedrock"}
}

func (b *anthropicBrain) label() string { return b.via + "/" + string(b.model) }

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
	// Sonnet 5.5 rejects a forced tool_choice ("tool" or "any") with a 400, so the
	// choice stays "auto" and the system prompt asks for the call; a reply without
	// it is an error. Thinking is left at the model's default (adaptive), which
	// max_tokens leaves room for.
	resp, err := b.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     b.model,
		MaxTokens: 8192,
		System:    []anthropic.TextBlockParam{{Text: "You are a codebase steward that reads source files and reports precise, structured analysis for a knowledge graph. Answer only by calling the analyze_file tool."}},
		Messages:  []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock(msg))},
		Tools:     tools,
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
