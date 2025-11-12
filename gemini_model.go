package contextwindow

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"google.golang.org/genai"
)

const (
	ModelGemini20Flash     = "gemini-2.0-flash"
	ModelGemini25Flash     = "gemini-2.5-flash"
	ModelGemini20FlashExp  = "gemini-2.0-flash-exp"
	ModelGemini25Pro       = "gemini-2.5-pro"
	ModelGemini25FlashLite = "gemini-2.5-flash-lite"
)

var AllGeminiModels = []string{
	ModelGemini20Flash,
	ModelGemini20FlashExp,
	ModelGemini25Flash,
	ModelGemini25Pro,
	ModelGemini25FlashLite,
}

type GeminiModel struct {
	client       *genai.Client
	model        string
	middleware   []Middleware
	toolExecutor ToolExecutor
}

func NewGeminiModel(model string) (*GeminiModel, error) {
	apiKey := os.Getenv("GOOGLE_GENAI_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY")
	}
	if apiKey == "" {
		return nil, fmt.Errorf("GOOGLE_GENAI_API_KEY or GEMINI_API_KEY not set")
	}

	ctx := context.Background()
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create Gemini client: %w", err)
	}

	return &GeminiModel{
		client: client,
		model:  model,
	}, nil
}

func (g *GeminiModel) MaxTokens() int {
	return 1_048_576
}

// SetMiddleware sets the middleware for the Gemini model
func (g *GeminiModel) SetMiddleware(middleware []Middleware) {
	g.middleware = middleware
}

// SetToolExecutor sets the tool executor for the Gemini model
func (g *GeminiModel) SetToolExecutor(executor ToolExecutor) {
	g.toolExecutor = executor
}

func (g *GeminiModel) Call(
	ctx context.Context,
	inputs []Record,
) ([]Record, int, error) {
	return g.CallWithOpts(ctx, inputs, CallModelOpts{})
}

func (g *GeminiModel) CallWithOpts(
	ctx context.Context,
	inputs []Record,
	opts CallModelOpts,
) ([]Record, int, error) {
	var availableTools []ToolDefinition
	if g.toolExecutor != nil && !opts.DisableTools {
		availableTools = g.toolExecutor.GetRegisteredTools()
	}

	// Convert Records to Gemini Content
	var contents []*genai.Content
	var systemInstruction string

	for _, rec := range inputs {
		switch rec.Source {
		case SystemPrompt:
			// Gemini supports system instructions separately
			if systemInstruction != "" {
				systemInstruction += "\n\n" + rec.Content
			} else {
				systemInstruction = rec.Content
			}
		case Prompt:
			contents = append(contents, &genai.Content{
				Parts: []*genai.Part{genai.NewPartFromText(rec.Content)},
				Role:  "user",
			})
		case ModelResp:
			contents = append(contents, &genai.Content{
				Parts: []*genai.Part{genai.NewPartFromText(rec.Content)},
				Role:  "model",
			})
		case ToolCall:
			// Parse the tool call format: "functionName({...})"
			// For now, append as text to user message (will be revisited)
			contents = append(contents, &genai.Content{
				Parts: []*genai.Part{genai.NewPartFromText(rec.Content)},
				Role:  "user",
			})
		case ToolOutput:
			contents = append(contents, &genai.Content{
				Parts: []*genai.Part{genai.NewPartFromText(rec.Content)},
				Role:  "user",
			})
		}
	}

	// Build config
	config := &genai.GenerateContentConfig{
		Temperature: genai.Ptr(float32(1.0)),
	}

	if systemInstruction != "" {
		config.SystemInstruction = &genai.Content{
			Parts: []*genai.Part{genai.NewPartFromText(systemInstruction)},
			Role:  "user",
		}
	}

	if len(availableTools) > 0 {
		tools := getGeminiToolParams(availableTools)
		config.Tools = tools
	}

	// Make initial request
	resp, err := g.client.Models.GenerateContent(ctx, g.model, contents, config)
	if err != nil {
		return nil, 0, fmt.Errorf("Gemini API: %w", err)
	}

	var events []Record
	totalTokens := 0
	if resp.UsageMetadata != nil {
		totalTokens = int(resp.UsageMetadata.TotalTokenCount)
	}

	// Tool calling loop
	for len(resp.FunctionCalls()) > 0 {
		var modelParts []*genai.Part

		// Collect all parts from the response (text + function calls)
		for _, part := range resp.Candidates[0].Content.Parts {
			if part.Text != "" {
				modelParts = append(modelParts, genai.NewPartFromText(part.Text))
			}
			if part.FunctionCall != nil {
				modelParts = append(modelParts, genai.NewPartFromFunctionCall(
					part.FunctionCall.Name,
					part.FunctionCall.Args,
				))
			}
		}

		// Add model's response to conversation
		contents = append(contents, &genai.Content{
			Parts: modelParts,
			Role:  "model",
		})

		// Execute tools and collect responses
		var toolResponseParts []*genai.Part

		for _, funcCall := range resp.FunctionCalls() {
			// Marshal args to JSON for middleware and execution
			argsJSON, err := json.Marshal(funcCall.Args)
			if err != nil {
				argsJSON = []byte("{}")
			}

			for _, m := range g.middleware {
				m.OnToolCall(ctx, funcCall.Name, string(argsJSON))
			}

			out, err := g.toolExecutor.ExecuteTool(ctx, funcCall.Name, json.RawMessage(argsJSON))
			if err != nil {
				out = fmt.Sprintf("error: %s", err)
			}

			for _, m := range g.middleware {
				m.OnToolResult(ctx, funcCall.Name, out, err)
			}

			// Record the tool call and output
			call := fmt.Sprintf("%s(%s)", funcCall.Name, string(argsJSON))
			events = append(events, Record{
				Source:    ToolCall,
				Content:   call,
				Live:      true,
				EstTokens: tokenCount(call),
			})
			events = append(events, Record{
				Source:    ToolOutput,
				Content:   out,
				Live:      true,
				EstTokens: tokenCount(out),
			})

			// Parse the output as the function response
			var response map[string]interface{}
			if err := json.Unmarshal([]byte(out), &response); err != nil {
				// If not JSON, wrap in a simple response
				response = map[string]interface{}{"result": out}
			}

			toolResponseParts = append(toolResponseParts, genai.NewPartFromFunctionResponse(
				funcCall.Name,
				response,
			))
		}

		// Add tool responses to conversation
		contents = append(contents, &genai.Content{
			Parts: toolResponseParts,
			Role:  "user",
		})

		// Continue the conversation
		resp, err = g.client.Models.GenerateContent(ctx, g.model, contents, config)
		if err != nil {
			return nil, 0, fmt.Errorf("Gemini API (tool continuation): %w", err)
		}

		if resp.UsageMetadata != nil {
			totalTokens += int(resp.UsageMetadata.TotalTokenCount)
		}
	}

	// Extract final text response
	responseText := resp.Text()

	events = append(events, Record{
		Source:    ModelResp,
		Content:   responseText,
		Live:      true,
		EstTokens: tokenCount(responseText),
	})

	return events, totalTokens, nil
}

// CallWithThreading implements ServerSideThreadingCapable interface
func (g *GeminiModel) CallWithThreading(
	ctx context.Context,
	useServerSideThreading bool,
	lastResponseID *string,
	inputs []Record,
) ([]Record, *string, int, error) {
	return g.CallWithThreadingAndOpts(ctx, useServerSideThreading, lastResponseID, inputs, CallModelOpts{})
}

// CallWithThreadingAndOpts implements CallOptsCapable interface
func (g *GeminiModel) CallWithThreadingAndOpts(
	ctx context.Context,
	useServerSideThreading bool,
	lastResponseID *string,
	inputs []Record,
	opts CallModelOpts,
) ([]Record, *string, int, error) {
	// Gemini doesn't support server-side threading, so we always use client-side
	events, tokensUsed, err := g.CallWithOpts(ctx, inputs, opts)
	return events, nil, tokensUsed, err
}

// getGeminiToolParams converts ToolDefinitions to Gemini tool parameters
func getGeminiToolParams(availableTools []ToolDefinition) []*genai.Tool {
	var functionDeclarations []*genai.FunctionDeclaration

	for _, tool := range availableTools {
		if geminiFunc, ok := tool.Definition.(*genai.FunctionDeclaration); ok {
			functionDeclarations = append(functionDeclarations, geminiFunc)
			continue
		}

		if builder, ok := tool.Definition.(*ToolBuilder); ok {
			geminiFunc := builder.ToGemini()
			functionDeclarations = append(functionDeclarations, geminiFunc)
			continue
		}

		panic(fmt.Sprintf("can't convert tool definition for %s to Gemini format (type: %T)", tool.Name, tool.Definition))
	}

	return []*genai.Tool{{FunctionDeclarations: functionDeclarations}}
}
