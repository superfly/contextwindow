package contextwindow

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

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
		//lint:ignore ST1005 - Proper noun error message
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
			//lint:ignore ST1005 - Proper noun error message
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

// CallStreaming implements StreamingCapable interface
func (g *GeminiModel) CallStreaming(
	ctx context.Context,
	inputs []Record,
	callback StreamCallback,
) ([]Record, int, error) {
	return g.CallStreamingWithOpts(ctx, inputs, CallModelOpts{}, callback)
}

// CallStreamingWithOpts implements StreamingOptsCapable interface
func (g *GeminiModel) CallStreamingWithOpts(
	ctx context.Context,
	inputs []Record,
	opts CallModelOpts,
	callback StreamCallback,
) ([]Record, int, error) {
	var availableTools []ToolDefinition
	if g.toolExecutor != nil && !opts.DisableTools {
		availableTools = g.toolExecutor.GetRegisteredTools()
	}

	// Convert Records to Gemini Content (reuse existing logic)
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

	// Build config (reuse existing logic)
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

	var events []Record
	var totalTokens int

	// Handle tool calls in a loop (similar to CallWithOpts)
	for {
		// Call GenerateContentStream
		stream := g.client.Models.GenerateContentStream(ctx, g.model, contents, config)

		// Accumulate deltas in buffer
		// Pre-allocate with 4KB capacity to reduce reallocations for typical responses
		// Profile with: go test -bench=. -benchmem -cpuprofile=cpu.prof -memprofile=mem.prof
		contentBuilder := strings.Builder{}
		contentBuilder.Grow(4096) // Pre-allocate 4KB buffer
		var functionCalls []*genai.FunctionCall
		var usageMetadata *genai.GenerateContentResponseUsageMetadata

		// Iterate over response chunks using Go's iterator pattern
		for chunk, err := range stream {
			// Check for context cancellation before processing chunk
			select {
			case <-ctx.Done():
				// Context was cancelled - handle partial response
				partialContent := contentBuilder.String()
				errChunk := StreamChunk{
					Error: fmt.Errorf("stream cancelled: %w", ctx.Err()),
					Done:  true,
				}
				if callback != nil {
					_ = callback(errChunk) // Best effort to notify callback
				}
				// Return partial response if we have any content
				if partialContent != "" {
					events = append(events, Record{
						Source:    ModelResp,
						Content:   partialContent,
						Live:      true,
						EstTokens: tokenCount(partialContent),
					})
					return events, tokenCount(partialContent), fmt.Errorf(" Gemini streaming cancelled (partial response saved): %w", ctx.Err())
				}
				return nil, 0, fmt.Errorf(" Gemini streaming cancelled: %w", ctx.Err())
			default:
				// Continue processing
			}

			// Check for errors in the stream
			if err != nil {
				streamErr := err
				partialContent := contentBuilder.String()

				// Wrap error with provider context and error type
				var wrappedErr error
				if isNetworkError(streamErr) {
					wrappedErr = fmt.Errorf(" Gemini streaming network error (partial response saved): %w", streamErr)
				} else {
					wrappedErr = fmt.Errorf(" Gemini streaming error (partial response saved): %w", streamErr)
				}

				errChunk := StreamChunk{
					Error: wrappedErr,
					Done:  true,
				}
				if callback != nil {
					if err := callback(errChunk); err != nil {
						// If callback also errors, return both errors
						return nil, 0, fmt.Errorf("callback error during stream error: %w (original: %w)", err, wrappedErr)
					}
				}

				// Return partial response if we have any content
				if partialContent != "" {
					events = append(events, Record{
						Source:    ModelResp,
						Content:   partialContent,
						Live:      true,
						EstTokens: tokenCount(partialContent),
					})
					return events, tokenCount(partialContent), wrappedErr
				}
				return nil, 0, wrappedErr
			}

			// Handle usage metadata
			if chunk.UsageMetadata != nil {
				usageMetadata = chunk.UsageMetadata
			}

			// Process candidates
			if len(chunk.Candidates) > 0 {
				candidate := chunk.Candidates[0]

				// Handle content parts
				if candidate.Content != nil {
					for _, part := range candidate.Content.Parts {
						// Handle text deltas
						if part.Text != "" {
							delta := part.Text
							contentBuilder.WriteString(delta)

							// Invoke callback for each chunk (minimize overhead by checking nil first)
							if callback != nil {
								// Reuse metadata map to reduce allocations
								metadata := make(map[string]any, 1)
								if candidate.FinishReason != "" {
									metadata["finish_reason"] = candidate.FinishReason
								}
								chunk := StreamChunk{
									Delta:    delta,
									Done:     false,
									Metadata: metadata,
								}
								if err := callback(chunk); err != nil {
									// Callback requested cancellation - save partial response
									partialContent := contentBuilder.String()
									if partialContent != "" {
										events = append(events, Record{
											Source:    ModelResp,
											Content:   partialContent,
											Live:      true,
											EstTokens: tokenCount(partialContent),
										})
										return events, tokenCount(partialContent), fmt.Errorf("callback error (partial response saved): %w", err)
									}
									return nil, 0, fmt.Errorf("callback error: %w", err)
								}
							}
						}

						// Handle function call deltas - buffer until complete
						if part.FunctionCall != nil {
							functionCalls = append(functionCalls, part.FunctionCall)
						}
					}
				}

				// Check if we have function calls to process
				if len(functionCalls) > 0 && candidate.FinishReason != "" {
					// Function calls are complete, break to process them
					break
				}
			}
		}

		// Check for context cancellation after stream completes
		select {
		case <-ctx.Done():
			partialContent := contentBuilder.String()
			if partialContent != "" {
				events = append(events, Record{
					Source:    ModelResp,
					Content:   partialContent,
					Live:      true,
					EstTokens: tokenCount(partialContent),
				})
				return events, tokenCount(partialContent), fmt.Errorf(" Gemini streaming cancelled after stream (partial response saved): %w", ctx.Err())
			}
			return nil, 0, fmt.Errorf(" Gemini streaming cancelled: %w", ctx.Err())
		default:
			// Continue
		}

		// Update token count
		if usageMetadata != nil {
			totalTokens += int(usageMetadata.TotalTokenCount)
		}

		// Handle function calls if present
		if len(functionCalls) > 0 {
			var modelParts []*genai.Part

			// Add any accumulated text
			if contentBuilder.Len() > 0 {
				modelParts = append(modelParts, genai.NewPartFromText(contentBuilder.String()))
			}

			// Add function calls
			for _, funcCall := range functionCalls {
				modelParts = append(modelParts, genai.NewPartFromFunctionCall(
					funcCall.Name,
					funcCall.Args,
				))
			}

			// Add model's response to conversation
			contents = append(contents, &genai.Content{
				Parts: modelParts,
				Role:  "model",
			})

			// Execute tools and collect responses
			var toolResponseParts []*genai.Part

			for _, funcCall := range functionCalls {
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

				// Stream tool result if callback provided
				if callback != nil {
					// Check for context cancellation before streaming tool result
					select {
					case <-ctx.Done():
						partialContent := contentBuilder.String()
						if partialContent != "" {
							events = append(events, Record{
								Source:    ModelResp,
								Content:   partialContent,
								Live:      true,
								EstTokens: tokenCount(partialContent),
							})
							return events, tokenCount(partialContent), fmt.Errorf(" Gemini streaming cancelled during tool execution (partial response saved): %w", ctx.Err())
						}
						return nil, 0, fmt.Errorf("google Gemini streaming cancelled: %w", ctx.Err())
					default:
						// Continue
					}

					// Pre-allocate metadata map to reduce allocations
					metadata := make(map[string]any, 1)
					metadata["tool_call"] = funcCall.Name
					toolResultChunk := StreamChunk{
						Delta:    fmt.Sprintf("\n[Tool: %s returned: %s]\n", funcCall.Name, out),
						Done:     false,
						Metadata: metadata,
					}
					if err := callback(toolResultChunk); err != nil {
						// Callback requested cancellation - save partial response
						partialContent := contentBuilder.String()
						if partialContent != "" {
							events = append(events, Record{
								Source:    ModelResp,
								Content:   partialContent,
								Live:      true,
								EstTokens: tokenCount(partialContent),
							})
							return events, tokenCount(partialContent), fmt.Errorf("callback error during tool execution (partial response saved): %w", err)
						}
						return nil, 0, fmt.Errorf("callback error: %w", err)
					}
				}

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

			// Continue loop to get next response after tool calls
			functionCalls = nil
			contentBuilder.Reset()
			contentBuilder.Grow(4096) // Re-allocate buffer for next iteration
			continue
		}

		// No function calls, we have the final response
		content := contentBuilder.String()

		// Send done chunk
		if callback != nil {
			// Pre-allocate metadata map to reduce allocations
			metadata := make(map[string]any, 1)
			metadata["total_tokens"] = totalTokens
			doneChunk := StreamChunk{
				Delta:    "",
				Done:     true,
				Metadata: metadata,
			}
			if err := callback(doneChunk); err != nil {
				return nil, 0, fmt.Errorf("callback error: %w", err)
			}
		}

		// Record final response
		events = append(events, Record{
			Source:    ModelResp,
			Content:   content,
			Live:      true,
			EstTokens: tokenCount(content),
		})

		return events, totalTokens, nil
	}
}
