//go:build integration

package contextwindow

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/openai/openai-go/v2/shared"
	"github.com/stretchr/testify/assert"
)

// TestStreaming_EndToEnd_OpenAI tests end-to-end streaming with OpenAI
func TestStreaming_EndToEnd_OpenAI(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("set OPENAI_API_KEY to run integration test")
	}

	m, err := NewOpenAIModel(shared.ChatModelGPT4o)
	if err != nil {
		t.Fatalf("NewOpenAIModel: %v", err)
	}

	db, err := NewContextDB(":memory:")
	if err != nil {
		t.Fatalf("NewContextDB: %v", err)
	}
	defer db.Close()

	cw, err := NewContextWindow(db, m, "test")
	if err != nil {
		t.Fatalf("NewContextWindow: %v", err)
	}

	err = cw.AddPrompt("Write a short story about a robot learning to paint. Make it exactly 3 sentences.")
	assert.NoError(t, err)

	var receivedChunks []StreamChunk
	var accumulatedText strings.Builder
	var firstChunkTime time.Time
	var streamStartTime time.Time

	callback := func(chunk StreamChunk) error {
		if len(receivedChunks) == 0 {
			streamStartTime = time.Now()
		}
		receivedChunks = append(receivedChunks, chunk)
		if chunk.Delta != "" {
			if firstChunkTime.IsZero() {
				firstChunkTime = time.Now()
			}
			accumulatedText.WriteString(chunk.Delta)
		}
		return nil
	}

	startTime := time.Now()
	response, err := cw.CallModelStreaming(context.Background(), callback)
	totalTime := time.Since(startTime)

	assert.NoError(t, err)
	assert.NotEmpty(t, response)
	assert.Greater(t, len(receivedChunks), 0, "should receive chunks")

	// Verify we got a done chunk
	var hasDoneChunk bool
	for _, chunk := range receivedChunks {
		if chunk.Done {
			hasDoneChunk = true
			break
		}
	}
	assert.True(t, hasDoneChunk, "should receive a done chunk")

	// Verify accumulated text matches final response
	assert.Equal(t, response, accumulatedText.String(), "accumulated text should match final response")

	// Measure latency improvements
	if !firstChunkTime.IsZero() {
		timeToFirstToken := firstChunkTime.Sub(streamStartTime)
		t.Logf("Time to first token: %v", timeToFirstToken)
		t.Logf("Total streaming time: %v", totalTime)
		assert.Less(t, timeToFirstToken, 2*time.Second, "first token should arrive quickly")
	}

	// Verify persistence
	recs, err := cw.Reader().LiveRecords()
	assert.NoError(t, err)
	var hasResponse bool
	for _, rec := range recs {
		if rec.Source == ModelResp && strings.Contains(rec.Content, "robot") {
			hasResponse = true
			break
		}
	}
	assert.True(t, hasResponse, "response should be persisted")
}

// TestStreaming_EndToEnd_Claude tests end-to-end streaming with Claude
func TestStreaming_EndToEnd_Claude(t *testing.T) {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("set ANTHROPIC_API_KEY to run integration test")
	}

	m, err := NewClaudeModel(ModelClaudeSonnet45)
	if err != nil {
		t.Fatalf("NewClaudeModel: %v", err)
	}

	db, err := NewContextDB(":memory:")
	if err != nil {
		t.Fatalf("NewContextDB: %v", err)
	}
	defer db.Close()

	cw, err := NewContextWindow(db, m, "test")
	if err != nil {
		t.Fatalf("NewContextWindow: %v", err)
	}

	err = cw.AddPrompt("Describe the color blue in exactly 2 sentences.")
	assert.NoError(t, err)

	var receivedChunks []StreamChunk
	var accumulatedText strings.Builder
	var firstChunkTime time.Time

	callback := func(chunk StreamChunk) error {
		receivedChunks = append(receivedChunks, chunk)
		if chunk.Delta != "" {
			if firstChunkTime.IsZero() {
				firstChunkTime = time.Now()
			}
			accumulatedText.WriteString(chunk.Delta)
		}
		return nil
	}

	startTime := time.Now()
	response, err := cw.CallModelStreaming(context.Background(), callback)
	totalTime := time.Since(startTime)

	assert.NoError(t, err)
	assert.NotEmpty(t, response)
	assert.Greater(t, len(receivedChunks), 0)
	assert.Equal(t, response, accumulatedText.String())

	if !firstChunkTime.IsZero() {
		timeToFirstToken := firstChunkTime.Sub(startTime)
		t.Logf("Time to first token: %v", timeToFirstToken)
		t.Logf("Total streaming time: %v", totalTime)
		assert.Less(t, timeToFirstToken, 2*time.Second)
	}
}

// TestStreaming_EndToEnd_Gemini tests end-to-end streaming with Gemini
func TestStreaming_EndToEnd_Gemini(t *testing.T) {
	if os.Getenv("GOOGLE_GENAI_API_KEY") == "" && os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("set GOOGLE_GENAI_API_KEY or GEMINI_API_KEY to run integration test")
	}

	m, err := NewGeminiModel(ModelGemini20Flash)
	if err != nil {
		t.Fatalf("NewGeminiModel: %v", err)
	}

	db, err := NewContextDB(":memory:")
	if err != nil {
		t.Fatalf("NewContextDB: %v", err)
	}
	defer db.Close()

	cw, err := NewContextWindow(db, m, "test")
	if err != nil {
		t.Fatalf("NewContextWindow: %v", err)
	}

	err = cw.AddPrompt("Explain quantum computing in exactly 2 sentences.")
	assert.NoError(t, err)

	var receivedChunks []StreamChunk
	var accumulatedText strings.Builder
	var firstChunkTime time.Time

	callback := func(chunk StreamChunk) error {
		receivedChunks = append(receivedChunks, chunk)
		if chunk.Delta != "" {
			if firstChunkTime.IsZero() {
				firstChunkTime = time.Now()
			}
			accumulatedText.WriteString(chunk.Delta)
		}
		return nil
	}

	startTime := time.Now()
	response, err := cw.CallModelStreaming(context.Background(), callback)
	totalTime := time.Since(startTime)

	assert.NoError(t, err)
	assert.NotEmpty(t, response)
	assert.Greater(t, len(receivedChunks), 0)
	assert.Equal(t, response, accumulatedText.String())

	if !firstChunkTime.IsZero() {
		timeToFirstToken := firstChunkTime.Sub(startTime)
		t.Logf("Time to first token: %v", timeToFirstToken)
		t.Logf("Total streaming time: %v", totalTime)
		assert.Less(t, timeToFirstToken, 2*time.Second)
	}
}

// TestStreaming_MultiTurnConversation tests multi-turn streaming conversations
func TestStreaming_MultiTurnConversation(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("set OPENAI_API_KEY to run integration test")
	}

	m, err := NewOpenAIModel(shared.ChatModelGPT4o)
	if err != nil {
		t.Fatalf("NewOpenAIModel: %v", err)
	}

	db, err := NewContextDB(":memory:")
	if err != nil {
		t.Fatalf("NewContextDB: %v", err)
	}
	defer db.Close()

	cw, err := NewContextWindow(db, m, "test")
	if err != nil {
		t.Fatalf("NewContextWindow: %v", err)
	}

	// First turn
	err = cw.AddPrompt("My name is Alice. Remember this.")
	assert.NoError(t, err)

	var turn1Chunks []StreamChunk
	callback1 := func(chunk StreamChunk) error {
		turn1Chunks = append(turn1Chunks, chunk)
		return nil
	}

	response1, err := cw.CallModelStreaming(context.Background(), callback1)
	assert.NoError(t, err)
	assert.NotEmpty(t, response1)
	assert.Greater(t, len(turn1Chunks), 0)

	// Second turn
	err = cw.AddPrompt("What is my name?")
	assert.NoError(t, err)

	var turn2Chunks []StreamChunk
	var accumulatedText strings.Builder
	callback2 := func(chunk StreamChunk) error {
		turn2Chunks = append(turn2Chunks, chunk)
		if chunk.Delta != "" {
			accumulatedText.WriteString(chunk.Delta)
		}
		return nil
	}

	response2, err := cw.CallModelStreaming(context.Background(), callback2)
	assert.NoError(t, err)
	assert.NotEmpty(t, response2)
	assert.Greater(t, len(turn2Chunks), 0)
	assert.Equal(t, response2, accumulatedText.String())

	// Verify context was maintained
	assert.Contains(t, strings.ToLower(response2), "alice", "should remember the name from previous turn")

	// Third turn
	err = cw.AddPrompt("Tell me a joke.")
	assert.NoError(t, err)

	var turn3Chunks []StreamChunk
	callback3 := func(chunk StreamChunk) error {
		turn3Chunks = append(turn3Chunks, chunk)
		return nil
	}

	response3, err := cw.CallModelStreaming(context.Background(), callback3)
	assert.NoError(t, err)
	assert.NotEmpty(t, response3)
	assert.Greater(t, len(turn3Chunks), 0)

	// Verify all turns are persisted
	recs, err := cw.Reader().LiveRecords()
	assert.NoError(t, err)
	var promptCount, responseCount int
	for _, rec := range recs {
		if rec.Source == Prompt {
			promptCount++
		}
		if rec.Source == ModelResp {
			responseCount++
		}
	}
	assert.GreaterOrEqual(t, promptCount, 3, "should have at least 3 prompts")
	assert.GreaterOrEqual(t, responseCount, 3, "should have at least 3 responses")
}

// TestStreaming_WithTools tests streaming with tools enabled
func TestStreaming_WithTools(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("set OPENAI_API_KEY to run integration test")
	}

	m, err := NewOpenAIModel(shared.ChatModelGPT4o)
	if err != nil {
		t.Fatalf("NewOpenAIModel: %v", err)
	}

	db, err := NewContextDB(":memory:")
	if err != nil {
		t.Fatalf("NewContextDB: %v", err)
	}
	defer db.Close()

	cw, err := NewContextWindow(db, m, "test")
	if err != nil {
		t.Fatalf("NewContextWindow: %v", err)
	}

	// Create a calculator tool
	calcTool := NewTool("calculate", "Performs basic arithmetic operations").
		AddStringParameter("operation", "The operation: add, subtract, multiply, or divide", true).
		AddNumberParameter("a", "First number", true).
		AddNumberParameter("b", "Second number", true)

	err = cw.AddTool(calcTool, ToolRunnerFunc(func(ctx context.Context, args json.RawMessage) (string, error) {
		var params struct {
			Operation string  `json:"operation"`
			A         float64 `json:"a"`
			B         float64 `json:"b"`
		}
		if err := json.Unmarshal(args, &params); err != nil {
			return "", err
		}

		var result float64
		switch params.Operation {
		case "add":
			result = params.A + params.B
		case "subtract":
			result = params.A - params.B
		case "multiply":
			result = params.A * params.B
		case "divide":
			if params.B == 0 {
				return "", assert.AnError
			}
			result = params.A / params.B
		default:
			return "", assert.AnError
		}

		data, err := json.Marshal(map[string]float64{"result": result})
		if err != nil {
			return "", err
		}
		return string(data), nil
	}))
	assert.NoError(t, err)

	err = cw.AddPrompt("Use the calculate tool to multiply 7 and 8, then tell me the result.")
	assert.NoError(t, err)

	var receivedChunks []StreamChunk
	var accumulatedText strings.Builder
	callback := func(chunk StreamChunk) error {
		receivedChunks = append(receivedChunks, chunk)
		if chunk.Delta != "" {
			accumulatedText.WriteString(chunk.Delta)
		}
		return nil
	}

	response, err := cw.CallModelStreaming(context.Background(), callback)
	assert.NoError(t, err)
	assert.NotEmpty(t, response)
	assert.Greater(t, len(receivedChunks), 0)

	// Verify tool was called
	recs, err := cw.Reader().LiveRecords()
	assert.NoError(t, err)

	var hasToolCall, hasToolOutput, hasFinalResponse bool
	for _, rec := range recs {
		if rec.Source == ToolCall {
			hasToolCall = true
			assert.Contains(t, strings.ToLower(rec.Content), "calculate", "tool call should mention calculate")
		}
		if rec.Source == ToolOutput {
			hasToolOutput = true
		}
		if rec.Source == ModelResp && strings.Contains(rec.Content, "56") {
			hasFinalResponse = true
		}
	}

	assert.True(t, hasToolCall, "should have tool call record")
	assert.True(t, hasToolOutput, "should have tool output record")
	assert.True(t, hasFinalResponse, "should have final response with result")

	// Verify accumulated text matches final response
	assert.Equal(t, response, accumulatedText.String())
	assert.Contains(t, strings.ToLower(response), "56", "response should contain the calculation result")
}

// TestStreaming_WithSummarization tests streaming with summarization
func TestStreaming_WithSummarization(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("set OPENAI_API_KEY to run integration test")
	}

	// Use a cheaper model for summarization
	summarizerModel, err := NewOpenAIModel(shared.ChatModelGPT4o)
	if err != nil {
		t.Fatalf("NewOpenAIModel for summarizer: %v", err)
	}

	mainModel, err := NewOpenAIModel(shared.ChatModelGPT4o)
	if err != nil {
		t.Fatalf("NewOpenAIModel for main: %v", err)
	}

	db, err := NewContextDB(":memory:")
	if err != nil {
		t.Fatalf("NewContextDB: %v", err)
	}
	defer db.Close()

	cw, err := NewContextWindow(db, mainModel, "test")
	if err != nil {
		t.Fatalf("NewContextWindow: %v", err)
	}

	cw.SetSummarizer(summarizerModel)

	// Add several prompts to build up context
	for i := 0; i < 3; i++ {
		err = cw.AddPrompt("Tell me a fact about space.")
		assert.NoError(t, err)

		var chunks []StreamChunk
		callback := func(chunk StreamChunk) error {
			chunks = append(chunks, chunk)
			return nil
		}

		response, err := cw.CallModelStreaming(context.Background(), callback)
		assert.NoError(t, err)
		assert.NotEmpty(t, response)
		assert.Greater(t, len(chunks), 0)
	}

	// Get initial record count
	recsBefore, err := cw.Reader().LiveRecords()
	assert.NoError(t, err)
	initialCount := len(recsBefore)
	assert.Greater(t, initialCount, 0, "should have records before summarization")

	// Summarize the context
	summaryResult, err := cw.SummarizeLiveContext(context.Background())
	assert.NoError(t, err)
	assert.NotNil(t, summaryResult)
	assert.NotEmpty(t, summaryResult.Summary)

	// Accept the summary
	err = cw.AcceptSummary(summaryResult)
	assert.NoError(t, err)

	// Verify summarization worked
	recsAfter, err := cw.Reader().LiveRecords()
	assert.NoError(t, err)
	afterCount := len(recsAfter)

	// Should have fewer records after summarization
	assert.Less(t, afterCount, initialCount, "should have fewer records after summarization")

	// Add a new prompt and verify context is maintained
	err = cw.AddPrompt("What did we discuss earlier?")
	assert.NoError(t, err)

	var chunks []StreamChunk
	var accumulatedText strings.Builder
	callback := func(chunk StreamChunk) error {
		chunks = append(chunks, chunk)
		if chunk.Delta != "" {
			accumulatedText.WriteString(chunk.Delta)
		}
		return nil
	}

	response, err := cw.CallModelStreaming(context.Background(), callback)
	assert.NoError(t, err)
	assert.NotEmpty(t, response)
	assert.Greater(t, len(chunks), 0)
	assert.Equal(t, response, accumulatedText.String())

	// Response should reference the summarized context
	assert.NotEmpty(t, response, "should have a response referencing previous context")
}

// TestStreaming_AcrossContextSwitches tests streaming across context switches
func TestStreaming_AcrossContextSwitches(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("set OPENAI_API_KEY to run integration test")
	}

	m, err := NewOpenAIModel(shared.ChatModelGPT4o)
	if err != nil {
		t.Fatalf("NewOpenAIModel: %v", err)
	}

	db, err := NewContextDB(":memory:")
	if err != nil {
		t.Fatalf("NewContextDB: %v", err)
	}
	defer db.Close()

	cw, err := NewContextWindow(db, m, "context1")
	if err != nil {
		t.Fatalf("NewContextWindow: %v", err)
	}

	// First context: conversation about animals
	err = cw.AddPrompt("Tell me about cats.")
	assert.NoError(t, err)

	var chunks1 []StreamChunk
	callback1 := func(chunk StreamChunk) error {
		chunks1 = append(chunks1, chunk)
		return nil
	}

	response1, err := cw.CallModelStreaming(context.Background(), callback1)
	assert.NoError(t, err)
	assert.NotEmpty(t, response1)
	assert.Greater(t, len(chunks1), 0)

	// Switch to second context: conversation about programming
	err = cw.SwitchContext("context2")
	assert.NoError(t, err)

	err = cw.AddPrompt("Explain what a function is in programming.")
	assert.NoError(t, err)

	var chunks2 []StreamChunk
	var accumulatedText strings.Builder
	callback2 := func(chunk StreamChunk) error {
		chunks2 = append(chunks2, chunk)
		if chunk.Delta != "" {
			accumulatedText.WriteString(chunk.Delta)
		}
		return nil
	}

	response2, err := cw.CallModelStreaming(context.Background(), callback2)
	assert.NoError(t, err)
	assert.NotEmpty(t, response2)
	assert.Greater(t, len(chunks2), 0)
	assert.Equal(t, response2, accumulatedText.String())

	// Verify response is about programming, not cats
	assert.Contains(t, strings.ToLower(response2), "function", "response should be about programming")
	assert.NotContains(t, strings.ToLower(response2), "cat", "response should not mention cats from previous context")

	// Switch back to first context
	err = cw.SwitchContext("context1")
	assert.NoError(t, err)

	err = cw.AddPrompt("What about dogs?")
	assert.NoError(t, err)

	var chunks3 []StreamChunk
	callback3 := func(chunk StreamChunk) error {
		chunks3 = append(chunks3, chunk)
		return nil
	}

	response3, err := cw.CallModelStreaming(context.Background(), callback3)
	assert.NoError(t, err)
	assert.NotEmpty(t, response3)
	assert.Greater(t, len(chunks3), 0)

	// Verify context was maintained (should reference cats from earlier)
	assert.Contains(t, strings.ToLower(response3), "dog", "response should mention dogs")

	// Verify contexts are separate
	recs1, err := cw.Reader().LiveRecords()
	assert.NoError(t, err)

	err = cw.SwitchContext("context2")
	assert.NoError(t, err)

	recs2, err := cw.Reader().LiveRecords()
	assert.NoError(t, err)

	// Contexts should have different records
	assert.NotEqual(t, len(recs1), len(recs2), "contexts should have different record counts")
}

// TestStreaming_LatencyComparison compares streaming vs non-streaming latency
func TestStreaming_LatencyComparison(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("set OPENAI_API_KEY to run integration test")
	}

	m, err := NewOpenAIModel(shared.ChatModelGPT4o)
	if err != nil {
		t.Fatalf("NewOpenAIModel: %v", err)
	}

	db, err := NewContextDB(":memory:")
	if err != nil {
		t.Fatalf("NewContextDB: %v", err)
	}
	defer db.Close()

	cw, err := NewContextWindow(db, m, "test")
	if err != nil {
		t.Fatalf("NewContextWindow: %v", err)
	}

	err = cw.AddPrompt("Write a 5-sentence story about a robot.")
	assert.NoError(t, err)

	// Test non-streaming
	startNonStream := time.Now()
	responseNonStream, err := cw.CallModel(context.Background())
	nonStreamTime := time.Since(startNonStream)
	assert.NoError(t, err)
	assert.NotEmpty(t, responseNonStream)

	// Reset context for streaming test
	cw2, err := NewContextWindow(db, m, "test2")
	if err != nil {
		t.Fatalf("NewContextWindow: %v", err)
	}

	err = cw2.AddPrompt("Write a 5-sentence story about a robot.")
	assert.NoError(t, err)

	// Test streaming
	var firstChunkTime time.Time
	var streamStartTime time.Time
	var receivedChunks []StreamChunk
	startStream := time.Now()
	callback := func(chunk StreamChunk) error {
		if len(receivedChunks) == 0 {
			streamStartTime = time.Now()
		}
		receivedChunks = append(receivedChunks, chunk)
		if chunk.Delta != "" && firstChunkTime.IsZero() {
			firstChunkTime = time.Now()
		}
		return nil
	}

	responseStream, err := cw2.CallModelStreaming(context.Background(), callback)
	streamTime := time.Since(startStream)
	assert.NoError(t, err)
	assert.NotEmpty(t, responseStream)

	// Log latency metrics
	t.Logf("Non-streaming total time: %v", nonStreamTime)
	t.Logf("Streaming total time: %v", streamTime)
	if !firstChunkTime.IsZero() {
		timeToFirstToken := firstChunkTime.Sub(streamStartTime)
		t.Logf("Time to first token: %v", timeToFirstToken)
		t.Logf("Latency improvement: %v", nonStreamTime-timeToFirstToken)
		assert.Less(t, timeToFirstToken, nonStreamTime, "first token should arrive before non-streaming completes")
	}

	// Both responses should be similar in content
	assert.Greater(t, len(responseNonStream), 50, "non-streaming response should be substantial")
	assert.Greater(t, len(responseStream), 50, "streaming response should be substantial")
}
