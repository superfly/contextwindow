//go:build integration

package contextwindow

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestClaudeModel_CallStreaming tests basic streaming functionality
func TestClaudeModel_CallStreaming(t *testing.T) {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("set ANTHROPIC_API_KEY to run integration test")
	}

	m, err := NewClaudeModel(ModelClaudeSonnet45)
	if err != nil {
		t.Fatalf("NewClaudeModel: %v", err)
	}

	inputs := []Record{
		{Source: Prompt, Content: "Say 'hello' and nothing else."},
	}

	var receivedChunks []StreamChunk
	var accumulatedText strings.Builder

	callback := func(chunk StreamChunk) error {
		receivedChunks = append(receivedChunks, chunk)
		if chunk.Delta != "" {
			accumulatedText.WriteString(chunk.Delta)
		}
		return nil
	}

	events, tokens, err := m.CallStreaming(context.Background(), inputs, callback)
	if err != nil {
		if isQuotaExhaustedError(err) {
			t.Skipf("Skipping test due to quota exhaustion: %v", err)
		}
		assert.NoError(t, err)
	}
	assert.Greater(t, len(events), 0)
	assert.Greater(t, tokens, 0)
	assert.Greater(t, len(receivedChunks), 0, "should receive at least one chunk")

	// Check that we got a done chunk
	var hasDoneChunk bool
	for _, chunk := range receivedChunks {
		if chunk.Done {
			hasDoneChunk = true
			break
		}
	}
	assert.True(t, hasDoneChunk, "should receive a done chunk")

	// Check that accumulated text matches final event
	finalContent := accumulatedText.String()
	assert.NotEmpty(t, finalContent)
	assert.Equal(t, finalContent, events[len(events)-1].Content)
	assert.Contains(t, strings.ToLower(finalContent), "hello")
}

// TestClaudeModel_CallStreamingWithOpts tests streaming with options
func TestClaudeModel_CallStreamingWithOpts(t *testing.T) {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("set ANTHROPIC_API_KEY to run integration test")
	}

	m, err := NewClaudeModel(ModelClaudeSonnet45)
	if err != nil {
		t.Fatalf("NewClaudeModel: %v", err)
	}

	inputs := []Record{
		{Source: Prompt, Content: "Count to 3."},
	}

	var chunkCount int
	callback := func(chunk StreamChunk) error {
		if !chunk.Done {
			chunkCount++
		}
		return nil
	}

	events, _, err := m.CallStreamingWithOpts(context.Background(), inputs, CallModelOpts{}, callback)
	if err != nil {
		if isQuotaExhaustedError(err) {
			t.Skipf("Skipping test due to quota exhaustion: %v", err)
		}
		assert.NoError(t, err)
	}
	assert.Greater(t, len(events), 0)
	assert.Greater(t, chunkCount, 0, "should receive content chunks")
}

// TestClaudeModel_CallStreaming_DeltaAccumulation tests that deltas are accumulated correctly
func TestClaudeModel_CallStreaming_DeltaAccumulation(t *testing.T) {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("set ANTHROPIC_API_KEY to run integration test")
	}

	m, err := NewClaudeModel(ModelClaudeSonnet45)
	if err != nil {
		t.Fatalf("NewClaudeModel: %v", err)
	}

	inputs := []Record{
		{Source: Prompt, Content: "Write the numbers 1, 2, 3 in sequence."},
	}

	var deltas []string
	callback := func(chunk StreamChunk) error {
		if chunk.Delta != "" {
			deltas = append(deltas, chunk.Delta)
		}
		return nil
	}

	events, _, err := m.CallStreaming(context.Background(), inputs, callback)
	assert.NoError(t, err)
	assert.Greater(t, len(deltas), 0, "should receive deltas")

	// Verify that all deltas combined equal the final content
	finalContent := events[len(events)-1].Content
	accumulated := strings.Join(deltas, "")
	assert.Equal(t, finalContent, accumulated, "accumulated deltas should match final content")
}

// TestClaudeModel_CallStreaming_ToolCalls tests tool calls in streaming mode
func TestClaudeModel_CallStreaming_ToolCalls(t *testing.T) {
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

	// Create a tool using ToolBuilder
	weatherTool := NewTool("get_weather", "Get the weather for a location").
		AddStringParameter("location", "The location to get weather for", true)

	// Convert to Claude format
	claudeTool := weatherTool.ToClaude()

	err = cw.RegisterTool("get_weather", claudeTool, ToolRunnerFunc(func(ctx context.Context, args json.RawMessage) (string, error) {
		return "sunny, 72°F", nil
	}))
	if err != nil {
		t.Fatalf("RegisterTool: %v", err)
	}

	var receivedChunks []StreamChunk
	callback := func(chunk StreamChunk) error {
		receivedChunks = append(receivedChunks, chunk)
		return nil
	}

	err = cw.AddPrompt("What's the weather in San Francisco? Use the get_weather tool.")
	assert.NoError(t, err)

	response, err := cw.CallModelStreaming(context.Background(), callback)
	if err != nil {
		if isQuotaExhaustedError(err) {
			t.Skipf("Skipping test due to quota exhaustion: %v", err)
		}
		assert.NoError(t, err)
	}
	assert.NotEmpty(t, response)
	assert.Greater(t, len(receivedChunks), 0, "should receive chunks")

	// Verify tool was called (check records)
	recs, err := cw.Reader().LiveRecords()
	assert.NoError(t, err)

	var hasToolCall bool
	var hasToolOutput bool
	for _, rec := range recs {
		if rec.Source == ToolCall {
			hasToolCall = true
		}
		if rec.Source == ToolOutput {
			hasToolOutput = true
		}
	}
	assert.True(t, hasToolCall, "should have tool call record")
	assert.True(t, hasToolOutput, "should have tool output record")
}

// TestClaudeModel_CallStreaming_ErrorHandling tests error handling mid-stream
func TestClaudeModel_CallStreaming_ErrorHandling(t *testing.T) {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("set ANTHROPIC_API_KEY to run integration test")
	}

	m, err := NewClaudeModel(ModelClaudeSonnet45)
	if err != nil {
		t.Fatalf("NewClaudeModel: %v", err)
	}

	inputs := []Record{
		{Source: Prompt, Content: "Say hello"},
	}

	// Callback that returns an error after first chunk
	var chunkCount int
	callback := func(chunk StreamChunk) error {
		chunkCount++
		if chunkCount == 1 {
			return assert.AnError
		}
		return nil
	}

	_, _, err = m.CallStreaming(context.Background(), inputs, callback)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "callback error")
}

// TestClaudeModel_CallStreaming_ContextCancellation tests context cancellation
func TestClaudeModel_CallStreaming_ContextCancellation(t *testing.T) {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("set ANTHROPIC_API_KEY to run integration test")
	}

	m, err := NewClaudeModel(ModelClaudeSonnet45)
	if err != nil {
		t.Fatalf("NewClaudeModel: %v", err)
	}

	inputs := []Record{
		{Source: Prompt, Content: "Count from 1 to 100."},
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	callback := func(chunk StreamChunk) error {
		return nil
	}

	_, _, err = m.CallStreaming(ctx, inputs, callback)
	// The error might be context cancellation or stream error
	assert.Error(t, err)
}

// TestClaudeModel_CallStreaming_DisableTools tests that tools can be disabled
func TestClaudeModel_CallStreaming_DisableTools(t *testing.T) {
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

	// Register a tool
	testTool := NewTool("test_tool", "A test tool")
	claudeTool := testTool.ToClaude()
	err = cw.RegisterTool("test_tool", claudeTool, ToolRunnerFunc(func(ctx context.Context, args json.RawMessage) (string, error) {
		return "should not be called", nil
	}))
	if err != nil {
		t.Fatalf("RegisterTool: %v", err)
	}

	var receivedChunks []StreamChunk
	callback := func(chunk StreamChunk) error {
		receivedChunks = append(receivedChunks, chunk)
		return nil
	}

	err = cw.AddPrompt("Hello")
	assert.NoError(t, err)

	// Test with tools disabled
	response, err := cw.CallModelStreamingWithOpts(context.Background(), CallModelOpts{DisableTools: true}, callback)
	if err != nil {
		if isQuotaExhaustedError(err) {
			t.Skipf("Skipping test due to quota exhaustion: %v", err)
		}
		assert.NoError(t, err)
	}
	assert.NotEmpty(t, response)

	// Verify no tool calls were made
	recs, err := cw.Reader().LiveRecords()
	assert.NoError(t, err)
	for _, rec := range recs {
		assert.NotEqual(t, ToolCall, rec.Source, "should not have tool calls when disabled")
	}
}

// TestClaudeModel_CallStreaming_MultiBlock tests multi-block responses
func TestClaudeModel_CallStreaming_MultiBlock(t *testing.T) {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("set ANTHROPIC_API_KEY to run integration test")
	}

	m, err := NewClaudeModel(ModelClaudeSonnet45)
	if err != nil {
		t.Fatalf("NewClaudeModel: %v", err)
	}

	inputs := []Record{
		{Source: Prompt, Content: "Write a short paragraph about AI, then write another paragraph about machine learning."},
	}

	var receivedChunks []StreamChunk
	var accumulatedText strings.Builder
	callback := func(chunk StreamChunk) error {
		receivedChunks = append(receivedChunks, chunk)
		if chunk.Delta != "" {
			accumulatedText.WriteString(chunk.Delta)
		}
		return nil
	}

	events, tokens, err := m.CallStreaming(context.Background(), inputs, callback)
	if err != nil {
		if isQuotaExhaustedError(err) {
			t.Skipf("Skipping test due to quota exhaustion: %v", err)
		}
		assert.NoError(t, err)
	}
	assert.Greater(t, len(events), 0)
	assert.Greater(t, tokens, 0)
	assert.Greater(t, len(receivedChunks), 0)

	// Verify accumulated text matches final content
	finalContent := events[len(events)-1].Content
	accumulated := accumulatedText.String()
	assert.Equal(t, finalContent, accumulated)
	assert.NotEmpty(t, finalContent)
}
