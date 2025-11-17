//go:build integration

package contextwindow

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/openai/openai-go/v2/packages/param"
	"github.com/openai/openai-go/v2/shared"
	"github.com/stretchr/testify/assert"
)

func TestOpenAIModel_HelloWorld(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("set OPENAI_API_KEY to run integration test")
	}
	m, err := NewOpenAIModel(shared.ChatModelGPT4o)
	if err != nil {
		t.Fatalf("NewOpenAIModel: %v", err)
	}
	inputs := []Record{
		{Source: Prompt, Content: "Please respond with \"hello world\""},
	}
	reply, _, err := m.Call(context.Background(), inputs)
	if err != nil {
		if isQuotaExhaustedError(err) {
			t.Skipf("Skipping test due to quota exhaustion: %v", err)
		}
		t.Fatalf("Call: %v", err)
	}
	if len(reply) == 0 {
		t.Fatalf("expected a non-empty reply")
	}

	assert.Contains(t, strings.ToLower(reply[len(reply)-1].Content), "hello")
}

func TestOpenAIModel_ToolCall(t *testing.T) {
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

	lsTool := shared.FunctionDefinitionParam{
		Name:        "ls",
		Description: param.NewOpt("list files in a directory"),
		Parameters: map[string]interface{}{
			"type":       "object",
			"properties": map[string]interface{}{},
		},
	}

	err = cw.RegisterTool("ls", lsTool, ToolRunnerFunc(func(ctx context.Context, args json.RawMessage) (string, error) {
		return "go.mod\nspiderman.txt\nbatman.txt", nil
	}))
	if err != nil {
		t.Fatalf("RegisterTool: %v", err)
	}

	inputs := []Record{
		{Source: Prompt, Content: "Please use the `ls` tool to list the files in the current directory."},
	}

	cw.AddPrompt(inputs[0].Content)

	result, err := cw.CallModel(context.Background())
	if err != nil {
		if isQuotaExhaustedError(err) {
			t.Skipf("Skipping test due to quota exhaustion: %v", err)
		}
		t.Fatalf("Call: %v", err)
	}

	assert.Contains(t, result, "go.mod")
	assert.Contains(t, result, "batman")
}

func TestOpenAIModel_SystemPrompt(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	db, err := NewContextDB(":memory:")
	assert.NoError(t, err)
	defer db.Close()

	model, err := NewOpenAIModel(ResponsesModel4o)
	assert.NoError(t, err)

	cw, err := NewContextWindow(db, model, "default")
	assert.NoError(t, err)

	err = cw.SetSystemPrompt("whatever you answer, the answer must include the string MUMON")
	assert.NoError(t, err)

	err = cw.AddPrompt("what's the weather like over there")
	assert.NoError(t, err)

	resp, err := cw.CallModel(context.Background())
	if err != nil {
		if isQuotaExhaustedError(err) {
			t.Skipf("Skipping test due to quota exhaustion: %v", err)
		}
		assert.NoError(t, err)
	}

	assert.Contains(t, resp, "MUMON")
}

// TestOpenAIModel_CallStreaming tests basic streaming functionality
func TestOpenAIModel_CallStreaming(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("set OPENAI_API_KEY to run integration test")
	}

	m, err := NewOpenAIModel(shared.ChatModelGPT4o)
	if err != nil {
		t.Fatalf("NewOpenAIModel: %v", err)
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

// TestOpenAIModel_CallStreamingWithOpts tests streaming with options
func TestOpenAIModel_CallStreamingWithOpts(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("set OPENAI_API_KEY to run integration test")
	}

	m, err := NewOpenAIModel(shared.ChatModelGPT4o)
	if err != nil {
		t.Fatalf("NewOpenAIModel: %v", err)
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

// TestOpenAIModel_CallStreaming_DeltaAccumulation tests that deltas are accumulated correctly
func TestOpenAIModel_CallStreaming_DeltaAccumulation(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("set OPENAI_API_KEY to run integration test")
	}

	m, err := NewOpenAIModel(shared.ChatModelGPT4o)
	if err != nil {
		t.Fatalf("NewOpenAIModel: %v", err)
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
	if err != nil {
		if isQuotaExhaustedError(err) {
			t.Skipf("Skipping test due to quota exhaustion: %v", err)
		}
		assert.NoError(t, err)
	}
	assert.Greater(t, len(deltas), 0, "should receive multiple deltas")

	// Accumulate deltas and verify they match final content
	accumulated := strings.Join(deltas, "")
	finalContent := events[len(events)-1].Content
	assert.Equal(t, accumulated, finalContent, "accumulated deltas should match final content")
}

// TestOpenAIModel_CallStreaming_ToolCalls tests tool calls in streaming mode
func TestOpenAIModel_CallStreaming_ToolCalls(t *testing.T) {
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

	lsTool := shared.FunctionDefinitionParam{
		Name:        "get_weather",
		Description: param.NewOpt("Get the weather for a location"),
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"location": map[string]interface{}{
					"type":        "string",
					"description": "The city and state, e.g. San Francisco, CA",
				},
			},
			"required": []string{"location"},
		},
	}

	err = cw.RegisterTool("get_weather", lsTool, ToolRunnerFunc(func(ctx context.Context, args json.RawMessage) (string, error) {
		return "Sunny, 72°F", nil
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

// TestOpenAIModel_CallStreaming_ErrorHandling tests error handling mid-stream
func TestOpenAIModel_CallStreaming_ErrorHandling(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("set OPENAI_API_KEY to run integration test")
	}

	m, err := NewOpenAIModel(shared.ChatModelGPT4o)
	if err != nil {
		t.Fatalf("NewOpenAIModel: %v", err)
	}

	inputs := []Record{
		{Source: Prompt, Content: "Say hello."},
	}

	// Test callback error propagation
	callbackError := fmt.Errorf("callback error")
	callback := func(chunk StreamChunk) error {
		if chunk.Delta != "" {
			return callbackError
		}
		return nil
	}

	_, _, err = m.CallStreaming(context.Background(), inputs, callback)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "callback error")
}

// TestOpenAIModel_CallStreaming_ContextCancellation tests context cancellation
func TestOpenAIModel_CallStreaming_ContextCancellation(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("set OPENAI_API_KEY to run integration test")
	}

	m, err := NewOpenAIModel(shared.ChatModelGPT4o)
	if err != nil {
		t.Fatalf("NewOpenAIModel: %v", err)
	}

	inputs := []Record{
		{Source: Prompt, Content: "Count from 1 to 100 slowly."},
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var chunkCount int
	callback := func(chunk StreamChunk) error {
		chunkCount++
		// Cancel after first chunk
		if chunkCount == 1 {
			cancel()
		}
		return nil
	}

	// This should eventually fail due to context cancellation
	_, _, err = m.CallStreaming(ctx, inputs, callback)
	// The error might be context cancellation or stream error
	assert.Error(t, err)
}

// TestOpenAIModel_CallStreaming_DisableTools tests that tools can be disabled
func TestOpenAIModel_CallStreaming_DisableTools(t *testing.T) {
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

	lsTool := shared.FunctionDefinitionParam{
		Name:        "test_tool",
		Description: param.NewOpt("A test tool"),
		Parameters: map[string]interface{}{
			"type":       "object",
			"properties": map[string]interface{}{},
		},
	}

	err = cw.RegisterTool("test_tool", lsTool, ToolRunnerFunc(func(ctx context.Context, args json.RawMessage) (string, error) {
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

	err = cw.AddPrompt("Say hello.")
	assert.NoError(t, err)

	// Call with tools disabled
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

// TestOpenAIModel_CallStreamingWithThreadingAndOpts tests streaming with threading fallback behavior
func TestOpenAIModel_CallStreamingWithThreadingAndOpts(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("set OPENAI_API_KEY to run integration test")
	}

	m, err := NewOpenAIModel(shared.ChatModelGPT4o)
	if err != nil {
		t.Fatalf("NewOpenAIModel: %v", err)
	}

	inputs := []Record{
		{Source: Prompt, Content: "Say 'hello' and nothing else."},
	}

	// Test 1: Server-side threading should return error
	var receivedChunks []StreamChunk
	callback := func(chunk StreamChunk) error {
		receivedChunks = append(receivedChunks, chunk)
		return nil
	}

	_, _, err = m.CallStreamingWithThreadingAndOpts(
		context.Background(),
		true, // useServerSideThreading = true
		nil,  // lastResponseID
		inputs,
		CallModelOpts{},
		callback,
	)
	assert.Error(t, err, "should return error when server-side threading is requested")
	assert.Contains(t, err.Error(), "server-side threading not supported")

	// Test 2: Client-side threading (fallback) should work
	receivedChunks = nil
	events, tokens, err := m.CallStreamingWithThreadingAndOpts(
		context.Background(),
		false, // useServerSideThreading = false
		nil,   // lastResponseID
		inputs,
		CallModelOpts{},
		callback,
	)
	if err != nil {
		if isQuotaExhaustedError(err) {
			t.Skipf("Skipping test due to quota exhaustion: %v", err)
		}
		assert.NoError(t, err, "should work with client-side threading fallback")
	}
	assert.Greater(t, len(events), 0, "should return events")
	assert.Greater(t, tokens, 0, "should return token count")
	assert.Greater(t, len(receivedChunks), 0, "should receive streaming chunks")

	// Verify we got a done chunk
	var hasDoneChunk bool
	for _, chunk := range receivedChunks {
		if chunk.Done {
			hasDoneChunk = true
			break
		}
	}
	assert.True(t, hasDoneChunk, "should receive a done chunk")

	// Verify accumulated text matches final event
	var accumulatedText strings.Builder
	for _, chunk := range receivedChunks {
		if chunk.Delta != "" {
			accumulatedText.WriteString(chunk.Delta)
		}
	}
	finalContent := accumulatedText.String()
	assert.NotEmpty(t, finalContent)
	assert.Equal(t, finalContent, events[len(events)-1].Content, "accumulated text should match final event")
}
