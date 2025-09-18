package contextwindow

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/packages/param"
	"github.com/openai/openai-go/v2/shared"
	"github.com/stretchr/testify/assert"
	_ "modernc.org/sqlite"
)

type dummyModel struct {
	cw      *ContextWindow
	events  []Record
	closeDB bool
}

func (m *dummyModel) Call(ctx context.Context, inputs []Record) ([]Record, int, error) {
	if m.closeDB && m.cw != nil {
		m.cw.db.Close()
	}
	return m.events, 0, nil
}

type MockModel struct {
	LastOptsDisableTools bool
	events               []Record
}

func (m *MockModel) Call(ctx context.Context, inputs []Record) ([]Record, int, error) {
	m.LastOptsDisableTools = false // Default behavior
	return m.events, 0, nil
}

func (m *MockModel) CallWithOpts(ctx context.Context, inputs []Record, opts CallModelOpts) ([]Record, int, error) {
	m.LastOptsDisableTools = opts.DisableTools
	return m.events, 0, nil
}

func (m *MockModel) CallWithThreadingAndOpts(
	ctx context.Context,
	useServerSideThreading bool,
	lastResponseID *string,
	inputs []Record,
	opts CallModelOpts,
) ([]Record, *string, int, error) {
	m.LastOptsDisableTools = opts.DisableTools
	return m.events, nil, 0, nil
}

func TestNewContextWindowAndClose(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")

	db, err := NewContextDB(path)
	assert.NoError(t, err)

	cw, err := NewContextWindow(db, &dummyModel{}, "")
	assert.NoError(t, err)
	assert.NotNil(t, cw.db)

	// Test record insertion before closing
	err = cw.AddPrompt("test prompt")
	assert.NoError(t, err)

	// Now close and test error
	err = cw.Close()
	assert.NoError(t, err)

	// Try to add another prompt after closing
	err = cw.AddPrompt("should fail")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "sql: database is closed")
}

func TestAddMethodsErrorPropagation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)

	cw, err := NewContextWindow(db, &dummyModel{}, "")
	assert.NoError(t, err)
	assert.NoError(t, cw.db.Close())
	assert.Contains(t, cw.AddPrompt("p").Error(), "sql: database is closed")
	assert.Contains(t, cw.AddToolCall("t", "a").Error(), "sql: database is closed")
	assert.Contains(t, cw.AddToolOutput("o").Error(), "sql: database is closed")
}

func TestAddPromptEstTokens(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)

	cw, err := NewContextWindow(db, &dummyModel{}, "")
	assert.NoError(t, err)
	err = cw.AddPrompt("hello world")
	assert.NoError(t, err)
	recs, err := cw.LiveRecords()
	assert.NoError(t, err)
	assert.Equal(t, tokenCount("hello world"), recs[0].EstTokens)
}

func TestCallModelInsertRecordError(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)
	m := &dummyModel{closeDB: true}
	cw, err := NewContextWindow(db, m, "")
	assert.NoError(t, err)
	m.cw = cw
	m.events = []Record{{
		Source:    ModelResp,
		Content:   "x",
		Live:      true,
		EstTokens: tokenCount("x"),
	}}
	_, err = cw.CallModel(context.Background())
	assert.Contains(t, err.Error(), "sql: database is closed")
}

func TestContextWindowToolCall(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("set OPENAI_API_KEY to run integration test")
	}

	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)

	m, err := NewOpenAIModel(shared.ChatModelGPT4o)
	assert.NoError(t, err)
	cw, err := NewContextWindow(db, m, "")
	assert.NoError(t, err)

	lsTool := shared.FunctionDefinitionParam{
		Name:        "ls",
		Description: param.NewOpt("list files in a directory"),
		Parameters: map[string]interface{}{
			"type":       "object",
			"properties": map[string]interface{}{},
		},
	}

	err = cw.RegisterTool("ls", lsTool, ToolRunnerFunc(func(ctx context.Context, args json.RawMessage) (string, error) {
		return "go.mod", nil
	}))
	assert.NoError(t, err)

	err = cw.AddPrompt("Please list the files in the current directory.")
	assert.NoError(t, err)
	reply, err := cw.CallModel(context.Background())
	assert.NoError(t, err)
	assert.True(t, strings.Contains(reply, "go.mod"))
	recs, err := cw.LiveRecords()
	assert.NoError(t, err)

	var (
		foundPrompt bool
		foundCall   bool
		foundOutput bool
		foundResp   bool
	)

	for _, r := range recs {
		if r.Source == Prompt && strings.Contains(r.Content, "list the") {
			foundPrompt = true
		}
		if r.Source == ToolCall && strings.Contains(r.Content, "ls") {
			foundCall = true
		}
		if r.Source == ToolOutput && strings.Contains(r.Content, "go.mod") {
			foundOutput = true
		}
		if r.Source == ModelResp && len(r.Content) > 0 {
			foundResp = true
		}
	}

	assert.True(t, foundPrompt)
	assert.True(t, foundCall)
	assert.True(t, foundOutput)
	assert.True(t, foundResp)
}

// Context management tests

func TestCreateAndListContexts(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)

	cw, err := NewContextWindow(db, &dummyModel{}, "")
	assert.NoError(t, err)
	defer cw.Close()

	contexts, err := cw.ListContexts()
	assert.NoError(t, err)
	assert.Equal(t, 1, len(contexts))
	assert.NotEmpty(t, contexts[0].Name)

	err = cw.CreateContext("test-context")
	assert.NoError(t, err)

	contexts, err = cw.ListContexts()
	assert.NoError(t, err)
	assert.Equal(t, 2, len(contexts))

	found := false
	for _, ctx := range contexts {
		if ctx.Name == "test-context" {
			found = true
			break
		}
	}
	assert.True(t, found)
}

func TestContextWindowIsolation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)

	cw1, err := NewContextWindow(db, &dummyModel{}, "context1")
	assert.NoError(t, err)
	defer cw1.Close()

	cw2, err := NewContextWindow(db, &dummyModel{}, "context2")
	assert.NoError(t, err)
	defer cw2.Close()

	err = cw1.AddPrompt("Hello from context 1")
	assert.NoError(t, err)

	err = cw2.AddPrompt("Hello from context 2")
	assert.NoError(t, err)

	ctx1Records, err := cw1.LiveRecords()
	assert.NoError(t, err)
	assert.Equal(t, 1, len(ctx1Records))
	assert.Contains(t, ctx1Records[0].Content, "context 1")

	ctx2Records, err := cw2.LiveRecords()
	assert.NoError(t, err)
	assert.Equal(t, 1, len(ctx2Records))
	assert.Contains(t, ctx2Records[0].Content, "context 2")
}

func TestNewContextWindowWithContext(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)

	cw, err := NewContextWindow(db, &dummyModel{}, "custom")
	assert.NoError(t, err)
	defer cw.Close()

	assert.Equal(t, "custom", cw.GetCurrentContext())

	ctx, err := cw.GetCurrentContextInfo()
	assert.NoError(t, err)
	assert.Equal(t, "custom", ctx.Name)

	contexts, err := cw.ListContexts()
	assert.NoError(t, err)
	assert.Equal(t, 1, len(contexts))
	assert.Equal(t, "custom", contexts[0].Name)
}

func TestContextNameConflict(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)

	cw, err := NewContextWindow(db, &dummyModel{}, "")
	assert.NoError(t, err)
	defer cw.Close()

	err = cw.CreateContext("test")
	assert.NoError(t, err)

	// With the new get-or-create behavior, this should succeed
	err = cw.CreateContext("test")
	assert.NoError(t, err)

	// Verify the context exists and can be accessed
	ctx, err := cw.GetContext("test")
	assert.NoError(t, err)
	assert.Equal(t, "test", ctx.Name)
}

func TestCreateContextWithExistingName(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)
	defer db.Close()

	_, err = NewContextWindow(db, &dummyModel{}, "shared")
	assert.NoError(t, err)
	// Note: we don't close cw1 here because it doesn't own the DB.

	// Second instance should use existing context
	cw2, err := NewContextWindow(db, &dummyModel{}, "shared")
	assert.NoError(t, err)

	assert.Equal(t, "shared", cw2.GetCurrentContext())

	contexts, err := cw2.ListContexts()
	assert.NoError(t, err)
	assert.Equal(t, 1, len(contexts))
	assert.Equal(t, "shared", contexts[0].Name)
}

func TestNewContextWindowWithDB(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")

	// Open database manually
	db, err := sql.Open("sqlite", path)
	assert.NoError(t, err)
	defer db.Close()

	// Initialize schema manually
	err = InitializeSchema(db)
	assert.NoError(t, err)

	// Create context window with existing DB
	cw, err := NewContextWindow(db, &dummyModel{}, "shared-db")
	assert.NoError(t, err)

	assert.Equal(t, "shared-db", cw.GetCurrentContext())

	// Add some data
	err = cw.AddPrompt("Hello from shared DB")
	assert.NoError(t, err)

	// Create another context window with same DB
	cw2, err := NewContextWindow(db, &dummyModel{}, "another-context")
	assert.NoError(t, err)

	assert.Equal(t, "another-context", cw2.GetCurrentContext())

	// Verify data isolation
	records1, err := cw.LiveRecords()
	assert.NoError(t, err)
	assert.Equal(t, 1, len(records1))

	records2, err := cw2.LiveRecords()
	assert.NoError(t, err)
	assert.Equal(t, 0, len(records2))

	// Both should see the same contexts list
	contexts1, err := cw.ListContexts()
	assert.NoError(t, err)
	contexts2, err := cw2.ListContexts()
	assert.NoError(t, err)
	assert.Equal(t, len(contexts1), len(contexts2))
	assert.Equal(t, 2, len(contexts1)) // shared-db + another-context

	// Note: We don't call cw.Close() or cw2.Close() because they don't own the DB
}

// testMiddleware collects tool call events for testing
type testMiddleware struct {
	toolCalls   []string
	toolResults []string
	mu          sync.Mutex
}

func (tm *testMiddleware) OnToolCall(ctx context.Context, name, args string) {
	tm.mu.Lock()
	defer tm.mu.Unlock()
	tm.toolCalls = append(tm.toolCalls, fmt.Sprintf("%s(%s)", name, args))
}

func (tm *testMiddleware) OnToolResult(ctx context.Context, name, result string, err error) {
	tm.mu.Lock()
	defer tm.mu.Unlock()
	if err != nil {
		tm.toolResults = append(tm.toolResults, fmt.Sprintf("%s:error:%s", name, err.Error()))
	} else {
		tm.toolResults = append(tm.toolResults, fmt.Sprintf("%s:%s", name, result))
	}
}

// toolCallModel simulates a model that makes tool calls and executes middleware
type toolCallModel struct {
	response   string
	middleware []Middleware
}

func (m *toolCallModel) Call(ctx context.Context, inputs []Record) ([]Record, int, error) {
	// Simulate middleware execution for tool calls
	for _, mw := range m.middleware {
		mw.OnToolCall(ctx, "hello_world", "{}")
		mw.OnToolResult(ctx, "hello_world", "hello world", nil)
	}

	// Simulate tool call events
	events := []Record{
		{
			Source:    ToolCall,
			Content:   "hello_world({})",
			Live:      true,
			EstTokens: 5,
		},
		{
			Source:    ToolOutput,
			Content:   "hello world",
			Live:      true,
			EstTokens: 2,
		},
		{
			Source:    ModelResp,
			Content:   m.response,
			Live:      true,
			EstTokens: 10,
		},
	}
	return events, 20, nil
}

// SetMiddleware allows the test model to receive middleware
func (m *toolCallModel) SetMiddleware(middleware []Middleware) {
	m.middleware = middleware
}

func TestMiddlewareHooks(t *testing.T) {
	path := filepath.Join(t.TempDir(), "middleware-test.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)
	defer db.Close()

	// Create a test model that simulates tool calls
	testModel := &toolCallModel{
		response: "I used the hello_world tool and got the result.",
	}

	cw, err := NewContextWindow(db, testModel, "test-middleware")
	assert.NoError(t, err)

	// Register test middleware - this should also update the model
	middleware := &testMiddleware{}
	cw.AddMiddleware(middleware)

	// Make a call that should trigger tool calls
	response, err := cw.CallModel(context.Background())
	assert.NoError(t, err)
	assert.Equal(t, "I used the hello_world tool and got the result.", response)

	// Verify middleware was called
	middleware.mu.Lock()
	assert.Len(t, middleware.toolCalls, 1)
	assert.Equal(t, "hello_world({})", middleware.toolCalls[0])
	assert.Len(t, middleware.toolResults, 1)
	assert.Equal(t, "hello_world:hello world", middleware.toolResults[0])
	middleware.mu.Unlock()
}

func TestContextWindowToolManagement(t *testing.T) {
	db, err := NewContextDB(":memory:")
	assert.NoError(t, err)
	defer db.Close()

	mockModel := &mockModel{}
	cw, err := NewContextWindow(db, mockModel, "test-context")
	assert.NoError(t, err)

	// Initially no tools
	tools, err := cw.ListTools()
	assert.NoError(t, err)
	assert.Len(t, tools, 0)

	// Register a tool (automatically adds it to the context)
	err = cw.RegisterTool("test_tool", "test definition", ToolRunnerFunc(func(ctx context.Context, args json.RawMessage) (string, error) {
		return "test result", nil
	}))
	assert.NoError(t, err)

	// Tool should be listed
	tools, err = cw.ListTools()
	assert.NoError(t, err)
	assert.Len(t, tools, 1)
	assert.Equal(t, "test_tool", tools[0])

	// Tool should exist
	has, err := cw.HasTool("test_tool")
	assert.NoError(t, err)
	assert.True(t, has)

	// Register another tool
	err = cw.RegisterTool("another_tool", "another definition", ToolRunnerFunc(func(ctx context.Context, args json.RawMessage) (string, error) {
		return "another result", nil
	}))
	assert.NoError(t, err)

	tools, err = cw.ListTools()
	assert.NoError(t, err)
	assert.Len(t, tools, 2)

	// Both tools should exist
	has, err = cw.HasTool("test_tool")
	assert.NoError(t, err)
	assert.True(t, has)

	has, err = cw.HasTool("another_tool")
	assert.NoError(t, err)
	assert.True(t, has)
}

func TestContextToolPersistence(t *testing.T) {
	db, err := NewContextDB(":memory:")
	assert.NoError(t, err)
	defer db.Close()

	mockModel := &mockModel{}

	// Create first context window and add tools
	cw1, err := NewContextWindow(db, mockModel, "persistence-test-context")
	assert.NoError(t, err)

	// Register tools with first context window
	err = cw1.RegisterTool("persistent_tool", "persistent definition", ToolRunnerFunc(func(ctx context.Context, args json.RawMessage) (string, error) {
		return "persistent result", nil
	}))
	assert.NoError(t, err)
	err = cw1.RegisterTool("another_tool", "another definition", ToolRunnerFunc(func(ctx context.Context, args json.RawMessage) (string, error) {
		return "another result", nil
	}))
	assert.NoError(t, err)

	// Create second context window with same context name
	cw2, err := NewContextWindow(db, mockModel, "persistence-test-context")
	assert.NoError(t, err)

	// Tool names are persisted, definitions are provided by the caller

	// Tools should be persisted
	tools, err := cw2.ListTools()
	assert.NoError(t, err)
	assert.Len(t, tools, 2)
	assert.Contains(t, tools, "persistent_tool")
	assert.Contains(t, tools, "another_tool")

	// Create context window with different name
	cw3, err := NewContextWindow(db, mockModel, "different-context")
	assert.NoError(t, err)

	// Should have no tools
	tools, err = cw3.ListTools()
	assert.NoError(t, err)
	assert.Len(t, tools, 0)
}

// mockModel for testing - implements Model interface
type mockModel struct{}

func (m *mockModel) Call(ctx context.Context, inputs []Record) ([]Record, int, error) {
	return []Record{
		{
			Source:    ModelResp,
			Content:   "Mock response",
			Live:      true,
			EstTokens: 10,
		},
	}, 10, nil
}

type dummyModelTokens struct {
	events []Record
	tokens int
}

func (m *dummyModelTokens) Call(ctx context.Context, inputs []Record) ([]Record, int, error) {
	return m.events, m.tokens, nil
}

func TestTokenCounts(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")

	db, err := NewContextDB(path)
	assert.NoError(t, err)

	m := &dummyModelTokens{
		events: []Record{{
			Source:    ModelResp,
			Content:   "z",
			Live:      true,
			EstTokens: tokenCount("z"),
		}},
		tokens: 5,
	}
	cw, err := NewContextWindow(db, m, "")
	assert.NoError(t, err)
	err = cw.AddPrompt("a b c")
	assert.NoError(t, err)
	live, err := cw.LiveTokens()
	assert.NoError(t, err)
	assert.Equal(t, tokenCount("a b c"), live)
	_, err = cw.CallModel(context.Background())
	assert.NoError(t, err)
	assert.Equal(t, 5, cw.TotalTokens())
	live, err = cw.LiveTokens()
	assert.NoError(t, err)
	assert.Equal(t, tokenCount("a b c")+tokenCount("z"), live)
}

func TestTokenUsage(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")

	db, err := NewContextDB(path)
	assert.NoError(t, err)

	m := &dummyModelTokens{
		events: []Record{{
			Source:    ModelResp,
			Content:   "response",
			Live:      true,
			EstTokens: tokenCount("response"),
		}},
		tokens: 10,
	}
	cw, err := NewContextWindow(db, m, "")
	assert.NoError(t, err)

	// Set a custom max tokens for testing
	cw.SetMaxTokens(100)
	assert.Equal(t, 100, cw.MaxTokens())

	err = cw.AddPrompt("hello world")
	assert.NoError(t, err)

	// Before calling model
	usage, err := cw.TokenUsage()
	assert.NoError(t, err)
	assert.Equal(t, tokenCount("hello world"), usage.Live)
	assert.Equal(t, 0, usage.Total) // no model calls yet
	assert.Equal(t, 100, usage.Max)
	assert.Equal(t, float64(usage.Live)/100.0, usage.Percent)

	// After calling model
	_, err = cw.CallModel(context.Background())
	assert.NoError(t, err)

	usage, err = cw.TokenUsage()
	assert.NoError(t, err)
	assert.Equal(t, tokenCount("hello world")+tokenCount("response"), usage.Live)
	assert.Equal(t, 10, usage.Total) // tokens from model call
	assert.Equal(t, 100, usage.Max)
	expectedPercent := float64(usage.Live) / 100.0
	assert.Equal(t, expectedPercent, usage.Percent)
}

func TestServerSideThreading(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test")
	}
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	db, err := NewContextDB(":memory:")
	assert.NoError(t, err)
	defer db.Close()

	// Test with responses model which supports server-side threading
	model, err := NewOpenAIResponsesModel(ResponsesModel4o)
	assert.NoError(t, err)

	// Create context with server-side threading enabled
	cw, err := NewContextWindowWithThreading(db, model, "test-threading", true)
	assert.NoError(t, err)
	defer cw.Close()

	// Verify server-side threading is enabled
	enabled, err := cw.IsServerSideThreadingEnabled()
	assert.NoError(t, err)
	assert.True(t, enabled)

	// Add a system prompt and user prompt
	err = cw.SetSystemPrompt("You are a helpful assistant. Please respond briefly.")
	assert.NoError(t, err)

	err = cw.AddPrompt("Hello, who are you?")
	assert.NoError(t, err)

	// Make first call - this should not use server-side threading since there's no previous response
	ctx := context.Background()
	resp1, err := cw.CallModel(ctx)
	assert.NoError(t, err)
	assert.NotEmpty(t, resp1)

	// Add another prompt
	err = cw.AddPrompt("What's 2+2?")
	assert.NoError(t, err)

	// Make second call - this should use server-side threading
	resp2, err := cw.CallModel(ctx)
	assert.NoError(t, err)
	assert.NotEmpty(t, resp2)

	// Verify that the context has a last response ID stored
	contextInfo, err := cw.GetCurrentContextInfo()
	assert.NoError(t, err)
	assert.NotNil(t, contextInfo.LastResponseID)
	assert.NotEmpty(t, *contextInfo.LastResponseID)
}

func TestClientSideThreading(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test")
	}
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	db, err := NewContextDB(":memory:")
	assert.NoError(t, err)
	defer db.Close()

	// Test with completions model which doesn't support server-side threading
	model, err := NewOpenAIModel(openai.ChatModelGPT4oMini)
	assert.NoError(t, err)

	// Create context with server-side threading disabled (default)
	cw, err := NewContextWindow(db, model, "test-client-threading")
	assert.NoError(t, err)
	defer cw.Close()

	// Verify server-side threading is disabled
	enabled, err := cw.IsServerSideThreadingEnabled()
	assert.NoError(t, err)
	assert.False(t, enabled)

	// Add a system prompt and user prompt
	err = cw.SetSystemPrompt("You are a helpful assistant. Please respond briefly.")
	assert.NoError(t, err)

	err = cw.AddPrompt("Hello, who are you?")
	assert.NoError(t, err)

	// Make first call
	ctx := context.Background()
	resp1, err := cw.CallModel(ctx)
	assert.NoError(t, err)
	assert.NotEmpty(t, resp1)

	// Add another prompt
	err = cw.AddPrompt("What's 2+2?")
	assert.NoError(t, err)

	// Make second call - should use client-side threading
	resp2, err := cw.CallModel(ctx)
	assert.NoError(t, err)
	assert.NotEmpty(t, resp2)

	// Context should not have a last response ID since client-side threading doesn't set it
	contextInfo, err := cw.GetCurrentContextInfo()
	assert.NoError(t, err)
	assert.Nil(t, contextInfo.LastResponseID)
}

func TestToggleThreadingMode(t *testing.T) {
	db, err := NewContextDB(":memory:")
	assert.NoError(t, err)
	defer db.Close()

	// Use dummy model to avoid needing API key for this test
	model := &dummyModel{}

	// Create context with client-side threading initially
	cw, err := NewContextWindow(db, model, "test-toggle")
	assert.NoError(t, err)
	defer cw.Close()

	// Initially should be disabled
	enabled, err := cw.IsServerSideThreadingEnabled()
	assert.NoError(t, err)
	assert.False(t, enabled)

	// Enable server-side threading
	err = cw.SetServerSideThreading(true)
	assert.NoError(t, err)

	// Should now be enabled
	enabled, err = cw.IsServerSideThreadingEnabled()
	assert.NoError(t, err)
	assert.True(t, enabled)

	// Disable again
	err = cw.SetServerSideThreading(false)
	assert.NoError(t, err)

	// Should be disabled
	enabled, err = cw.IsServerSideThreadingEnabled()
	assert.NoError(t, err)
	assert.False(t, enabled)
}

func TestDatabaseSchemaWithResponseID(t *testing.T) {
	db, err := NewContextDB(":memory:")
	assert.NoError(t, err)
	defer db.Close()

	// Create a context with server-side threading
	ctx, err := CreateContextWithThreading(db, "test-schema", true)
	assert.NoError(t, err)
	assert.True(t, ctx.UseServerSideThreading)
	assert.Nil(t, ctx.LastResponseID)

	// Insert a record with response ID
	responseID := "resp_123456"
	rec, err := InsertRecordWithResponseID(db, ctx.ID, ModelResp, "Hello world", true, &responseID)
	assert.NoError(t, err)
	assert.NotNil(t, rec.ResponseID)
	assert.Equal(t, responseID, *rec.ResponseID)

	// Update context's last response ID
	err = UpdateContextLastResponseID(db, ctx.ID, responseID)
	assert.NoError(t, err)

	// Retrieve context and verify
	updatedCtx, err := GetContext(db, ctx.ID)
	assert.NoError(t, err)
	assert.NotNil(t, updatedCtx.LastResponseID)
	assert.Equal(t, responseID, *updatedCtx.LastResponseID)

	// List records and verify response ID is preserved
	recs, err := ListRecordsInContext(db, ctx.ID)
	assert.NoError(t, err)
	assert.Len(t, recs, 1)
	assert.NotNil(t, recs[0].ResponseID)
	assert.Equal(t, responseID, *recs[0].ResponseID)
}

func TestSetContextServerSideThreading(t *testing.T) {
	db, err := NewContextDB(":memory:")
	assert.NoError(t, err)
	defer db.Close()

	// Create a context with default threading (client-side)
	ctx, err := CreateContext(db, "test-set-threading")
	assert.NoError(t, err)
	assert.False(t, ctx.UseServerSideThreading)

	// Enable server-side threading
	err = SetContextServerSideThreading(db, ctx.ID, true)
	assert.NoError(t, err)

	// Verify it's enabled
	updatedCtx, err := GetContext(db, ctx.ID)
	assert.NoError(t, err)
	assert.True(t, updatedCtx.UseServerSideThreading)

	// Disable server-side threading
	err = SetContextServerSideThreading(db, ctx.ID, false)
	assert.NoError(t, err)

	// Verify it's disabled
	updatedCtx, err = GetContext(db, ctx.ID)
	assert.NoError(t, err)
	assert.False(t, updatedCtx.UseServerSideThreading)
}

func TestServerSideThreadingFallback(t *testing.T) {
	db, err := NewContextDB(":memory:")
	assert.NoError(t, err)
	defer db.Close()

	// Create a mock responses model that we can control
	mockModel := &mockResponsesModel{}

	// Create context with server-side threading enabled
	cw, err := NewContextWindowWithThreading(db, mockModel, "test-fallback", true)
	assert.NoError(t, err)
	defer cw.Close()

	// Add a prompt and make first call (no previous response ID, should use client-side)
	err = cw.AddPrompt("Hello")
	assert.NoError(t, err)

	ctx := context.Background()
	_, err = cw.CallModel(ctx)
	assert.NoError(t, err)

	// Verify the mock was called with client-side threading (no previous response ID)
	assert.False(t, mockModel.lastCallUsedServerSide)
	assert.Nil(t, mockModel.lastPreviousResponseID)

	// Add another prompt - this should use server-side threading
	err = cw.AddPrompt("How are you?")
	assert.NoError(t, err)

	_, err = cw.CallModel(ctx)
	assert.NoError(t, err)

	// Verify the mock was called with server-side threading
	assert.True(t, mockModel.lastCallUsedServerSide)
	assert.NotNil(t, mockModel.lastPreviousResponseID)
}

// Mock model for testing server-side threading behavior
type mockResponsesModel struct {
	lastCallUsedServerSide bool
	lastPreviousResponseID *string
	lastInputs             []Record
}

func (m *mockResponsesModel) Call(ctx context.Context, inputs []Record) ([]Record, int, error) {
	events, _, tokens, err := m.CallWithThreading(ctx, false, nil, inputs)
	return events, tokens, err
}

func (m *mockResponsesModel) CallWithThreading(
	ctx context.Context,
	useServerSideThreading bool,
	lastResponseID *string,
	inputs []Record,
) ([]Record, *string, int, error) {
	// Record the call parameters
	m.lastCallUsedServerSide = useServerSideThreading && lastResponseID != nil
	m.lastPreviousResponseID = lastResponseID
	m.lastInputs = inputs

	// Return a mock response
	responseID := "mock_response_123"
	return []Record{
		{
			Source:     ModelResp,
			Content:    "Mock response",
			Live:       true,
			EstTokens:  10,
			ResponseID: &responseID,
		},
	}, &responseID, 25, nil
}

func (m *mockResponsesModel) SetToolExecutor(executor ToolExecutor) {
	// No-op for mock
}

func (m *mockResponsesModel) SetMiddleware(middleware []Middleware) {
	// No-op for mock
}

func TestSchemaMigrationWithNewContextDB(t *testing.T) {
	// This test verifies that NewContextDB properly handles schema migration
	db, err := NewContextDB(":memory:")
	assert.NoError(t, err)
	defer db.Close()

	// Should be able to create a context with server-side threading
	ctx, err := CreateContextWithThreading(db, "migration-test", true)
	assert.NoError(t, err)
	assert.True(t, ctx.UseServerSideThreading)

	// Should be able to read it back
	retrievedCtx, err := GetContextByName(db, "migration-test")
	assert.NoError(t, err)
	assert.Equal(t, ctx.ID, retrievedCtx.ID)
	assert.True(t, retrievedCtx.UseServerSideThreading)

	// Should be able to insert records with response IDs
	responseID := "test_resp_123"
	rec, err := InsertRecordWithResponseID(db, ctx.ID, ModelResp, "Test response", true, &responseID)
	assert.NoError(t, err)
	assert.NotNil(t, rec.ResponseID)
	assert.Equal(t, responseID, *rec.ResponseID)

	// Should be able to update last response ID
	err = UpdateContextLastResponseID(db, ctx.ID, responseID)
	assert.NoError(t, err)

	finalCtx, err := GetContext(db, ctx.ID)
	assert.NoError(t, err)
	assert.NotNil(t, finalCtx.LastResponseID)
	assert.Equal(t, responseID, *finalCtx.LastResponseID)
}

func TestCallModelWithOpts(t *testing.T) {
	db, err := NewContextDB(":memory:")
	assert.NoError(t, err)
	defer db.Close()

	mockModel := &MockModel{}
	cw, err := NewContextWindow(db, mockModel, "test-context")
	assert.NoError(t, err)

	err = cw.AddPrompt("test prompt")
	assert.NoError(t, err)

	// Test default behavior (tools enabled)
	_, err = cw.CallModel(context.Background())
	assert.NoError(t, err)
	assert.False(t, mockModel.LastOptsDisableTools, "Expected tools to be enabled by default")

	// Test with tools disabled
	_, err = cw.CallModelWithOpts(context.Background(), CallModelOpts{DisableTools: true})
	assert.NoError(t, err)
	assert.True(t, mockModel.LastOptsDisableTools, "Expected tools to be disabled")
}

func TestSwitchContext(t *testing.T) {
	db, err := NewContextDB(":memory:")
	assert.NoError(t, err)
	defer db.Close()

	mockModel := &dummyModel{}
	cw, err := NewContextWindow(db, mockModel, "context1")
	assert.NoError(t, err)

	// Add some data to context1
	err = cw.AddPrompt("Hello from context1")
	assert.NoError(t, err)

	// Create a second context
	err = cw.CreateContext("context2")
	assert.NoError(t, err)

	// Switch to context2
	err = cw.SwitchContext("context2")
	assert.NoError(t, err)
	assert.Equal(t, "context2", cw.GetCurrentContext())

	// Add different data to context2
	err = cw.AddPrompt("Hello from context2")
	assert.NoError(t, err)

	// Verify context2 has only its own data
	ctx2Records, err := cw.LiveRecords()
	assert.NoError(t, err)
	assert.Len(t, ctx2Records, 1)
	assert.Contains(t, ctx2Records[0].Content, "context2")

	// Switch back to context1
	err = cw.SwitchContext("context1")
	assert.NoError(t, err)
	assert.Equal(t, "context1", cw.GetCurrentContext())

	// Verify context1 still has its original data
	ctx1Records, err := cw.LiveRecords()
	assert.NoError(t, err)
	assert.Len(t, ctx1Records, 1)
	assert.Contains(t, ctx1Records[0].Content, "context1")
}

func TestSwitchContextErrors(t *testing.T) {
	db, err := NewContextDB(":memory:")
	assert.NoError(t, err)
	defer db.Close()

	mockModel := &dummyModel{}
	cw, err := NewContextWindow(db, mockModel, "initial")
	assert.NoError(t, err)

	// Test switching to empty name
	err = cw.SwitchContext("")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "context name cannot be empty")
	assert.Equal(t, "initial", cw.GetCurrentContext()) // Should remain unchanged

	// Test switching to non-existent context - should now create it
	err = cw.SwitchContext("non-existent")
	assert.NoError(t, err)
	assert.Equal(t, "non-existent", cw.GetCurrentContext())

	// Verify the context was created
	ctx, err := cw.GetContext("non-existent")
	assert.NoError(t, err)
	assert.Equal(t, "non-existent", ctx.Name)
}

func TestSwitchContextWithComplexOperations(t *testing.T) {
	db, err := NewContextDB(":memory:")
	assert.NoError(t, err)
	defer db.Close()

	mockModel := &dummyModelTokens{
		events: []Record{{
			Source:    ModelResp,
			Content:   "Response",
			Live:      true,
			EstTokens: tokenCount("Response"),
		}},
		tokens: 10,
	}
	cw, err := NewContextWindow(db, mockModel, "work")
	assert.NoError(t, err)

	// Create contexts for different purposes
	err = cw.CreateContext("personal")
	assert.NoError(t, err)
	err = cw.CreateContext("research")
	assert.NoError(t, err)

	// Work context operations
	err = cw.SetSystemPrompt("You are a work assistant")
	assert.NoError(t, err)
	err = cw.AddPrompt("What are my tasks today?")
	assert.NoError(t, err)
	_, err = cw.CallModel(context.Background())
	assert.NoError(t, err)

	// Switch to personal context
	err = cw.SwitchContext("personal")
	assert.NoError(t, err)
	err = cw.SetSystemPrompt("You are a personal assistant")
	assert.NoError(t, err)
	err = cw.AddPrompt("What's the weather like?")
	assert.NoError(t, err)
	_, err = cw.CallModel(context.Background())
	assert.NoError(t, err)

	// Switch to research context
	err = cw.SwitchContext("research")
	assert.NoError(t, err)
	err = cw.SetSystemPrompt("You are a research assistant")
	assert.NoError(t, err)
	err = cw.AddPrompt("Explain quantum computing")
	assert.NoError(t, err)
	_, err = cw.CallModel(context.Background())
	assert.NoError(t, err)

	// Verify each context has its own isolated data
	contexts := []string{"work", "personal", "research"}
	expectedPrompts := []string{"tasks today", "weather like", "quantum computing"}
	expectedSystems := []string{"work assistant", "personal assistant", "research assistant"}

	for i, contextName := range contexts {
		err = cw.SwitchContext(contextName)
		assert.NoError(t, err)

		recs, err := cw.LiveRecords()
		assert.NoError(t, err)
		assert.GreaterOrEqual(t, len(recs), 3) // system + prompt + response

		// Find system prompt, user prompt, and response
		foundSystem := false
		foundPrompt := false
		foundResponse := false

		for _, rec := range recs {
			switch rec.Source {
			case SystemPrompt:
				assert.Contains(t, rec.Content, expectedSystems[i])
				foundSystem = true
			case Prompt:
				assert.Contains(t, rec.Content, expectedPrompts[i])
				foundPrompt = true
			case ModelResp:
				foundResponse = true
			}
		}

		assert.True(t, foundSystem, "Context %s should have system prompt", contextName)
		assert.True(t, foundPrompt, "Context %s should have user prompt", contextName)
		assert.True(t, foundResponse, "Context %s should have response", contextName)
	}
}

func TestGetContextStats(t *testing.T) {
	db, err := NewContextDB(":memory:")
	assert.NoError(t, err)
	defer db.Close()

	model := &MockModel{}
	cw, err := NewContextWindow(db, model, "test-context")
	assert.NoError(t, err)

	// Test stats for empty context
	ctx, err := cw.GetContext("test-context")
	assert.NoError(t, err)

	stats, err := cw.GetContextStats(ctx)
	assert.NoError(t, err)
	assert.Equal(t, 0, stats.LiveTokens)
	assert.Equal(t, 0, stats.TotalRecords)
	assert.Equal(t, 0, stats.LiveRecords)
	assert.Nil(t, stats.LastActivity)

	// Add some records
	err = cw.SetSystemPrompt("You are a helpful assistant")
	assert.NoError(t, err)

	err = cw.AddPrompt("Hello")
	assert.NoError(t, err)

	err = cw.AddPrompt("How are you?")
	assert.NoError(t, err)

	// Test stats with records
	stats, err = cw.GetContextStats(ctx)
	assert.NoError(t, err)
	assert.Greater(t, stats.LiveTokens, 0)
	assert.Equal(t, 3, stats.TotalRecords) // system prompt + 2 user prompts
	assert.Equal(t, 3, stats.LiveRecords)
	assert.NotNil(t, stats.LastActivity)

	// Create another context to ensure stats are isolated
	err = cw.CreateContext("other-context")
	assert.NoError(t, err)

	otherCtx, err := cw.GetContext("other-context")
	assert.NoError(t, err)

	otherStats, err := cw.GetContextStats(otherCtx)
	assert.NoError(t, err)
	assert.Equal(t, 0, otherStats.LiveTokens)
	assert.Equal(t, 0, otherStats.TotalRecords)
	assert.Equal(t, 0, otherStats.LiveRecords)
	assert.Nil(t, otherStats.LastActivity)

	// Verify original context stats unchanged
	stats, err = cw.GetContextStats(ctx)
	assert.NoError(t, err)
	assert.Greater(t, stats.LiveTokens, 0)
	assert.Equal(t, 3, stats.TotalRecords)
	assert.Equal(t, 3, stats.LiveRecords)
}

func TestGetContextStatsWithNonLiveRecords(t *testing.T) {
	db, err := NewContextDB(":memory:")
	assert.NoError(t, err)
	defer db.Close()

	model := &MockModel{}
	cw, err := NewContextWindow(db, model, "test-context")
	assert.NoError(t, err)

	// Add records and then mark some as non-live by adding a new system prompt
	err = cw.SetSystemPrompt("First system prompt")
	assert.NoError(t, err)

	err = cw.AddPrompt("Hello")
	assert.NoError(t, err)

	// Replace system prompt - this marks old one as non-live
	err = cw.SetSystemPrompt("Second system prompt")
	assert.NoError(t, err)

	err = cw.AddPrompt("How are you?")
	assert.NoError(t, err)

	ctx, err := cw.GetContext("test-context")
	assert.NoError(t, err)

	stats, err := cw.GetContextStats(ctx)
	assert.NoError(t, err)

	// Should have 4 total records (2 system prompts + 2 user prompts)
	// but only 3 live records (current system prompt + 2 user prompts)
	assert.Equal(t, 4, stats.TotalRecords)
	assert.Equal(t, 3, stats.LiveRecords)
	assert.Greater(t, stats.LiveTokens, 0)
	assert.NotNil(t, stats.LastActivity)

	// Live tokens should be less than if all records were live
	recs, err := cw.LiveRecords()
	assert.NoError(t, err)
	expectedLiveTokens := 0
	for _, rec := range recs {
		expectedLiveTokens += rec.EstTokens
	}
	assert.Equal(t, expectedLiveTokens, stats.LiveTokens)
}

func TestContextStatsForTableView(t *testing.T) {
	// Simulate the use case: building a table view of contexts with their stats
	db, err := NewContextDB(":memory:")
	assert.NoError(t, err)
	defer db.Close()

	model := &MockModel{}
	cw, err := NewContextWindow(db, model, "chat-1")
	assert.NoError(t, err)

	// Create multiple contexts with different amounts of content
	contexts := []struct {
		name    string
		prompts []string
	}{
		{"chat-1", []string{"Hello", "How are you?", "Tell me about Go"}},
		{"chat-2", []string{"What's the weather?"}},
		{"empty-chat", []string{}},
	}

	for _, c := range contexts {
		if c.name != "chat-1" {
			err = cw.CreateContext(c.name)
			assert.NoError(t, err)
			err = cw.SwitchContext(c.name)
			assert.NoError(t, err)
		}

		if len(c.prompts) > 0 {
			err = cw.SetSystemPrompt("You are a helpful assistant")
			assert.NoError(t, err)
			for _, prompt := range c.prompts {
				err = cw.AddPrompt(prompt)
				assert.NoError(t, err)
			}
		}
	}

	// Now simulate getting stats for table view - this is the main use case
	allContexts, err := cw.ListContexts()
	assert.NoError(t, err)
	assert.Len(t, allContexts, 3)

	type ContextRow struct {
		Name         string
		Created      string
		LiveTokens   int
		TotalRecords int
		LiveRecords  int
		LastActivity string
	}

	var tableRows []ContextRow
	for _, ctx := range allContexts {
		stats, err := cw.GetContextStats(ctx)
		assert.NoError(t, err)

		lastActivity := "never"
		if stats.LastActivity != nil {
			lastActivity = stats.LastActivity.Format("2006-01-02 15:04")
		}

		row := ContextRow{
			Name:         ctx.Name,
			Created:      ctx.StartTime.Format("2006-01-02 15:04"),
			LiveTokens:   stats.LiveTokens,
			TotalRecords: stats.TotalRecords,
			LiveRecords:  stats.LiveRecords,
			LastActivity: lastActivity,
		}
		tableRows = append(tableRows, row)
	}

	// Verify the table data looks reasonable
	assert.Len(t, tableRows, 3)

	// Find each context and verify its stats
	for _, row := range tableRows {
		switch row.Name {
		case "chat-1":
			assert.Greater(t, row.LiveTokens, 0, "chat-1 should have tokens")
			assert.Equal(t, 4, row.TotalRecords, "chat-1: system + 3 prompts")
			assert.Equal(t, 4, row.LiveRecords, "chat-1: all records live")
			assert.NotEqual(t, "never", row.LastActivity, "chat-1 should have activity")
		case "chat-2":
			assert.Greater(t, row.LiveTokens, 0, "chat-2 should have tokens")
			assert.Equal(t, 2, row.TotalRecords, "chat-2: system + 1 prompt")
			assert.Equal(t, 2, row.LiveRecords, "chat-2: all records live")
			assert.NotEqual(t, "never", row.LastActivity, "chat-2 should have activity")
		case "empty-chat":
			assert.Equal(t, 0, row.LiveTokens, "empty-chat should have no tokens")
			assert.Equal(t, 0, row.TotalRecords, "empty-chat should have no records")
			assert.Equal(t, 0, row.LiveRecords, "empty-chat should have no live records")
			assert.Equal(t, "never", row.LastActivity, "empty-chat should have no activity")
		}
	}

	// Verify we can get stats efficiently for any context without switching
	currentContext := cw.GetCurrentContext()
	for _, ctx := range allContexts {
		_, err := cw.GetContextStats(ctx)
		assert.NoError(t, err, "Should be able to get stats for %s", ctx.Name)
	}
	// Current context should be unchanged
	assert.Equal(t, currentContext, cw.GetCurrentContext())
}

// TestGetOrCreateBehavior tests that all context functions consistently support get-or-create
func TestGetOrCreateBehavior(t *testing.T) {
	db, err := NewContextDB(":memory:")
	assert.NoError(t, err)
	defer db.Close()

	model := &MockModel{}

	// Test CreateContextWithThreading get-or-create behavior
	t.Run("CreateContextWithThreading", func(t *testing.T) {
		// First call should create the context
		ctx1, err := CreateContextWithThreading(db, "test-create", true)
		assert.NoError(t, err)
		assert.Equal(t, "test-create", ctx1.Name)
		assert.Equal(t, true, ctx1.UseServerSideThreading)

		// Second call should return existing context
		ctx2, err := CreateContextWithThreading(db, "test-create", true)
		assert.NoError(t, err)
		assert.Equal(t, ctx1.ID, ctx2.ID)
		assert.Equal(t, ctx1.Name, ctx2.Name)

		// Third call with different threading should update and return existing
		ctx3, err := CreateContextWithThreading(db, "test-create", false)
		assert.NoError(t, err)
		assert.Equal(t, ctx1.ID, ctx3.ID)
		assert.Equal(t, false, ctx3.UseServerSideThreading)
	})

	// Test SwitchContext create-if-not-exists behavior
	t.Run("SwitchContext", func(t *testing.T) {
		cw, err := NewContextWindow(db, model, "initial-context")
		assert.NoError(t, err)

		// Switch to non-existent context should create it
		err = cw.SwitchContext("new-context")
		assert.NoError(t, err)
		assert.Equal(t, "new-context", cw.GetCurrentContext())

		// Verify the context was actually created
		ctx, err := GetContextByName(db, "new-context")
		assert.NoError(t, err)
		assert.Equal(t, "new-context", ctx.Name)

		// Switch to existing context should work normally
		err = cw.SwitchContext("new-context")
		assert.NoError(t, err)
		assert.Equal(t, "new-context", cw.GetCurrentContext())
	})

	// Test NewContextWindowWithThreading get-or-create behavior
	t.Run("NewContextWindowWithThreading", func(t *testing.T) {
		// Create context with threading
		cw1, err := NewContextWindowWithThreading(db, model, "threading-test", true)
		assert.NoError(t, err)

		// Create another instance with same name - should use existing context
		cw2, err := NewContextWindowWithThreading(db, model, "threading-test", false)
		assert.NoError(t, err)

		// Both should refer to the same context
		assert.Equal(t, cw1.GetCurrentContext(), cw2.GetCurrentContext())

		// Verify context exists in database
		ctx, err := GetContextByName(db, "threading-test")
		assert.NoError(t, err)
		assert.Equal(t, "threading-test", ctx.Name)
	})
}

// TestContextContinuation tests resuming contexts with existing messages
func TestContextContinuation(t *testing.T) {
	db, err := NewContextDB(":memory:")
	assert.NoError(t, err)
	defer db.Close()

	model := &MockModel{}

	// Create initial context and add some messages
	cw1, err := NewContextWindow(db, model, "continuation-test")
	assert.NoError(t, err)

	err = cw1.AddPrompt("First message")
	assert.NoError(t, err)

	err = cw1.AddPrompt("Second message")
	assert.NoError(t, err)

	// "Close" the context by creating a new instance
	cw2, err := NewContextWindow(db, model, "continuation-test")
	assert.NoError(t, err)

	// Verify it loads the existing context
	assert.Equal(t, "continuation-test", cw2.GetCurrentContext())

	// Add a new message and verify all history is available
	err = cw2.AddPrompt("Third message")
	assert.NoError(t, err)

	// Get context ID to check records
	contextID, err := getContextIDByName(db, "continuation-test")
	assert.NoError(t, err)

	// Verify all messages are present
	records, err := ListLiveRecords(db, contextID)
	assert.NoError(t, err)

	prompts := []string{}
	for _, record := range records {
		if record.Source == Prompt {
			prompts = append(prompts, record.Content)
		}
	}

	assert.Contains(t, prompts, "First message")
	assert.Contains(t, prompts, "Second message")
	assert.Contains(t, prompts, "Third message")
	assert.Equal(t, 3, len(prompts))
}

// TestThreadingBehaviorResume tests threading behavior when resuming contexts
func TestThreadingBehaviorResume(t *testing.T) {
	db, err := NewContextDB(":memory:")
	assert.NoError(t, err)
	defer db.Close()

	// Use a mock model that tracks threading calls
	threadingModel := &MockThreadingModel{}

	t.Run("ResumeWithServerSideThreading", func(t *testing.T) {
		// Create context with server-side threading
		cw1, err := NewContextWindowWithThreading(db, threadingModel, "threading-resume", true)
		assert.NoError(t, err)

		// Add initial prompt and get response
		err = cw1.AddPrompt("Initial prompt")
		assert.NoError(t, err)

		_, err = cw1.CallModel(context.Background())
		assert.NoError(t, err)

		// "Close" and reopen context
		cw2, err := NewContextWindowWithThreading(db, threadingModel, "threading-resume", true)
		assert.NoError(t, err)

		// Add another prompt
		err = cw2.AddPrompt("Follow-up prompt")
		assert.NoError(t, err)

		// Verify the model receives the call appropriately
		_, err = cw2.CallModel(context.Background())
		assert.NoError(t, err)

		// Check that the context was properly continued
		ctx, err := GetContextByName(db, "threading-resume")
		assert.NoError(t, err)
		assert.Equal(t, true, ctx.UseServerSideThreading)
	})

	t.Run("ResumeWithClientSideThreading", func(t *testing.T) {
		// Create context with client-side threading
		cw1, err := NewContextWindowWithThreading(db, threadingModel, "client-resume", false)
		assert.NoError(t, err)

		// Add messages
		err = cw1.AddPrompt("First client prompt")
		assert.NoError(t, err)

		_, err = cw1.CallModel(context.Background())
		assert.NoError(t, err)

		// Reopen context
		cw2, err := NewContextWindowWithThreading(db, threadingModel, "client-resume", false)
		assert.NoError(t, err)

		err = cw2.AddPrompt("Second client prompt")
		assert.NoError(t, err)

		_, err = cw2.CallModel(context.Background())
		assert.NoError(t, err)

		// Verify context settings
		ctx, err := GetContextByName(db, "client-resume")
		assert.NoError(t, err)
		assert.Equal(t, false, ctx.UseServerSideThreading)
	})
}

// MockThreadingModel implements both interfaces for testing threading behavior
type MockThreadingModel struct {
	callCount int
	lastInputs []Record
	lastServerSide bool
	lastResponseID *string
}

func (m *MockThreadingModel) Call(ctx context.Context, inputs []Record) ([]Record, int, error) {
	m.callCount++
	m.lastInputs = inputs
	m.lastServerSide = false
	m.lastResponseID = nil

	return []Record{
		{
			Source:  ModelResp,
			Content: fmt.Sprintf("Mock response %d", m.callCount),
		},
	}, 10, nil
}

func (m *MockThreadingModel) CallWithThreading(
	ctx context.Context,
	useServerSideThreading bool,
	lastResponseID *string,
	inputs []Record,
) ([]Record, *string, int, error) {
	m.callCount++
	m.lastInputs = inputs
	m.lastServerSide = useServerSideThreading
	m.lastResponseID = lastResponseID

	responseID := fmt.Sprintf("resp-%d", m.callCount)
	return []Record{
		{
			Source:     ModelResp,
			Content:    fmt.Sprintf("Mock threaded response %d", m.callCount),
			ResponseID: &responseID,
		},
	}, &responseID, 10, nil
}
