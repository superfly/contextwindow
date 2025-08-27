// Package contextwindow implements sqlite-backed context windows for LLM sessions,
// with opt-in LLM summarization for compaction, token count tracking, and a simple
// tool call abstraction.
//
// A context window is just a list of strings, representing the history of a
// "conversation" with an LLM. The "live" strings in a conversation will be
// fed to the LLM on every call; we retain the "unalive" strings in records
// so that tool calls can fetch them later.
package contextwindow

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/google/uuid"

	"github.com/peterheb/gotoken"
	_ "github.com/peterheb/gotoken/cl100kbase"
	_ "modernc.org/sqlite"
)

// Model abstracts out an LLM client library.
type Model interface {
	// Call sends messages, returns model reply and token usage.
	Call(ctx context.Context, inputs []Record) (events []Record, tokensUsed int, err error)
}

// ServerSideThreadingCapable is an optional interface for models that support server-side threading.
type ServerSideThreadingCapable interface {
	// CallWithThreading calls the model with optional server-side threading.
	CallWithThreading(
		ctx context.Context,
		useServerSideThreading bool,
		lastResponseID *string,
		inputs []Record,
	) (events []Record, responseID *string, tokensUsed int, err error)
}

// ToolCapable is an optional interface that models can implement
// to receive a ToolExecutor for handling tool calls.
type ToolCapable interface {
	SetToolExecutor(ToolExecutor)
}

// MiddlewareCapable is an optional interface that models can implement
// to receive middleware updates.
type MiddlewareCapable interface {
	SetMiddleware([]Middleware)
}

// CallOptsCapable is an optional interface that models can implement
// to support call options like disabling tools.
type CallOptsCapable interface {
	CallWithOpts(ctx context.Context, inputs []Record, opts CallModelOpts) (events []Record, tokensUsed int, err error)
	CallWithThreadingAndOpts(
		ctx context.Context,
		useServerSideThreading bool,
		lastResponseID *string,
		inputs []Record,
		opts CallModelOpts,
	) (events []Record, responseID *string, tokensUsed int, err error)
}

// Middleware allows hooking into tool call lifecycle events.
type Middleware interface {
	// OnToolCall is invoked when a tool is about to be called.
	OnToolCall(ctx context.Context, name, args string)
	// OnToolResult is invoked when a tool call completes.
	OnToolResult(ctx context.Context, name, result string, err error)
}

// ContextWindow holds our LLM context manager state.
type ContextWindow struct {
	model            Model
	db               *sql.DB
	maxTokens        int
	summarizer       Summarizer
	summarizerPrompt string
	middleware       []Middleware
	metrics          *Metrics
	currentContext   string
	registeredTools  map[string]ToolDefinition
	toolRunners      map[string]ToolRunner
}

// NewContextDB opens a database to store context windows in (pass
// ":memory:" as the name for a transient database), and migrates
// its schema. You'll need to do this before creating context windows.
func NewContextDB(dbpath string) (*sql.DB, error) {
	db, err := sql.Open("sqlite", dbpath)
	if err != nil {
		return nil, fmt.Errorf("open db: %w", err)
	}
	if err = InitializeSchema(db); err != nil {
		db.Close()
		return nil, fmt.Errorf("init schema: %w", err)
	}

	return db, nil
}

// NewContextWindow initializes a ContextWindow with an existing database
// connection and a specific context. If the context doesn't exist, it will be created.
// The caller is responsible for closing the database. You can provide an
// empty context name, in which case we'll generate a UUID for it.
func NewContextWindow(
	db *sql.DB,
	model Model,
	contextName string,
) (*ContextWindow, error) {
	return NewContextWindowWithThreading(db, model, contextName, false)
}

// NewContextWindowWithThreading creates a context window with specified threading mode.
func NewContextWindowWithThreading(
	db *sql.DB,
	model Model,
	contextName string,
	useServerSideThreading bool,
) (*ContextWindow, error) {
	if contextName == "" {
		contextName = uuid.New().String()
	}

	cw := &ContextWindow{
		model:           model,
		db:              db,
		maxTokens:       4096,
		metrics:         &Metrics{},
		currentContext:  contextName,
		registeredTools: make(map[string]ToolDefinition),
		toolRunners:     make(map[string]ToolRunner),
	}

	// If the model supports tool execution, configure it
	if toolCapable, ok := model.(ToolCapable); ok {
		toolCapable.SetToolExecutor(cw)
	}

	_, err := GetContextByName(db, contextName)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			_, err = CreateContextWithThreading(db, contextName, useServerSideThreading)
			if err != nil {
				return nil, fmt.Errorf("create context: %w", err)
			}
		} else {
			return nil, fmt.Errorf("get context: %w", err)
		}
	}

	return cw, nil
}

// Close closes the database connection. Only call this if you opened the database
// using NewContextWindow or NewContextWindowWithContext. If you used
// NewContextWindowWithDB, you should close the database yourself.
func (cw *ContextWindow) Close() error {
	if cw.db == nil {
		return nil
	}
	return cw.db.Close()
}

// AddPrompt logs a user prompt to the current context.
func (cw *ContextWindow) AddPrompt(text string) error {
	contextID, err := getContextIDByName(cw.db, cw.currentContext)
	if err != nil {
		return fmt.Errorf("add prompt: %w", err)
	}
	_, err = InsertRecord(cw.db, contextID, Prompt, text, true)
	if err != nil {
		return fmt.Errorf("add prompt: %w", err)
	}
	return nil
}

// AddToolCall logs a tool invocation to the current context.
func (cw *ContextWindow) AddToolCall(name, args string) error {
	contextID, err := getContextIDByName(cw.db, cw.currentContext)
	if err != nil {
		return fmt.Errorf("add tool call: %w", err)
	}
	content := fmt.Sprintf("%s(%s)", name, args)
	_, err = InsertRecord(cw.db, contextID, ToolCall, content, true)
	if err != nil {
		return fmt.Errorf("add tool call: %w", err)
	}
	return nil
}

// AddToolOutput logs a tool's output to the current context.
func (cw *ContextWindow) AddToolOutput(output string) error {
	contextID, err := getContextIDByName(cw.db, cw.currentContext)
	if err != nil {
		return fmt.Errorf("add tool output: %w", err)
	}
	_, err = InsertRecord(cw.db, contextID, ToolOutput, output, true)
	if err != nil {
		return fmt.Errorf("add tool output: %w", err)
	}
	return nil
}

// SetSystemPrompt sets the system prompt for the current context.
func (cw *ContextWindow) SetSystemPrompt(text string) error {
	contextID, err := getContextIDByName(cw.db, cw.currentContext)
	if err != nil {
		return fmt.Errorf("set system prompt: %w", err)
	}

	tx, err := cw.db.Begin()
	if err != nil {
		return fmt.Errorf("set system prompt: %w", err)
	}
	defer tx.Rollback()

	_, err = tx.Exec(`UPDATE records SET live = 0 WHERE context_id = ? AND source = ?`, contextID, SystemPrompt)
	if err != nil {
		return fmt.Errorf("set system prompt: %w", err)
	}

	_, err = insertRecordTx(tx, contextID, SystemPrompt, text, true)
	if err != nil {
		return fmt.Errorf("set system prompt: %w", err)
	}

	return tx.Commit()
}

// AddMiddleware registers middleware to hook into tool call events.
func (cw *ContextWindow) AddMiddleware(m Middleware) {
	cw.middleware = append(cw.middleware, m)
	// If the model supports middleware, update it
	if middlewareCapable, ok := cw.model.(MiddlewareCapable); ok {
		middlewareCapable.SetMiddleware(cw.middleware)
	}
}

// LiveRecords retrieves all "live" records from the context. This is an
// important function, since this is usually what you want to call to get
// what's currently meaningful in your context --- it's what gets sent
// to the LLM.
func (cw *ContextWindow) LiveRecords() ([]Record, error) {
	contextID, err := getContextIDByName(cw.db, cw.currentContext)
	if err != nil {
		return nil, fmt.Errorf("live records: %w", err)
	}
	recs, err := ListLiveRecords(cw.db, contextID)
	if err != nil {
		return nil, fmt.Errorf("live records: %w", err)
	}
	return recs, nil
}

// CallModelOpts contains options for model calls.
type CallModelOpts struct {
	DisableTools bool
}

// CallModel drives an LLM. It composes live messages, invokes cw.model.Call,
// logs the response, updates token count, and triggers compaction.
func (cw *ContextWindow) CallModel(ctx context.Context) (string, error) {
	return cw.CallModelWithOpts(ctx, CallModelOpts{})
}

// CallModelWithOpts drives an LLM with options. It composes live messages, invokes cw.model.Call,
// logs the response, updates token count, and triggers compaction.
func (cw *ContextWindow) CallModelWithOpts(ctx context.Context, opts CallModelOpts) (string, error) {
	contextID, err := getContextIDByName(cw.db, cw.currentContext)
	if err != nil {
		return "", fmt.Errorf("call model in context: %w", err)
	}

	// Get current context info to check threading mode
	contextInfo, err := GetContext(cw.db, contextID)
	if err != nil {
		return "", fmt.Errorf("get context info: %w", err)
	}

	recs, err := ListLiveRecords(cw.db, contextID)
	if err != nil {
		return "", fmt.Errorf("list live records: %w", err)
	}

	var events []Record
	var tokensUsed int
	var responseID *string

	// Use server-side threading if supported and enabled
	if contextInfo.UseServerSideThreading {
		if threadingModel, ok := cw.model.(ServerSideThreadingCapable); ok {
			if optsModel, ok := threadingModel.(CallOptsCapable); ok {
				events, responseID, tokensUsed, err = optsModel.CallWithThreadingAndOpts(
					ctx,
					true,
					contextInfo.LastResponseID,
					recs,
					opts,
				)
			} else {
				events, responseID, tokensUsed, err = threadingModel.CallWithThreading(
					ctx,
					true,
					contextInfo.LastResponseID,
					recs,
				)
			}
			if err != nil {
				return "", fmt.Errorf("call model with threading: %w", err)
			}
		} else {
			return "", fmt.Errorf("model does not support server-side threading")
		}
	} else {
		// Fall back to traditional client-side threading
		if optsModel, ok := cw.model.(CallOptsCapable); ok {
			events, tokensUsed, err = optsModel.CallWithOpts(ctx, recs, opts)
		} else {
			events, tokensUsed, err = cw.model.Call(ctx, recs)
		}
		if err != nil {
			return "", fmt.Errorf("call model: %w", err)
		}
	}

	cw.metrics.Add(tokensUsed)
	var lastMsg string
	for _, event := range events {
		_, err = InsertRecordWithResponseID(
			cw.db,
			contextID,
			event.Source,
			event.Content,
			event.Live,
			event.ResponseID,
		)
		if err != nil {
			return "", fmt.Errorf("insert model response: %w", err)
		}
		lastMsg = event.Content
	}

	// Update the context's last response ID if we got one
	if responseID != nil {
		err = UpdateContextLastResponseID(cw.db, contextID, *responseID)
		if err != nil {
			return lastMsg, fmt.Errorf("update last response ID: %w", err)
		}
	}

	return lastMsg, nil
}

func (cw *ContextWindow) TotalTokens() int {
	return cw.metrics.Total()
}

// LiveTokens estimates the total number of tokens in all "live" messages
// in the context. Depending on your model, at some threshold of tokens
// you'll want either to summarize, or to start a new context window.
func (cw *ContextWindow) LiveTokens() (int, error) {
	contextID, err := getContextIDByName(cw.db, cw.currentContext)
	if err != nil {
		return 0, fmt.Errorf("live tokens in context: %w", err)
	}
	recs, err := ListLiveRecords(cw.db, contextID)
	if err != nil {
		return 0, fmt.Errorf("list live records: %w", err)
	}
	var n int
	for _, r := range recs {
		n += r.EstTokens
	}
	return n, nil
}

// Metrics tracks token usage across model calls.
type Metrics struct {
	mu    sync.Mutex
	total int
}

func (m *Metrics) Add(n int) {
	m.mu.Lock()
	m.total += n
	m.mu.Unlock()
}

func (m *Metrics) Total() int {
	m.mu.Lock()
	n := m.total
	m.mu.Unlock()
	return n
}

// TokenUsage provides a snapshot of current token usage for UI display.
type TokenUsage struct {
	Live    int     // tokens currently in context window
	Total   int     // cumulative tokens used across all calls
	Max     int     // maximum tokens allowed in context window
	Percent float64 // live/max as percentage (0.0-1.0)
}

// TokenUsage returns current token usage metrics optimized for UI display.
func (cw *ContextWindow) TokenUsage() (TokenUsage, error) {
	live, err := cw.LiveTokens()
	if err != nil {
		return TokenUsage{}, err
	}

	percent := 0.0
	if cw.maxTokens > 0 {
		percent = float64(live) / float64(cw.maxTokens)
	}

	return TokenUsage{
		Live:    live,
		Total:   cw.metrics.Total(),
		Max:     cw.maxTokens,
		Percent: percent,
	}, nil
}

type TokenReporter interface {
	TotalTokens() int
	LiveTokens() (int, error)
	TokenUsage() (TokenUsage, error)
}

func tokenCount(s string) int {
	tokOnce.Do(func() {
		tok, tokErr = gotoken.GetTokenizer("cl100k_base")
	})
	if tokErr != nil {
		return len(strings.Fields(s))
	}
	return tok.Count(s)
}

var (
	tok     gotoken.Tokenizer
	tokOnce sync.Once
	tokErr  error
)

// Context management methods

// CreateContext creates a new named context window.
func (cw *ContextWindow) CreateContext(name string) error {
	_, err := CreateContext(cw.db, name)
	if err != nil {
		return fmt.Errorf("create context: %w", err)
	}
	return nil
}

// ListContexts returns all available context windows.
func (cw *ContextWindow) ListContexts() ([]Context, error) {
	contexts, err := ListContexts(cw.db)
	if err != nil {
		return nil, fmt.Errorf("list contexts: %w", err)
	}
	return contexts, nil
}

// GetContext retrieves context metadata by name.
func (cw *ContextWindow) GetContext(name string) (Context, error) {
	ctx, err := GetContextByName(cw.db, name)
	if err != nil {
		return Context{}, fmt.Errorf("get context: %w", err)
	}
	return ctx, nil
}

// DeleteContext removes a context and all its records.
func (cw *ContextWindow) DeleteContext(name string) error {
	if name == cw.currentContext {
		contexts, err := ListContexts(cw.db)
		if err != nil {
			return fmt.Errorf("list contexts for deletion: %w", err)
		}
		if len(contexts) <= 1 {
			_, err := CreateContext(cw.db, "default")
			if err != nil {
				return fmt.Errorf("create replacement context: %w", err)
			}
			cw.currentContext = "default"
		} else {
			for _, ctx := range contexts {
				if ctx.Name != name {
					cw.currentContext = ctx.Name
					break
				}
			}
		}
	}

	err := DeleteContextByName(cw.db, name)
	if err != nil {
		return fmt.Errorf("delete context: %w", err)
	}
	return nil
}

// ExportContext extracts a complete context with all its records.
func (cw *ContextWindow) ExportContext(name string) (ContextExport, error) {
	export, err := ExportContextByName(cw.db, name)
	if err != nil {
		return ContextExport{}, fmt.Errorf("export context: %w", err)
	}
	return export, nil
}

// ExportContextJSON exports a context as JSON bytes.
func (cw *ContextWindow) ExportContextJSON(name string) ([]byte, error) {
	jsonData, err := ExportContextJSONByName(cw.db, name)
	if err != nil {
		return nil, fmt.Errorf("export context json: %w", err)
	}
	return jsonData, nil
}

func (cw *ContextWindow) GetCurrentContext() string {
	return cw.currentContext
}

// GetCurrentContextInfo returns the current context metadata.
func (cw *ContextWindow) GetCurrentContextInfo() (Context, error) {
	return cw.GetContext(cw.currentContext)
}

// MaxTokens returns the maximum number of tokens allowed in the context window.
func (cw *ContextWindow) MaxTokens() int {
	return cw.maxTokens
}

// SetMaxTokens updates the maximum number of tokens allowed in the context window.
func (cw *ContextWindow) SetMaxTokens(max int) {
	cw.maxTokens = max
}

// SetServerSideThreading enables or disables server-side threading for the current context.
func (cw *ContextWindow) SetServerSideThreading(enabled bool) error {
	contextID, err := getContextIDByName(cw.db, cw.currentContext)
	if err != nil {
		return fmt.Errorf("get context ID: %w", err)
	}
	return SetContextServerSideThreading(cw.db, contextID, enabled)
}

// IsServerSideThreadingEnabled returns whether server-side threading is enabled.
func (cw *ContextWindow) IsServerSideThreadingEnabled() (bool, error) {
	contextInfo, err := cw.GetCurrentContextInfo()
	if err != nil {
		return false, err
	}
	return contextInfo.UseServerSideThreading, nil
}
