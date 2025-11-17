package contextwindow

import (
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"
)

// RecordType distinguishes entry kinds.
type RecordType int

const (
	Prompt RecordType = iota
	ModelResp
	ToolCall
	ToolOutput
	SystemPrompt
)

// Record is one row in context history.
type Record struct {
	ID                int64      `json:"id"`
	Timestamp         time.Time  `json:"timestamp"`
	Source            RecordType `json:"source"`
	Content           string     `json:"content"`
	Live              bool       `json:"live"`
	EstTokens         int        `json:"est_tokens"`
	ContextID         string     `json:"context_id"`
	ResponseID        *string    `json:"response_id,omitempty"`
	Streamed          bool       `json:"streamed"`
	PartialResponseID *string    `json:"partial_response_id,omitempty"`
	AccumulatedTokens *int       `json:"accumulated_tokens,omitempty"`
}

// Context represents a named context window with metadata.
type Context struct {
	ID                     string    `json:"id"`
	Name                   string    `json:"name"`
	StartTime              time.Time `json:"start_time"`
	UseServerSideThreading bool      `json:"use_server_side_threading"`
	LastResponseID         *string   `json:"last_response_id,omitempty"`
}

// ContextTool represents a tool available in a specific context.
type ContextTool struct {
	ID        int64     `json:"id"`
	ContextID string    `json:"context_id"`
	ToolName  string    `json:"tool_name"`
	CreatedAt time.Time `json:"created_at"`
}

// ContextExport represents a complete context with all its records.
type ContextExport struct {
	Context Context       `json:"context"`
	Records []Record      `json:"records"`
	Tools   []ContextTool `json:"tools"`
}

// InitializeSchema ensures the contexts and records tables and indexes exist.
// Also handles migrations by adding new columns to existing tables.
func InitializeSchema(db *sql.DB) error {
	// Create base tables first
	const baseTables = `
CREATE TABLE IF NOT EXISTS contexts (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    start_time DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS records (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    context_id TEXT NOT NULL,
    ts         DATETIME NOT NULL,
    source     INTEGER NOT NULL,
    content    TEXT NOT NULL,
    live       BOOLEAN NOT NULL,
    est_tokens INTEGER NOT NULL,
    FOREIGN KEY (context_id) REFERENCES contexts(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS context_tools (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    created_at DATETIME NOT NULL,
    FOREIGN KEY (context_id) REFERENCES contexts(id) ON DELETE CASCADE,
    UNIQUE(context_id, tool_name)
);
`

	_, err := db.Exec(baseTables)
	if err != nil {
		return fmt.Errorf("create base tables: %w", err)
	}

	// Add new columns if they don't exist (migration)
	err = addColumnIfNotExists(db, "contexts", "use_server_side_threading", "BOOLEAN NOT NULL DEFAULT 0")
	if err != nil {
		return fmt.Errorf("add use_server_side_threading column: %w", err)
	}

	err = addColumnIfNotExists(db, "contexts", "last_response_id", "TEXT NULL")
	if err != nil {
		return fmt.Errorf("add last_response_id column: %w", err)
	}

	err = addColumnIfNotExists(db, "records", "response_id", "TEXT NULL")
	if err != nil {
		return fmt.Errorf("add response_id column: %w", err)
	}

	err = addColumnIfNotExists(db, "records", "streamed", "BOOLEAN NOT NULL DEFAULT 0")
	if err != nil {
		return fmt.Errorf("add streamed column: %w", err)
	}

	err = addColumnIfNotExists(db, "records", "partial_response_id", "TEXT NULL")
	if err != nil {
		return fmt.Errorf("add partial_response_id column: %w", err)
	}

	err = addColumnIfNotExists(db, "records", "accumulated_tokens", "INTEGER NULL")
	if err != nil {
		return fmt.Errorf("add accumulated_tokens column: %w", err)
	}

	// Create indexes
	const indexes = `
CREATE INDEX IF NOT EXISTS idx_context_live ON records(context_id, live);
CREATE INDEX IF NOT EXISTS idx_context_ts ON records(context_id, ts);
CREATE INDEX IF NOT EXISTS idx_context_tools_context ON context_tools(context_id);
`
	_, err = db.Exec(indexes)
	if err != nil {
		return fmt.Errorf("create indexes: %w", err)
	}

	return nil
}

// CreateContext creates a new context with the given name.
// Name must not be empty and must be unique.
func CreateContext(db *sql.DB, name string) (Context, error) {
	return CreateContextWithThreading(db, name, false)
}

// CreateContextWithThreading creates a new context with threading mode specified.
func CreateContextWithThreading(db *sql.DB, name string, useServerSideThreading bool) (Context, error) {
	if name == "" {
		return Context{}, fmt.Errorf("context name cannot be empty")
	}

	// Check if context already exists - if so, return it
	existingContext, err := GetContextByName(db, name)
	if err == nil {
		// Context exists, update threading mode if different
		if existingContext.UseServerSideThreading != useServerSideThreading {
			err = SetContextServerSideThreading(db, existingContext.ID, useServerSideThreading)
			if err != nil {
				return Context{}, fmt.Errorf("update threading mode: %w", err)
			}
			existingContext.UseServerSideThreading = useServerSideThreading
		}
		return existingContext, nil
	}
	if !errors.Is(err, sql.ErrNoRows) {
		return Context{}, fmt.Errorf("check existing context: %w", err)
	}

	id := uuid.New().String()
	now := time.Now().UTC()

	_, err = db.Exec(
		`INSERT INTO contexts (id, name, start_time, use_server_side_threading) VALUES (?, ?, ?, ?)`,
		id, name, now, useServerSideThreading,
	)
	if err != nil {
		return Context{}, fmt.Errorf("create context: %w", err)
	}

	return Context{
		ID:                     id,
		Name:                   name,
		StartTime:              now,
		UseServerSideThreading: useServerSideThreading,
	}, nil
}

// ListContexts returns all contexts ordered by start time.
func ListContexts(db *sql.DB) ([]Context, error) {
	rows, err := db.Query(
		`SELECT id, name, start_time, 
		 COALESCE(use_server_side_threading, 0) as use_server_side_threading,
		 last_response_id 
		 FROM contexts ORDER BY start_time DESC`,
	)
	if err != nil {
		return nil, fmt.Errorf("query contexts: %w", err)
	}
	defer rows.Close()

	var contexts []Context
	for rows.Next() {
		var c Context
		if err := rows.Scan(&c.ID, &c.Name, &c.StartTime, &c.UseServerSideThreading, &c.LastResponseID); err != nil {
			return nil, fmt.Errorf("scan context: %w", err)
		}
		contexts = append(contexts, c)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("contexts rows: %w", err)
	}
	return contexts, nil
}

// GetContext retrieves a context by ID.
func GetContext(db *sql.DB, contextID string) (Context, error) {
	var c Context
	err := db.QueryRow(
		`SELECT id, name, start_time,
		 COALESCE(use_server_side_threading, 0) as use_server_side_threading,
		 last_response_id
		 FROM contexts WHERE id = ?`,
		contextID,
	).Scan(&c.ID, &c.Name, &c.StartTime, &c.UseServerSideThreading, &c.LastResponseID)
	if err != nil {
		return Context{}, fmt.Errorf("get context %s: %w", contextID, err)
	}
	return c, nil
}

// GetContextByName retrieves a context by name.
func GetContextByName(db *sql.DB, name string) (Context, error) {
	var c Context
	err := db.QueryRow(
		`SELECT id, name, start_time,
		 COALESCE(use_server_side_threading, 0) as use_server_side_threading,
		 last_response_id
		 FROM contexts WHERE name = ?`,
		name,
	).Scan(&c.ID, &c.Name, &c.StartTime, &c.UseServerSideThreading, &c.LastResponseID)
	if err != nil {
		return Context{}, fmt.Errorf("get context '%s': %w", name, err)
	}
	return c, nil
}

// DeleteContext removes a context and all its records by ID.
func DeleteContext(db *sql.DB, contextID string) error {
	tx, err := db.Begin()
	if err != nil {
		return fmt.Errorf("begin transaction: %w", err)
	}
	defer tx.Rollback()

	_, err = tx.Exec(`DELETE FROM records WHERE context_id = ?`, contextID)
	if err != nil {
		return fmt.Errorf("delete context records: %w", err)
	}

	_, err = tx.Exec(`DELETE FROM contexts WHERE id = ?`, contextID)
	if err != nil {
		return fmt.Errorf("delete context: %w", err)
	}

	return tx.Commit()
}

// DeleteContextByName removes a context and all its records by name.
func DeleteContextByName(db *sql.DB, name string) error {
	ctx, err := GetContextByName(db, name)
	if err != nil {
		return err
	}
	return DeleteContext(db, ctx.ID)
}

// ExportContext extracts a complete context with all its records by ID.
func ExportContext(db *sql.DB, contextID string) (ContextExport, error) {
	context, err := GetContext(db, contextID)
	if err != nil {
		return ContextExport{}, err
	}

	records, err := ListRecordsInContext(db, contextID)
	if err != nil {
		return ContextExport{}, err
	}

	tools, err := ListContextTools(db, contextID)
	if err != nil {
		return ContextExport{}, err
	}

	return ContextExport{
		Context: context,
		Records: records,
		Tools:   tools,
	}, nil
}

// ExportContextByName extracts a complete context with all its records by name.
func ExportContextByName(db *sql.DB, name string) (ContextExport, error) {
	ctx, err := GetContextByName(db, name)
	if err != nil {
		return ContextExport{}, err
	}
	return ExportContext(db, ctx.ID)
}

// ExportContextJSON exports a context as JSON by ID.
func ExportContextJSON(db *sql.DB, contextID string) ([]byte, error) {
	export, err := ExportContext(db, contextID)
	if err != nil {
		return nil, err
	}
	return json.MarshalIndent(export, "", "  ")
}

// ExportContextJSONByName exports a context as JSON by name.
func ExportContextJSONByName(db *sql.DB, name string) ([]byte, error) {
	export, err := ExportContextByName(db, name)
	if err != nil {
		return nil, err
	}
	return json.MarshalIndent(export, "", "  ")
}

// InsertRecord inserts a new record in the specified context.
func InsertRecord(
	db *sql.DB,
	contextID string,
	source RecordType,
	content string,
	live bool,
) (Record, error) {
	return InsertRecordWithResponseID(db, contextID, source, content, live, nil)
}

// InsertRecordStreamed inserts a new record with streaming flag.
func InsertRecordStreamed(
	db *sql.DB,
	contextID string,
	source RecordType,
	content string,
	live bool,
	streamed bool,
) (Record, error) {
	return InsertRecordWithResponseIDAndStreamed(db, contextID, source, content, live, nil, streamed)
}

// InsertRecordWithResponseID inserts a new record with optional response ID.
func InsertRecordWithResponseID(
	db *sql.DB,
	contextID string,
	source RecordType,
	content string,
	live bool,
	responseID *string,
) (Record, error) {
	return InsertRecordWithResponseIDAndStreamed(db, contextID, source, content, live, responseID, false)
}

// InsertRecordWithResponseIDAndStreamed inserts a new record with optional response ID and streaming flag.
func InsertRecordWithResponseIDAndStreamed(
	db *sql.DB,
	contextID string,
	source RecordType,
	content string,
	live bool,
	responseID *string,
	streamed bool,
) (Record, error) {
	return InsertRecordWithPartialResponse(db, contextID, source, content, live, responseID, streamed, nil, nil)
}

// InsertRecordWithPartialResponse inserts a new record with optional response ID, streaming flag, and partial response tracking.
func InsertRecordWithPartialResponse(
	db *sql.DB,
	contextID string,
	source RecordType,
	content string,
	live bool,
	responseID *string,
	streamed bool,
	partialResponseID *string,
	accumulatedTokens *int,
) (Record, error) {
	now := time.Now().UTC()
	t := tokenCount(content)
	res, err := db.Exec(
		`INSERT INTO records (context_id, ts, source, content, live, est_tokens, response_id, streamed, partial_response_id, accumulated_tokens) 
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		contextID, now, int(source), content, live, t, responseID, streamed, partialResponseID, accumulatedTokens,
	)
	if err != nil {
		return Record{}, fmt.Errorf("insert record: %w", err)
	}
	id, err := res.LastInsertId()
	if err != nil {
		return Record{}, fmt.Errorf("get last insert id: %w", err)
	}
	return Record{
		ID:                id,
		Timestamp:         now,
		Source:            source,
		Content:           content,
		Live:              live,
		EstTokens:         t,
		ContextID:         contextID,
		ResponseID:        responseID,
		Streamed:          streamed,
		PartialResponseID: partialResponseID,
		AccumulatedTokens: accumulatedTokens,
	}, nil
}

// ListLiveRecords returns all live records in a context in timestamp order.
func ListLiveRecords(db *sql.DB, contextID string) ([]Record, error) {
	return listRecordsWhere(db, "context_id = ? AND live = 1", contextID)
}

// ListRecordsInContext returns all records in a context in timestamp order.
func ListRecordsInContext(db *sql.DB, contextID string) ([]Record, error) {
	return listRecordsWhere(db, "context_id = ?", contextID)
}

func listRecordsWhere(db *sql.DB, whereClause string, args ...interface{}) ([]Record, error) {
	query := fmt.Sprintf(
		`SELECT id, context_id, ts, source, content, live, est_tokens, response_id, 
		 COALESCE(streamed, 0) as streamed, partial_response_id, accumulated_tokens
		 FROM records WHERE %s ORDER BY ts ASC`,
		whereClause,
	)
	rows, err := db.Query(query, args...)
	if err != nil {
		return nil, fmt.Errorf("query records: %w", err)
	}
	defer rows.Close()

	var recs []Record
	for rows.Next() {
		var r Record
		var src int
		if err := rows.Scan(
			&r.ID,
			&r.ContextID,
			&r.Timestamp,
			&src,
			&r.Content,
			&r.Live,
			&r.EstTokens,
			&r.ResponseID,
			&r.Streamed,
			&r.PartialResponseID,
			&r.AccumulatedTokens,
		); err != nil {
			return nil, fmt.Errorf("scan record: %w", err)
		}
		r.Source = RecordType(src)
		recs = append(recs, r)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("records rows: %w", err)
	}
	return recs, nil
}

func markRecordNotAlive(tx *sql.Tx, id int64) error {
	_, err := tx.Exec(
		`UPDATE records SET live = 0 WHERE id = ?`,
		id,
	)
	if err != nil {
		return fmt.Errorf("mark record not alive: %w", err)
	}
	return nil
}

func insertRecordTx(
	tx *sql.Tx,
	contextID string,
	source RecordType,
	content string,
	live bool,
) (Record, error) {
	return insertRecordTxWithResponseID(tx, contextID, source, content, live, nil)
}

func insertRecordTxWithResponseID(
	tx *sql.Tx,
	contextID string,
	source RecordType,
	content string,
	live bool,
	responseID *string,
) (Record, error) {
	return insertRecordTxWithResponseIDAndStreamed(tx, contextID, source, content, live, responseID, false)
}

func insertRecordTxWithResponseIDAndStreamed(
	tx *sql.Tx,
	contextID string,
	source RecordType,
	content string,
	live bool,
	responseID *string,
	streamed bool,
) (Record, error) {
	return insertRecordTxWithPartialResponse(tx, contextID, source, content, live, responseID, streamed, nil, nil)
}

func insertRecordTxWithPartialResponse(
	tx *sql.Tx,
	contextID string,
	source RecordType,
	content string,
	live bool,
	responseID *string,
	streamed bool,
	partialResponseID *string,
	accumulatedTokens *int,
) (Record, error) {
	now := time.Now().UTC()
	t := tokenCount(content)
	res, err := tx.Exec(
		`INSERT INTO records (context_id, ts, source, content, live, est_tokens, response_id, streamed, partial_response_id, accumulated_tokens) 
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		contextID, now, int(source), content, live, t, responseID, streamed, partialResponseID, accumulatedTokens,
	)
	if err != nil {
		return Record{}, fmt.Errorf("insert record tx: %w", err)
	}
	id, err := res.LastInsertId()
	if err != nil {
		return Record{}, fmt.Errorf("get last insert id tx: %w", err)
	}
	return Record{
		ID:                id,
		Timestamp:         now,
		Source:            source,
		Content:           content,
		Live:              live,
		EstTokens:         t,
		ContextID:         contextID,
		ResponseID:        responseID,
		Streamed:          streamed,
		PartialResponseID: partialResponseID,
		AccumulatedTokens: accumulatedTokens,
	}, nil
}

// getContextIDByName is a helper to get the internal UUID by context name.
func getContextIDByName(db *sql.DB, name string) (string, error) {
	var id string
	err := db.QueryRow(`SELECT id FROM contexts WHERE name = ?`, name).Scan(&id)
	if err != nil {
		return "", fmt.Errorf("get context ID for '%s': %w", name, err)
	}
	return id, nil
}

// AddContextTool adds a tool name to a specific context.
func AddContextTool(db *sql.DB, contextID, toolName string) (ContextTool, error) {
	now := time.Now().UTC()
	res, err := db.Exec(
		`INSERT INTO context_tools (context_id, tool_name, created_at)
		 VALUES (?, ?, ?)`,
		contextID, toolName, now,
	)
	if err != nil {
		return ContextTool{}, fmt.Errorf("add context tool: %w", err)
	}
	id, err := res.LastInsertId()
	if err != nil {
		return ContextTool{}, fmt.Errorf("get last insert id: %w", err)
	}
	return ContextTool{
		ID:        id,
		ContextID: contextID,
		ToolName:  toolName,
		CreatedAt: now,
	}, nil
}

// ListContextTools returns all tools for a specific context.
func ListContextTools(db *sql.DB, contextID string) ([]ContextTool, error) {
	rows, err := db.Query(
		`SELECT id, context_id, tool_name, created_at 
		 FROM context_tools WHERE context_id = ? ORDER BY created_at ASC`,
		contextID,
	)
	if err != nil {
		return nil, fmt.Errorf("query context tools: %w", err)
	}
	defer rows.Close()

	var tools []ContextTool
	for rows.Next() {
		var t ContextTool
		if err := rows.Scan(&t.ID, &t.ContextID, &t.ToolName, &t.CreatedAt); err != nil {
			return nil, fmt.Errorf("scan context tool: %w", err)
		}
		tools = append(tools, t)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("context tools rows: %w", err)
	}
	return tools, nil
}

// ListContextToolNames returns just the tool names for a specific context.
func ListContextToolNames(db *sql.DB, contextID string) ([]string, error) {
	rows, err := db.Query(
		`SELECT tool_name FROM context_tools WHERE context_id = ? ORDER BY created_at ASC`,
		contextID,
	)
	if err != nil {
		return nil, fmt.Errorf("query context tool names: %w", err)
	}
	defer rows.Close()

	var names []string
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			return nil, fmt.Errorf("scan tool name: %w", err)
		}
		names = append(names, name)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("tool names rows: %w", err)
	}
	return names, nil
}

// RemoveContextTool removes a tool from a specific context.
func RemoveContextTool(db *sql.DB, contextID, toolName string) error {
	_, err := db.Exec(
		`DELETE FROM context_tools WHERE context_id = ? AND tool_name = ?`,
		contextID, toolName,
	)
	if err != nil {
		return fmt.Errorf("remove context tool: %w", err)
	}
	return nil
}

// HasContextTool checks if a specific tool is available in a context.
func HasContextTool(db *sql.DB, contextID, toolName string) (bool, error) {
	var exists bool
	err := db.QueryRow(
		`SELECT 1 FROM context_tools WHERE context_id = ? AND tool_name = ?`,
		contextID, toolName,
	).Scan(&exists)
	if err != nil && err != sql.ErrNoRows {
		return false, fmt.Errorf("check context tool: %w", err)
	}
	return exists, nil
}

// UpdateContextLastResponseID updates the last response ID for a context.
func UpdateContextLastResponseID(db *sql.DB, contextID, responseID string) error {
	_, err := db.Exec(
		`UPDATE contexts SET last_response_id = ? WHERE id = ?`,
		responseID, contextID,
	)
	if err != nil {
		return fmt.Errorf("update context last response ID: %w", err)
	}
	return nil
}

// SetContextServerSideThreading enables or disables server-side threading for a context.
func SetContextServerSideThreading(db *sql.DB, contextID string, useServerSideThreading bool) error {
	_, err := db.Exec(
		`UPDATE contexts SET use_server_side_threading = ? WHERE id = ?`,
		useServerSideThreading, contextID,
	)
	if err != nil {
		return fmt.Errorf("set context server side threading: %w", err)
	}
	return nil
}

// addColumnIfNotExists adds a column to a table if it doesn't already exist
func addColumnIfNotExists(db *sql.DB, tableName, columnName, columnDef string) error {
	// Check if column exists by querying table info
	rows, err := db.Query("PRAGMA table_info(" + tableName + ")")
	if err != nil {
		return fmt.Errorf("query table info: %w", err)
	}
	defer rows.Close()

	columnExists := false
	for rows.Next() {
		var cid int
		var name, typ string
		var notnull, pk int
		var dfltValue interface{}
		err := rows.Scan(&cid, &name, &typ, &notnull, &dfltValue, &pk)
		if err != nil {
			return fmt.Errorf("scan table info: %w", err)
		}
		if name == columnName {
			columnExists = true
			break
		}
	}

	if err := rows.Err(); err != nil {
		return fmt.Errorf("rows error: %w", err)
	}

	if !columnExists {
		alterSQL := fmt.Sprintf("ALTER TABLE %s ADD COLUMN %s %s", tableName, columnName, columnDef)
		_, err := db.Exec(alterSQL)
		if err != nil {
			return fmt.Errorf("add column %s to %s: %w", columnName, tableName, err)
		}
	}

	return nil
}

// CloneContext creates a copy of the specified source context with a new name.
func CloneContext(db *sql.DB, sourceName, destName string) error {
	if sourceName == "" || destName == "" {
		return fmt.Errorf("source and destination context names cannot be empty")
	}

	// Get source context
	sourceContext, err := GetContextByName(db, sourceName)
	if err != nil {
		return fmt.Errorf("clone from %s to %s: source context not found: %w", sourceName, destName, err)
	}

	// Check if destination already exists
	_, err = GetContextByName(db, destName)
	if err == nil {
		return fmt.Errorf("clone from %s to %s: destination context already exists", sourceName, destName)
	}
	if !errors.Is(err, sql.ErrNoRows) {
		return fmt.Errorf("clone from %s to %s: %w", sourceName, destName, err)
	}

	// Create destination context with same threading settings
	destContext, err := CreateContextWithThreading(db, destName, sourceContext.UseServerSideThreading)
	if err != nil {
		return fmt.Errorf("clone from %s to %s: create destination context: %w", sourceName, destName, err)
	}

	// Copy all records from source to destination
	_, err = db.Exec(`
		INSERT INTO records (context_id, source, content, live, est_tokens, ts, response_id, streamed, partial_response_id, accumulated_tokens)
		SELECT ?, source, content, live, est_tokens, ts, response_id, COALESCE(streamed, 0), partial_response_id, accumulated_tokens
		FROM records
		WHERE context_id = ?`,
		destContext.ID, sourceContext.ID)
	if err != nil {
		return fmt.Errorf("clone from %s to %s: copy records: %w", sourceName, destName, err)
	}

	return nil
}

// ValidateResponseIDChain checks if a response_id chain is valid for server-side threading.
func ValidateResponseIDChain(db *sql.DB, ctx Context) (valid bool, reason string) {
	var (
		err     error
		isValid = func(id *string) bool {
			return id != nil && *id != ""
		}
	)

	// If no LastResponseID, chain is invalid
	if !isValid(ctx.LastResponseID) {
		ctx, err = GetContext(db, ctx.ID)
		if err != nil {
			return false, "can't load context from db"
		}
	}

	if !isValid(ctx.LastResponseID) {
		return false, "no last_response_id set"
	}

	records, err := ListLiveRecords(db, ctx.ID)
	if err != nil {
		return false, fmt.Sprintf("cannot list records: %v", err)
	}

	// Check for tool calls - these break server-side threading
	for _, rec := range records {
		if rec.Source == ToolCall || rec.Source == ToolOutput {
			return false, "tool calls present (break server-side threading)"
		}
	}

	var modelResponses []Record
	for _, rec := range records {
		if rec.Source == ModelResp {
			modelResponses = append(modelResponses, rec)
		}
	}

	// If no model responses, chain is valid (first call)
	if len(modelResponses) == 0 {
		return true, "no model responses yet (first call)"
	}

	// Check for gaps in response_id chain
	var (
		hasResponseIDs        = false
		hasMissingResponseIDs = false
		lastResponseIDExists  = false
	)

	for _, rec := range modelResponses {
		if isValid(rec.ResponseID) {
			hasResponseIDs = true
			// Check if this matches the context's LastResponseID (for existence check)
			if ctx.LastResponseID != nil && *rec.ResponseID == *ctx.LastResponseID {
				lastResponseIDExists = true
			}
		} else {
			hasMissingResponseIDs = true
		}
	}

	// Edge case: Context has LastResponseID but no matching record exists
	// This can happen after export/import or manual database edits
	if isValid(ctx.LastResponseID) && !lastResponseIDExists {
		return false, fmt.Sprintf("last_response_id (%v) does not exist in records (chain broken, possibly after export/import)",
			*ctx.LastResponseID)
	}

	if hasResponseIDs && hasMissingResponseIDs {
		return false, "mixed response_id state (some records missing IDs)"
	}

	// Last response must match context's LastResponseID
	// Get the last response ID from records
	lastResponseID := getLastResponseID(records)
	if lastResponseID == nil || ctx.LastResponseID == nil || *lastResponseID != *ctx.LastResponseID {
		return false, fmt.Sprintf("last response_id (%v) does not match context (%v)",
			lastResponseID, ctx.LastResponseID)
	}

	return true, "chain valid"
}

// getLastResponseID is a helper to get the last response ID from records.
func getLastResponseID(records []Record) *string {
	for i := len(records) - 1; i >= 0; i-- {
		if records[i].Source == ModelResp && records[i].ResponseID != nil {
			return records[i].ResponseID
		}
	}
	return nil
}

// SavePartialResponse saves a partial streaming response with tracking information.
// This allows resuming a stream after interruption.
// partialResponseID should be a unique identifier for this partial response session.
// accumulatedTokens is the total number of tokens accumulated so far in the stream.
func SavePartialResponse(
	db *sql.DB,
	contextID string,
	content string,
	partialResponseID string,
	accumulatedTokens int,
) (Record, error) {
	return InsertRecordWithPartialResponse(
		db,
		contextID,
		ModelResp,
		content,
		true, // live
		nil,  // responseID
		true, // streamed
		&partialResponseID,
		&accumulatedTokens,
	)
}

// FindPartialResponse finds a partial response by its partial_response_id.
// Returns the record if found, or sql.ErrNoRows if not found.
func FindPartialResponse(db *sql.DB, partialResponseID string) (Record, error) {
	var r Record
	var src int
	err := db.QueryRow(
		`SELECT id, context_id, ts, source, content, live, est_tokens, response_id, 
		 COALESCE(streamed, 0) as streamed, partial_response_id, accumulated_tokens
		 FROM records WHERE partial_response_id = ? AND live = 1`,
		partialResponseID,
	).Scan(
		&r.ID,
		&r.ContextID,
		&r.Timestamp,
		&src,
		&r.Content,
		&r.Live,
		&r.EstTokens,
		&r.ResponseID,
		&r.Streamed,
		&r.PartialResponseID,
		&r.AccumulatedTokens,
	)
	if err != nil {
		return Record{}, fmt.Errorf("find partial response %s: %w", partialResponseID, err)
	}
	r.Source = RecordType(src)
	return r, nil
}

// ResumeFromPartialResponse retrieves a partial response and returns its content and accumulated token count.
// This allows resuming a stream from where it left off.
func ResumeFromPartialResponse(db *sql.DB, partialResponseID string) (content string, accumulatedTokens int, err error) {
	rec, err := FindPartialResponse(db, partialResponseID)
	if err != nil {
		return "", 0, err
	}
	if rec.AccumulatedTokens == nil {
		return rec.Content, 0, nil
	}
	return rec.Content, *rec.AccumulatedTokens, nil
}

// UpdatePartialResponse updates an existing partial response with new content and token count.
// This is used to update a partial response as more tokens arrive during streaming.
func UpdatePartialResponse(
	db *sql.DB,
	partialResponseID string,
	content string,
	accumulatedTokens int,
) error {
	_, err := db.Exec(
		`UPDATE records SET content = ?, accumulated_tokens = ?, est_tokens = ? 
		 WHERE partial_response_id = ? AND live = 1`,
		content, accumulatedTokens, tokenCount(content), partialResponseID,
	)
	if err != nil {
		return fmt.Errorf("update partial response %s: %w", partialResponseID, err)
	}
	return nil
}

// CompletePartialResponse marks a partial response as complete by removing the partial_response_id
// and optionally setting a final response_id. This should be called when streaming finishes successfully.
func CompletePartialResponse(
	db *sql.DB,
	partialResponseID string,
	responseID *string,
) error {
	_, err := db.Exec(
		`UPDATE records SET partial_response_id = NULL, accumulated_tokens = NULL, response_id = ? 
		 WHERE partial_response_id = ? AND live = 1`,
		responseID, partialResponseID,
	)
	if err != nil {
		return fmt.Errorf("complete partial response %s: %w", partialResponseID, err)
	}
	return nil
}
