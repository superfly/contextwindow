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
	ID         int64      `json:"id"`
	Timestamp  time.Time  `json:"timestamp"`
	Source     RecordType `json:"source"`
	Content    string     `json:"content"`
	Live       bool       `json:"live"`
	EstTokens  int        `json:"est_tokens"`
	ContextID  string     `json:"context_id"`
	ResponseID *string    `json:"response_id,omitempty"`
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

// InsertRecordWithResponseID inserts a new record with optional response ID.
func InsertRecordWithResponseID(
	db *sql.DB,
	contextID string,
	source RecordType,
	content string,
	live bool,
	responseID *string,
) (Record, error) {
	now := time.Now().UTC()
	t := tokenCount(content)
	res, err := db.Exec(
		`INSERT INTO records (context_id, ts, source, content, live, est_tokens, response_id) 
		 VALUES (?, ?, ?, ?, ?, ?, ?)`,
		contextID, now, int(source), content, live, t, responseID,
	)
	if err != nil {
		return Record{}, fmt.Errorf("insert record: %w", err)
	}
	id, err := res.LastInsertId()
	if err != nil {
		return Record{}, fmt.Errorf("get last insert id: %w", err)
	}
	return Record{
		ID:         id,
		Timestamp:  now,
		Source:     source,
		Content:    content,
		Live:       live,
		EstTokens:  t,
		ContextID:  contextID,
		ResponseID: responseID,
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
		`SELECT id, context_id, ts, source, content, live, est_tokens, response_id 
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
	now := time.Now().UTC()
	t := tokenCount(content)
	res, err := tx.Exec(
		`INSERT INTO records (context_id, ts, source, content, live, est_tokens, response_id) 
		 VALUES (?, ?, ?, ?, ?, ?, ?)`,
		contextID, now, int(source), content, live, t, responseID,
	)
	if err != nil {
		return Record{}, fmt.Errorf("insert record tx: %w", err)
	}
	id, err := res.LastInsertId()
	if err != nil {
		return Record{}, fmt.Errorf("get last insert id tx: %w", err)
	}
	return Record{
		ID:         id,
		Timestamp:  now,
		Source:     source,
		Content:    content,
		Live:       live,
		EstTokens:  t,
		ContextID:  contextID,
		ResponseID: responseID,
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
