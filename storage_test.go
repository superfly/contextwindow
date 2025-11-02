package contextwindow

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	_ "modernc.org/sqlite"
)

func TestValidateResponseIDChain_ValidChain(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)
	defer db.Close()

	// Create context with threading enabled
	ctx, err := CreateContextWithThreading(db, "test-context", true)
	assert.NoError(t, err)

	// Add a prompt
	_, err = InsertRecord(db, ctx.ID, Prompt, "test prompt", true)
	assert.NoError(t, err)

	// Add a model response with response_id
	responseID := "resp-123"
	_, err = InsertRecordWithResponseID(db, ctx.ID, ModelResp, "test response", true, &responseID)
	assert.NoError(t, err)

	// Update context with last response ID
	err = UpdateContextLastResponseID(db, ctx.ID, responseID)
	assert.NoError(t, err)

	// Validate chain - should be valid
	valid, reason := ValidateResponseIDChain(db, ctx.ID)
	assert.True(t, valid)
	assert.Equal(t, "chain valid", reason)
}

func TestValidateResponseIDChain_MissingLastResponseID(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)
	defer db.Close()

	// Create context with threading enabled but no LastResponseID set
	ctx, err := CreateContextWithThreading(db, "test-context", true)
	assert.NoError(t, err)

	// Add a prompt
	_, err = InsertRecord(db, ctx.ID, Prompt, "test prompt", true)
	assert.NoError(t, err)

	// Validate chain - should be invalid (no LastResponseID)
	valid, reason := ValidateResponseIDChain(db, ctx.ID)
	assert.False(t, valid)
	assert.Equal(t, "no last_response_id set", reason)
}

func TestValidateResponseIDChain_ToolCallsPresent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)
	defer db.Close()

	// Create context with threading enabled
	ctx, err := CreateContextWithThreading(db, "test-context", true)
	assert.NoError(t, err)

	// Add a prompt
	_, err = InsertRecord(db, ctx.ID, Prompt, "test prompt", true)
	assert.NoError(t, err)

	// Add a model response with response_id
	responseID := "resp-123"
	_, err = InsertRecordWithResponseID(db, ctx.ID, ModelResp, "test response", true, &responseID)
	assert.NoError(t, err)

	// Update context with last response ID
	err = UpdateContextLastResponseID(db, ctx.ID, responseID)
	assert.NoError(t, err)

	// Add a tool call - this should break server-side threading
	_, err = InsertRecord(db, ctx.ID, ToolCall, "tool call", true)
	assert.NoError(t, err)

	// Validate chain - should be invalid (tool calls present)
	valid, reason := ValidateResponseIDChain(db, ctx.ID)
	assert.False(t, valid)
	assert.Equal(t, "tool calls present (break server-side threading)", reason)
}

func TestValidateResponseIDChain_ToolOutputPresent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)
	defer db.Close()

	// Create context with threading enabled
	ctx, err := CreateContextWithThreading(db, "test-context", true)
	assert.NoError(t, err)

	// Add a prompt
	_, err = InsertRecord(db, ctx.ID, Prompt, "test prompt", true)
	assert.NoError(t, err)

	// Add a model response with response_id
	responseID := "resp-123"
	_, err = InsertRecordWithResponseID(db, ctx.ID, ModelResp, "test response", true, &responseID)
	assert.NoError(t, err)

	// Update context with last response ID
	err = UpdateContextLastResponseID(db, ctx.ID, responseID)
	assert.NoError(t, err)

	// Add a tool output - this should break server-side threading
	_, err = InsertRecord(db, ctx.ID, ToolOutput, "tool output", true)
	assert.NoError(t, err)

	// Validate chain - should be invalid (tool calls present)
	valid, reason := ValidateResponseIDChain(db, ctx.ID)
	assert.False(t, valid)
	assert.Equal(t, "tool calls present (break server-side threading)", reason)
}

func TestValidateResponseIDChain_MixedResponseIDState(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)
	defer db.Close()

	// Create context with threading enabled
	ctx, err := CreateContextWithThreading(db, "test-context", true)
	assert.NoError(t, err)

	// Add a prompt
	_, err = InsertRecord(db, ctx.ID, Prompt, "test prompt", true)
	assert.NoError(t, err)

	// Add a model response with response_id
	responseID1 := "resp-123"
	_, err = InsertRecordWithResponseID(db, ctx.ID, ModelResp, "response 1", true, &responseID1)
	assert.NoError(t, err)

	// Add a model response WITHOUT response_id (mixed state)
	_, err = InsertRecord(db, ctx.ID, ModelResp, "response 2", true)
	assert.NoError(t, err)

	// Update context with last response ID
	err = UpdateContextLastResponseID(db, ctx.ID, responseID1)
	assert.NoError(t, err)

	// Validate chain - should be invalid (mixed state)
	valid, reason := ValidateResponseIDChain(db, ctx.ID)
	assert.False(t, valid)
	assert.Equal(t, "mixed response_id state (some records missing IDs)", reason)
}

func TestValidateResponseIDChain_LastResponseIDMismatch(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)
	defer db.Close()

	// Create context with threading enabled
	ctx, err := CreateContextWithThreading(db, "test-context", true)
	assert.NoError(t, err)

	// Add a prompt
	_, err = InsertRecord(db, ctx.ID, Prompt, "test prompt", true)
	assert.NoError(t, err)

	// Add a model response with response_id
	responseID1 := "resp-123"
	_, err = InsertRecordWithResponseID(db, ctx.ID, ModelResp, "response 1", true, &responseID1)
	assert.NoError(t, err)

	// Add another model response with different response_id
	responseID2 := "resp-456"
	_, err = InsertRecordWithResponseID(db, ctx.ID, ModelResp, "response 2", true, &responseID2)
	assert.NoError(t, err)

	// Update context with a different response ID that doesn't exist in records
	// This is more serious than a mismatch - the ID doesn't exist at all
	wrongResponseID := "resp-wrong"
	err = UpdateContextLastResponseID(db, ctx.ID, wrongResponseID)
	assert.NoError(t, err)

	// Validate chain - should be invalid (LastResponseID doesn't exist in records)
	valid, reason := ValidateResponseIDChain(db, ctx.ID)
	assert.False(t, valid)
	assert.Contains(t, reason, "does not exist in records")
	assert.Contains(t, reason, "export/import")
}

func TestValidateResponseIDChain_LastResponseIDMismatchWithExistingID(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)
	defer db.Close()

	// Create context with threading enabled
	ctx, err := CreateContextWithThreading(db, "test-context", true)
	assert.NoError(t, err)

	// Add a prompt
	_, err = InsertRecord(db, ctx.ID, Prompt, "test prompt", true)
	assert.NoError(t, err)

	// Add a model response with response_id
	responseID1 := "resp-123"
	_, err = InsertRecordWithResponseID(db, ctx.ID, ModelResp, "response 1", true, &responseID1)
	assert.NoError(t, err)

	// Add another model response with different response_id (this is the last one)
	responseID2 := "resp-456"
	_, err = InsertRecordWithResponseID(db, ctx.ID, ModelResp, "response 2", true, &responseID2)
	assert.NoError(t, err)

	// Update context with responseID1 (exists but doesn't match the last response)
	err = UpdateContextLastResponseID(db, ctx.ID, responseID1)
	assert.NoError(t, err)

	// Validate chain - should be invalid (LastResponseID doesn't match last response)
	valid, reason := ValidateResponseIDChain(db, ctx.ID)
	assert.False(t, valid)
	assert.Contains(t, reason, "does not match context")
}

func TestValidateResponseIDChain_EmptyContext(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)
	defer db.Close()

	// Create context with threading enabled
	ctx, err := CreateContextWithThreading(db, "test-context", true)
	assert.NoError(t, err)

	// Set a LastResponseID even though there are no records
	responseID := "resp-123"
	err = UpdateContextLastResponseID(db, ctx.ID, responseID)
	assert.NoError(t, err)

	// Validate chain - should be valid (no model responses yet, first call)
	valid, reason := ValidateResponseIDChain(db, ctx.ID)
	assert.True(t, valid)
	assert.Equal(t, "no model responses yet (first call)", reason)
}

func TestValidateResponseIDChain_NoModelResponsesButHasPrompt(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)
	defer db.Close()

	// Create context with threading enabled
	ctx, err := CreateContextWithThreading(db, "test-context", true)
	assert.NoError(t, err)

	// Add a prompt but no model responses
	_, err = InsertRecord(db, ctx.ID, Prompt, "test prompt", true)
	assert.NoError(t, err)

	// Set a LastResponseID
	responseID := "resp-123"
	err = UpdateContextLastResponseID(db, ctx.ID, responseID)
	assert.NoError(t, err)

	// Validate chain - should be valid (no model responses yet, first call)
	valid, reason := ValidateResponseIDChain(db, ctx.ID)
	assert.True(t, valid)
	assert.Equal(t, "no model responses yet (first call)", reason)
}

func TestValidateResponseIDChain_ContextNotFound(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)
	defer db.Close()

	// Try to validate with non-existent context ID
	valid, reason := ValidateResponseIDChain(db, "non-existent-id")
	assert.False(t, valid)
	assert.Contains(t, reason, "cannot get context")
}

func TestValidateResponseIDChain_MultipleValidResponses(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)
	defer db.Close()

	// Create context with threading enabled
	ctx, err := CreateContextWithThreading(db, "test-context", true)
	assert.NoError(t, err)

	// Add a prompt
	_, err = InsertRecord(db, ctx.ID, Prompt, "test prompt", true)
	assert.NoError(t, err)

	// Add multiple model responses with response_ids
	responseID1 := "resp-123"
	_, err = InsertRecordWithResponseID(db, ctx.ID, ModelResp, "response 1", true, &responseID1)
	assert.NoError(t, err)

	responseID2 := "resp-456"
	_, err = InsertRecordWithResponseID(db, ctx.ID, ModelResp, "response 2", true, &responseID2)
	assert.NoError(t, err)

	// Update context with the last response ID
	err = UpdateContextLastResponseID(db, ctx.ID, responseID2)
	assert.NoError(t, err)

	// Validate chain - should be valid (last response matches)
	valid, reason := ValidateResponseIDChain(db, ctx.ID)
	assert.True(t, valid)
	assert.Equal(t, "chain valid", reason)
}

func TestGetLastResponseID(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)
	defer db.Close()

	// Create context
	ctx, err := CreateContextWithThreading(db, "test-context", true)
	assert.NoError(t, err)

	// Add records
	_, err = InsertRecord(db, ctx.ID, Prompt, "prompt", true)
	assert.NoError(t, err)

	responseID1 := "resp-1"
	_, err = InsertRecordWithResponseID(db, ctx.ID, ModelResp, "response 1", true, &responseID1)
	assert.NoError(t, err)

	responseID2 := "resp-2"
	_, err = InsertRecordWithResponseID(db, ctx.ID, ModelResp, "response 2", true, &responseID2)
	assert.NoError(t, err)

	// Get records
	records, err := ListLiveRecords(db, ctx.ID)
	assert.NoError(t, err)

	// Test getLastResponseID helper
	lastID := getLastResponseID(records)
	assert.NotNil(t, lastID)
	assert.Equal(t, responseID2, *lastID)
}

func TestGetLastResponseID_NoResponseIDs(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)
	defer db.Close()

	// Create context
	ctx, err := CreateContextWithThreading(db, "test-context", true)
	assert.NoError(t, err)

	// Add records without response IDs
	_, err = InsertRecord(db, ctx.ID, Prompt, "prompt", true)
	assert.NoError(t, err)
	_, err = InsertRecord(db, ctx.ID, ModelResp, "response", true)
	assert.NoError(t, err)

	// Get records
	records, err := ListLiveRecords(db, ctx.ID)
	assert.NoError(t, err)

	// Test getLastResponseID helper - should return nil
	lastID := getLastResponseID(records)
	assert.Nil(t, lastID)
}

// TestValidateResponseIDChain_LastResponseIDNoMatchingRecord tests the edge case
// where a context has a LastResponseID but no matching record exists in the database.
// This can happen after export/import or manual database edits.
func TestValidateResponseIDChain_LastResponseIDNoMatchingRecord(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)
	defer db.Close()

	// Create context with threading enabled
	ctx, err := CreateContextWithThreading(db, "test-context", true)
	assert.NoError(t, err)

	// Add a prompt
	_, err = InsertRecord(db, ctx.ID, Prompt, "test prompt", true)
	assert.NoError(t, err)

	// Add a model response with a different response_id
	responseID1 := "resp-123"
	_, err = InsertRecordWithResponseID(db, ctx.ID, ModelResp, "response 1", true, &responseID1)
	assert.NoError(t, err)

	// Set a LastResponseID that doesn't match any record (simulating export/import scenario)
	wrongResponseID := "resp-nonexistent-999"
	err = UpdateContextLastResponseID(db, ctx.ID, wrongResponseID)
	assert.NoError(t, err)

	// Validate chain - should be invalid (LastResponseID doesn't exist in records)
	valid, reason := ValidateResponseIDChain(db, ctx.ID)
	assert.False(t, valid)
	assert.Contains(t, reason, "does not exist in records")
	assert.Contains(t, reason, "export/import")
}

