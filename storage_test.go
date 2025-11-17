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
	valid, reason := ValidateResponseIDChain(db, ctx)
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
	valid, reason := ValidateResponseIDChain(db, ctx)
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
	valid, reason := ValidateResponseIDChain(db, ctx)
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
	valid, reason := ValidateResponseIDChain(db, ctx)
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
	valid, reason := ValidateResponseIDChain(db, ctx)
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
	valid, reason := ValidateResponseIDChain(db, ctx)
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
	valid, reason := ValidateResponseIDChain(db, ctx)
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
	valid, reason := ValidateResponseIDChain(db, ctx)
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
	valid, reason := ValidateResponseIDChain(db, ctx)
	assert.True(t, valid)
	assert.Equal(t, "no model responses yet (first call)", reason)
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
	valid, reason := ValidateResponseIDChain(db, ctx)
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
	valid, _ := ValidateResponseIDChain(db, ctx)
	assert.False(t, valid)
}

// TestStreamedFlagMigration tests that the streamed column is added to existing databases
func TestStreamedFlagMigration(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)
	defer db.Close()

	// Create a context and insert a record before migration
	ctx, err := CreateContext(db, "test-context")
	assert.NoError(t, err)

	// Insert a record using the old API (without streamed flag)
	rec, err := InsertRecord(db, ctx.ID, Prompt, "test prompt", true)
	assert.NoError(t, err)
	assert.False(t, rec.Streamed, "old records should have streamed=false by default")

	// Verify the record can be read back with streamed field
	records, err := ListRecordsInContext(db, ctx.ID)
	assert.NoError(t, err)
	assert.Len(t, records, 1)
	assert.False(t, records[0].Streamed, "migrated records should have streamed=false")

	// Now test that new records can be inserted with streamed flag
	rec2, err := InsertRecordWithResponseIDAndStreamed(db, ctx.ID, ModelResp, "streamed response", true, nil, true)
	assert.NoError(t, err)
	assert.True(t, rec2.Streamed, "new streamed records should have streamed=true")

	// Verify both records are correctly stored
	records, err = ListRecordsInContext(db, ctx.ID)
	assert.NoError(t, err)
	assert.Len(t, records, 2)
	assert.False(t, records[0].Streamed, "first record should not be streamed")
	assert.True(t, records[1].Streamed, "second record should be streamed")
}

// TestStreamedFlagNewRecords tests that new records properly store the streamed flag
func TestStreamedFlagNewRecords(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)
	defer db.Close()

	ctx, err := CreateContext(db, "test-context")
	assert.NoError(t, err)

	// Test InsertRecordStreamed
	rec1, err := InsertRecordStreamed(db, ctx.ID, Prompt, "streamed prompt", true, true)
	assert.NoError(t, err)
	assert.True(t, rec1.Streamed)

	// Test InsertRecordWithResponseIDAndStreamed
	responseID := "resp-123"
	rec2, err := InsertRecordWithResponseIDAndStreamed(db, ctx.ID, ModelResp, "streamed response", true, &responseID, true)
	assert.NoError(t, err)
	assert.True(t, rec2.Streamed)
	assert.Equal(t, &responseID, rec2.ResponseID)

	// Test non-streamed record
	rec3, err := InsertRecord(db, ctx.ID, ToolCall, "tool call", true)
	assert.NoError(t, err)
	assert.False(t, rec3.Streamed)

	// Verify all records
	records, err := ListRecordsInContext(db, ctx.ID)
	assert.NoError(t, err)
	assert.Len(t, records, 3)
	assert.True(t, records[0].Streamed, "first record should be streamed")
	assert.True(t, records[1].Streamed, "second record should be streamed")
	assert.False(t, records[2].Streamed, "third record should not be streamed")
}

// TestStreamedFlagBackwardCompatibility tests that existing code continues to work
func TestStreamedFlagBackwardCompatibility(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)
	defer db.Close()

	ctx, err := CreateContext(db, "test-context")
	assert.NoError(t, err)

	// Test that InsertRecord (old API) still works and defaults to streamed=false
	rec1, err := InsertRecord(db, ctx.ID, Prompt, "prompt", true)
	assert.NoError(t, err)
	assert.False(t, rec1.Streamed, "InsertRecord should default to streamed=false")

	// Test that InsertRecordWithResponseID (old API) still works
	responseID := "resp-123"
	rec2, err := InsertRecordWithResponseID(db, ctx.ID, ModelResp, "response", true, &responseID)
	assert.NoError(t, err)
	assert.False(t, rec2.Streamed, "InsertRecordWithResponseID should default to streamed=false")
	assert.Equal(t, &responseID, rec2.ResponseID)

	// Verify records can be read back correctly
	records, err := ListLiveRecords(db, ctx.ID)
	assert.NoError(t, err)
	assert.Len(t, records, 2)
	for _, rec := range records {
		assert.False(t, rec.Streamed, "all records from old API should have streamed=false")
	}

	// Test that records without streamed column (from old schema) are handled gracefully
	// This simulates reading from an old database that was migrated
	records, err = ListRecordsInContext(db, ctx.ID)
	assert.NoError(t, err)
	assert.Len(t, records, 2)
	for _, rec := range records {
		assert.False(t, rec.Streamed, "migrated records should default to streamed=false")
	}
}

// TestPartialResponseSave tests saving a partial response with tracking information
func TestPartialResponseSave(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)
	defer db.Close()

	ctx, err := CreateContext(db, "test-context")
	assert.NoError(t, err)

	// Save a partial response
	partialID := "partial-123"
	content := "Partial response content"
	accumulatedTokens := 50

	rec, err := SavePartialResponse(db, ctx.ID, content, partialID, accumulatedTokens)
	assert.NoError(t, err)
	assert.NotZero(t, rec.ID)
	assert.Equal(t, content, rec.Content)
	assert.True(t, rec.Streamed)
	assert.True(t, rec.Live)
	assert.Equal(t, ModelResp, rec.Source)
	assert.NotNil(t, rec.PartialResponseID)
	assert.Equal(t, partialID, *rec.PartialResponseID)
	assert.NotNil(t, rec.AccumulatedTokens)
	assert.Equal(t, accumulatedTokens, *rec.AccumulatedTokens)

	// Verify the record can be read back
	records, err := ListRecordsInContext(db, ctx.ID)
	assert.NoError(t, err)
	assert.Len(t, records, 1)
	assert.Equal(t, partialID, *records[0].PartialResponseID)
	assert.Equal(t, accumulatedTokens, *records[0].AccumulatedTokens)
}

// TestPartialResponseResume tests resuming from a partial response
func TestPartialResponseResume(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)
	defer db.Close()

	ctx, err := CreateContext(db, "test-context")
	assert.NoError(t, err)

	// Save a partial response
	partialID := "partial-456"
	content := "Partial content so far"
	accumulatedTokens := 100

	_, err = SavePartialResponse(db, ctx.ID, content, partialID, accumulatedTokens)
	assert.NoError(t, err)

	// Resume from the partial response
	resumedContent, resumedTokens, err := ResumeFromPartialResponse(db, partialID)
	assert.NoError(t, err)
	assert.Equal(t, content, resumedContent)
	assert.Equal(t, accumulatedTokens, resumedTokens)

	// Test resuming from non-existent partial response
	_, _, err = ResumeFromPartialResponse(db, "non-existent")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "find partial response")
}

// TestPartialResponseUpdate tests updating a partial response
func TestPartialResponseUpdate(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)
	defer db.Close()

	ctx, err := CreateContext(db, "test-context")
	assert.NoError(t, err)

	// Save initial partial response
	partialID := "partial-789"
	initialContent := "Initial content"
	initialTokens := 25

	_, err = SavePartialResponse(db, ctx.ID, initialContent, partialID, initialTokens)
	assert.NoError(t, err)

	// Update with more content
	updatedContent := "Initial content with more text"
	updatedTokens := 50

	err = UpdatePartialResponse(db, partialID, updatedContent, updatedTokens)
	assert.NoError(t, err)

	// Verify the update
	rec, err := FindPartialResponse(db, partialID)
	assert.NoError(t, err)
	assert.Equal(t, updatedContent, rec.Content)
	assert.NotNil(t, rec.AccumulatedTokens)
	assert.Equal(t, updatedTokens, *rec.AccumulatedTokens)
}

// TestPartialResponseComplete tests completing a partial response
func TestPartialResponseComplete(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)
	defer db.Close()

	ctx, err := CreateContext(db, "test-context")
	assert.NoError(t, err)

	// Save a partial response
	partialID := "partial-999"
	content := "Complete response content"
	accumulatedTokens := 200

	_, err = SavePartialResponse(db, ctx.ID, content, partialID, accumulatedTokens)
	assert.NoError(t, err)

	// Verify it exists as partial
	rec, err := FindPartialResponse(db, partialID)
	assert.NoError(t, err)
	assert.NotNil(t, rec.PartialResponseID)

	// Complete the partial response
	finalResponseID := "final-response-123"
	err = CompletePartialResponse(db, partialID, &finalResponseID)
	assert.NoError(t, err)

	// Verify it's no longer a partial response
	_, err = FindPartialResponse(db, partialID)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "find partial response")

	// Verify the record now has the final response ID
	records, err := ListRecordsInContext(db, ctx.ID)
	assert.NoError(t, err)
	assert.Len(t, records, 1)
	assert.Nil(t, records[0].PartialResponseID)
	assert.Nil(t, records[0].AccumulatedTokens)
	assert.NotNil(t, records[0].ResponseID)
	assert.Equal(t, finalResponseID, *records[0].ResponseID)
}

// TestPartialResponseMigration tests that the new columns are added to existing databases
func TestPartialResponseMigration(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)
	defer db.Close()

	ctx, err := CreateContext(db, "test-context")
	assert.NoError(t, err)

	// Insert a record using the old API (without partial response fields)
	rec, err := InsertRecord(db, ctx.ID, Prompt, "test prompt", true)
	assert.NoError(t, err)
	assert.Nil(t, rec.PartialResponseID)
	assert.Nil(t, rec.AccumulatedTokens)

	// Verify the record can be read back with the new fields (should be nil)
	records, err := ListRecordsInContext(db, ctx.ID)
	assert.NoError(t, err)
	assert.Len(t, records, 1)
	assert.Nil(t, records[0].PartialResponseID)
	assert.Nil(t, records[0].AccumulatedTokens)

	// Now test that new records can be inserted with partial response fields
	partialID := "partial-migration-test"
	content := "Partial content"
	tokens := 75

	rec2, err := InsertRecordWithPartialResponse(db, ctx.ID, ModelResp, content, true, nil, true, &partialID, &tokens)
	assert.NoError(t, err)
	assert.NotNil(t, rec2.PartialResponseID)
	assert.Equal(t, partialID, *rec2.PartialResponseID)
	assert.NotNil(t, rec2.AccumulatedTokens)
	assert.Equal(t, tokens, *rec2.AccumulatedTokens)

	// Verify both records are correctly stored
	records, err = ListRecordsInContext(db, ctx.ID)
	assert.NoError(t, err)
	assert.Len(t, records, 2)
	assert.Nil(t, records[0].PartialResponseID)
	assert.Nil(t, records[0].AccumulatedTokens)
	assert.NotNil(t, records[1].PartialResponseID)
	assert.NotNil(t, records[1].AccumulatedTokens)
}
