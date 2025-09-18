# Context Window Resume Plan

**Date:** 2025-09-17

## Problem Statement

Currently, the ContextWindow library has inconsistent behavior when dealing with existing context names:

- `NewContextWindowWithThreading` allows existing names (get-or-create behavior)
- `CreateContext`/`CreateContextWithThreading` reject existing names with an error
- `SwitchContext` requires the context to exist

This creates confusion and prevents the library from behaving like a persistent conversation system where users can always resume any named context.

## Current Behavior Analysis

The system currently:
- Loads all "live" records when making LLM calls via `ListLiveRecords`
- Supports two threading modes:
  - **Server-side threading**: Only sends the latest prompt + `previous_response_id` to LLM
  - **Client-side threading**: Sends full message history to LLM
- `NewContextWindowWithThreading` already works correctly (get-or-create)
- Context switching properly loads existing message history

## Proposed Solution

Implement consistent get-or-create behavior across all context-related functions, making the library behave as if someone just continued the context window that was previously saved.

## Implementation Plan

### 1. Update CreateContextWithThreading (storage.go:133)

**Current behavior:** Fails if context name already exists
**New behavior:** Return existing context if name already exists

```go
// Remove the name collision check (lines 139-146)
// Instead, check if context exists and return it
```

### 2. Update SwitchContext (contextwindow.go:634)

**Current behavior:** Fails if context doesn't exist
**New behavior:** Create context if it doesn't exist

```go
// If GetContextByName fails with sql.ErrNoRows, create the context
// Otherwise, proceed as normal
```

### 3. Handle Threading Complexity

**Key issue:** When resuming an existing context, we need to handle cases where `response_id` threading chain may be broken.

**Solution:** Enhance the threading logic in `CallModelWithOpts`:
- If server-side threading is enabled AND we have a valid `last_response_id`, use server-side threading
- Otherwise, gracefully fall back to client-side threading with full history
- Ensure seamless continuation regardless of how the context was previously used

### 4. Context Continuation Logic

When someone adds a prompt to an existing context:
1. Load existing message history via `ListLiveRecords`
2. Determine appropriate threading mode based on context state
3. Handle broken `response_id` chains gracefully
4. Ensure no message history is lost during continuation

## Testing Strategy

### 5. Test Get-or-Create Behavior
- Verify `CreateContext` with existing names returns existing context
- Verify `SwitchContext` creates context if it doesn't exist
- Ensure consistency across all context functions

### 6. Test Context Continuation
- Create context, add messages, close
- Reopen context and add new messages
- Verify all previous messages are included in LLM calls
- Test with both threading modes

### 7. Test Threading Behavior
- Test resuming contexts with valid `response_id` chains
- Test resuming contexts with broken `response_id` chains
- Test switching between server-side and client-side threading
- Verify graceful fallback behavior

### 8. Test Edge Cases
- Empty contexts
- Contexts with only system prompts
- Contexts with mixed threading histories
- Very long context histories

## Files to Modify

1. `storage.go` - Update `CreateContextWithThreading`
2. `contextwindow.go` - Update `SwitchContext`
3. `contextwindow.go` - Enhance threading logic in `CallModelWithOpts`
4. `contextwindow_test.go` - Add comprehensive tests

## Success Criteria

- All context functions consistently support get-or-create behavior
- Existing contexts can be seamlessly resumed with full message history
- Threading mode transitions work gracefully
- No message history is lost during context continuation
- Comprehensive test coverage for all scenarios

## Risk Mitigation

- Maintain backward compatibility for existing threading behavior
- Ensure database operations remain atomic
- Add detailed logging for threading decisions
- Graceful fallback for any threading failures