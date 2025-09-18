# ContextReader Implementation Plan

## Rationale
The current `ContextWindow` API doesn't clearly communicate what's safe for concurrent use. Users might reasonably assume database operations are safe but be uncertain about methods like `GetCurrentContext()` or `TokenUsage()`. By creating a `ContextReader` type, we:

1. **Make safety explicit** - The type system enforces that UI goroutines can only access safe operations
2. **Follow Go idioms** - Similar to `strings.Reader`, `bytes.Reader` - concrete types for read-only access
3. **Zero overhead** - Just a thin proxy wrapper, no state duplication
4. **Solve the real problem** - UI needs consistent reads while one goroutine manages state

## Implementation Strategy

**Proxy approach**: ContextReader is just a thin wrapper that forwards method calls to the underlying ContextWindow. This provides type safety (restricts API surface) without any overhead or state duplication.

**Eventually consistent reads**: Reads may see intermediate states if concurrent writes occur, but this is acceptable for UI usage where perfect consistency isn't required.

## 1. Create ContextReader struct
```go
// ContextReader provides thread-safe read access to context window data.
// It's a thin proxy that forwards calls to safe read methods.
type ContextReader struct {
    cw *ContextWindow // reference to underlying context window
}
```

## 2. Add Reader() method to ContextWindow
- Create `Reader()` method that returns a `*ContextReader`
- Simply wraps the ContextWindow instance
- Documents that returned reader is safe for concurrent use

## 3. Implement safe read methods on ContextReader
Add these proxy methods to ContextReader (forward to underlying ContextWindow):
- `LiveRecords()`, `LiveTokens()`, `TokenUsage()`
- `GetCurrentContext()`, `GetCurrentContextInfo()`
- `ListContexts()`, `GetContext()`, `GetContextStats()`
- `ExportContext()`, `ExportContextJSON()`
- `TotalTokens()` (metrics access)

All methods simply forward calls to the underlying ContextWindow.

## 4. Update package documentation
- Add "Thread Safety" section explaining the Reader pattern
- Document that ContextWindow write operations need external coordination
- Show example: `reader := cw.Reader(); go updateUI(reader)`

## 5. Add tests
- Test that Reader properly forwards method calls
- Test concurrent access patterns (Reader + Writer)
- Verify Reader has all expected read methods

## Benefits
- **Clear API contract** - Types enforce safe usage patterns
- **No performance cost** - Pure method forwarding, zero overhead
- **Familiar pattern** - Follows established Go conventions
- **Solves real UX problem** - UI can safely read while state changes