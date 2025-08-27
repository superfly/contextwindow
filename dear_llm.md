### Basic Go Stuff
* use github.com/stretchr/testify/assert to write unit tests (assert.Equal, assert.NoError, assert.Contains, etc)
* keep line length to around 80 characters; prefer vertical to horizontal
* one line per comma-delimited struct or array field.
* follow the user's requirements carefully & to the letter
* never write comments
* Use the standard library's net/http package for HTTP API development
* Use structured logging (slog) when logging
* use simple fmt.Errorf() error wrapping, include only information in the format string that the CALLER would not have
* If unsure about a best practice or implementation detail, say so instead of guessing.
* Prefer small helper functions to repeated code
* Use named functions for goroutines unless the goroutine only spans a couple lines
* "defer" unlocks of locks when possible rather than bracketing critical sections.
* don't create "else" arms on error checking conditions; keep code straight-line
* be familiar with RESTful API design principles, best practices, and Go idioms
* use fmt.Fprintf in preference to Write([]byte(stringValue))
* use StringBuilder or something like it in preference to append/join
