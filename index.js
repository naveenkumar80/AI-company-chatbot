// Compatibility entry point.
// The real CLI lives in rag.js, but keeping index.js avoids broken old commands
// such as `node index.js chat`.
import "./rag.js";
