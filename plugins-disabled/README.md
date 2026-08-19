# Disabled plugins

Kept for reference, not loaded (the loader only reads `plugins/*.js`).

## language-enforcer.js
Injects `請用繁體中文回答。` ("answer in Traditional Chinese") into the system
prompt whenever ANY message contains a character in `一-鿿`. Since the
assistant's own replies feed back into the next request's history, one Chinese
character anywhere makes the instruction stick permanently. Observed in
practice: the agent started narrating its reasoning in Chinese and would not
stop.

## content-filter.js
Rewrites the model's output, replacing regex matches with `[REDACTED]`. The
patterns are far too broad for a developer agent:

    "A proxy a 127.0.0.1:3456 címen fut. ... password: titkos. Verzió: 1.2.3.4"
 -> "A proxy a [REDACTED]:3456 címen fut. ... [REDACTED] Verzió: [REDACTED]"

Any IPv4 (localhost included), anything after the word "password", and any
four-part version number are destroyed silently. `/firebase.*\.json/gi` is
greedy and eats a whole line. Redaction also happens after the answer is
generated, so it cannot prevent a leak — it only corrupts the text.
