# Chat Markdown Rendering Design

## Goal

Upgrade the Next.js web app chat page to render assistant messages as styled markdown with syntax-highlighted code blocks, copy/download action buttons on hover — matching the Chrome extension's chat UI exactly.

## Architecture

A new `MarkdownMessage` component encapsulates all rendering logic. The chat page delegates assistant message rendering to it. User message rendering stays inline in `chat/page.tsx` but gains a hover copy button.

```
chat/page.tsx
  └── MarkdownMessage (assistant messages)
        ├── marked (markdown → HTML)
        ├── highlight.js (code block syntax highlighting)
        └── copy / download buttons (hover)
```

## Files

**New:**
- `components/app/MarkdownMessage.tsx` — streaming-aware markdown renderer + action buttons

**Modified:**
- `app/app/(auth)/chat/page.tsx` — use `MarkdownMessage` for assistant messages; add user copy button
- `styles/app.css` — append markdown styles + action button styles (ported from extension `popup.css`)
- `package.json` — add `marked`, `highlight.js`

## MarkdownMessage Component

**Props:**
```ts
interface Props {
  content: string
  streaming: boolean
}
```

**Rendering logic:**
- While `streaming === true`: render raw text + cursor `▋` as plain `<div>`. Avoids re-parsing HTML on every token append.
- When `streaming === false` (or content non-empty and streaming done): call `marked.parse(content)` and set via `dangerouslySetInnerHTML`. This is safe because content comes from our own API, not user input.
- `marked` is configured once at module level: `gfm: true`, `breaks: true`.
- `highlight.js` is set as the `highlight` function in marked options: auto-detect language, wrap in `<pre><code class="hljs">`.

**Action buttons (assistant messages):**
- Container `div.message-action-buttons` is absolutely positioned top-right, `opacity: 0`, transitions to `opacity: 1` on `.chat-message.assistant:hover` via CSS.
- Copy button: writes `content` (raw markdown) to clipboard. On success shows checkmark for 2s then reverts.
- Download button: downloads `content` as `CopiioAI_Chat_<timestamp>.md`. On success shows checkmark for 2s then reverts.
- Buttons only rendered when `streaming === false` and `content.trim()` is non-empty.

**User message copy button:**
- Positioned absolutely to the left of the message bubble (`left: -44px`).
- Visible on `.chat-message.user:hover` via CSS.
- Implemented directly in `chat/page.tsx` as a small inline component `UserCopyButton`.

## CSS Strategy

Append to `styles/app.css`:

1. **Markdown content styles** — ported verbatim from `.chat-message.assistant` rules in `copiioai-ext/styles/popup.css`: headings, paragraphs, lists, code/pre, tables, blockquotes, links, hr, images.
2. **Action button styles** — `.message-action-buttons`, `.copy-button`, `.download-button`, `.user-copy-button`, `.user-copy-button-bridge` ported verbatim from extension.
3. **highlight.js theme** — import `highlight.js/styles/github.css` in `MarkdownMessage.tsx` (CSS module import). The github theme matches the extension's `#f6f8fa` code block background.

## Streaming UX Detail

During streaming the component renders:
```tsx
<div className="chat-message assistant">
  {content || '▋'}
</div>
```

Once streaming completes it switches to:
```tsx
<div
  className="chat-message assistant"
  dangerouslySetInnerHTML={{ __html: marked.parse(content) }}
/>
```

This single switch at stream-end avoids 100s of re-parses mid-stream, which would cause visible flicker and layout thrashing on large responses.

## Dependencies

```json
"marked": "^12.0.0",
"highlight.js": "^11.9.0"
```

No `@types/marked` needed — `marked` v12 ships its own types. No `@types/highlight.js` needed — `highlight.js` v11 ships its own types.
