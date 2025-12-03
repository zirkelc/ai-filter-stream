<div align='center'>

# ai-stream-utils

<p align="center">AI SDK: Stream transformation utilities for UI message streams</p>
<p align="center">
  <a href="https://www.npmjs.com/package/ai-stream-utils" alt="ai-stream-utils"><img src="https://img.shields.io/npm/dt/ai-stream-utils?label=ai-stream-utils"></a> <a href="https://github.com/zirkelc/ai-stream-utils/actions/workflows/ci.yml" alt="CI"><img src="https://img.shields.io/github/actions/workflow/status/zirkelc/ai-stream-utils/ci.yml?branch=main"></a>
</p>

</div>

This library provides composable stream transformation and filter utilities for UI message streams created by [`streamText()`](https://ai-sdk.dev/docs/reference/ai-sdk-core/stream-text) in the AI SDK.

### Why?

The AI SDK UI message stream created by [`toUIMessageStream()`](https://ai-sdk.dev/docs/reference/ai-sdk-core/stream-text#to-ui-message-stream) streams all parts (text, tools, reasoning, etc.) to the client by default. However, you may want to:

- **Filter**: Tool calls like database queries often contain large amounts of data or sensitive information that should not be visible on the client
- **Transform**: Modify text or tool outputs while they are streamed to the client

This library provides type-safe, composable utilities for all these use cases.

### Installation

This library only supports AI SDK v5.

```bash
npm install ai-stream-utils
```

## Overview

| Function | Object | Use Case |
|----------|-------------|----------|
| [`mapUIMessageStream`](#mapuimessagestream) | [UIMessageChunk](https://github.com/vercel/ai/blob/main/packages/ai/src/ui-message-stream/ui-message-chunks.ts) | Transform chunks while streaming to the client |
| [`flatMapUIMessageStream`](#flatmapuimessagestream) | [UIMessagePart](https://ai-sdk.dev/docs/reference/ai-sdk-core/ui-message#uimessagepart-types) | Buffer chunks until a part is complete, then transform the part |
| [`filterUIMessageStream`](#filteruimessagestream) | [UIMessageChunk](https://github.com/vercel/ai/blob/main/packages/ai/src/ui-message-stream/ui-message-chunks.ts) | Filter chunks while streaming to the client |

## Usage

### `mapUIMessageStream`

Transform or filter individual chunks as they stream through. The map function receives the chunk and a partial representation of the part it belongs to.

```typescript
import { mapUIMessageStream } from 'ai-stream-utils';
import { streamText } from 'ai';

const tools = {
  weather: tool({
    description: 'Get the weather in a location',
    inputSchema: z.object({
      location: z.string().describe('The location to get the weather for'),
    }),
    execute: ({ location }) => ({
      location,
      temperature: 72 + Math.floor(Math.random() * 21) - 10,
      unit: "C",
    }),
  }),
};

const result = streamText({
  model,
  prompt: 'What is the weather in Tokyo?',
  tools,
});

// Filter out tool-call chunks by part type
const stream = mapUIMessageStream(
  result.toUIMessageStream<MyUIMessage>(),
  ({ chunk, part }) => {
    if (part.type === "tool-weather") {
      return null;
    }
    return chunk;
  }
);

// Transform text chunks to uppercase
const stream = mapUIMessageStream(
  result.toUIMessageStream<MyUIMessage>(),
  ({ chunk, part }) => {
    if (chunk.type === 'text-delta') {
      return { ...chunk, delta: chunk.delta.toUpperCase() };
    }
    return chunk;
  }
);

// Access chunk history and index
const stream = mapUIMessageStream(
  result.toUIMessageStream<MyUIMessage>(),
  ({ chunk }, { index, chunks }) => {
    console.log(`Chunk ${index}, total seen: ${chunks.length}`);
    return chunk;
  }
);
```

### `flatMapUIMessageStream`

Buffer all chunks for a part until it's complete, then transform the complete part. This is useful when you need access to the full part content before deciding how to transform it.

When a predicate is provided (e.g., `partTypeIs('text')`), only matching parts are buffered for transformation. Non-matching parts stream through immediately without buffering, preserving real-time streaming behavior.

```typescript
import { flatMapUIMessageStream, partTypeIs } from 'ai-stream-utils';

// Filter out reasoning parts
const stream = flatMapUIMessageStream(
  result.toUIMessageStream<MyUIMessage>(),
  ({ part }) => part.type === 'reasoning' ? null : part
);

// Transform text content
const stream = flatMapUIMessageStream(
  result.toUIMessageStream<MyUIMessage>(),
  ({ part }) => {
    if (part.type === 'text') {
      return { ...part, text: part.text.toUpperCase() };
    }
    return part;
  }
);

// Buffer only specific parts, pass through others immediately
const stream = flatMapUIMessageStream(
  result.toUIMessageStream<MyUIMessage>(),
  partTypeIs('text'),
  ({ part }) => ({ ...part, text: part.text.toUpperCase() })
);

// Buffer multiple part types
const stream = flatMapUIMessageStream(
  result.toUIMessageStream<MyUIMessage>(),
  partTypeIs(['text', 'reasoning']),
  ({ part }) => part // part is typed as TextUIPart | ReasoningUIPart
);

// Access part history
const stream = flatMapUIMessageStream(
  result.toUIMessageStream<MyUIMessage>(),
  ({ part }, { index, parts }) => {
    console.log(`Part ${index}, previous parts:`, parts.slice(0, -1));
    return part;
  }
);
```

### `filterUIMessageStream`

Filter individual chunks as they stream through. This is a convenience wrapper around `mapUIMessageStream` that provides a simpler API for filtering chunks by part type. Use the `includeParts()` and `excludeParts()` helper functions for common patterns, or provide a custom filter function.

```typescript
import { filterUIMessageStream, includeParts, excludeParts } from 'ai-stream-utils';

// Include only specific parts
const stream = filterUIMessageStream(
  result.toUIMessageStream<MyUIMessage>(),
  includeParts(['text', 'tool-weather'])
);

// Exclude specific parts
const stream = filterUIMessageStream(
  result.toUIMessageStream<MyUIMessage>(),
  excludeParts(['reasoning', 'tool-database'])
);

// Custom filter function
const stream = filterUIMessageStream(
  result.toUIMessageStream<MyUIMessage>(),
  ({ part }, { index }) => {
    // Include text parts
    if (part.type === 'text') return true;
    // Include only first 5 parts
    if (index < 5) return true;
    return false;
  }
);
```

## Type Safety

The [`toUIMessageStream()`](https://ai-sdk.dev/docs/reference/ai-sdk-core/stream-text#to-ui-message-stream) from [`streamText()`](https://ai-sdk.dev/docs/reference/ai-sdk-core/stream-text) returns a generic `ReadableStream<UIMessageChunk>`, which means the part types cannot be inferred automatically.

To enable autocomplete and type-safety, pass your [`UIMessage`](https://ai-sdk.dev/docs/reference/ai-sdk-core/ui-message#creating-your-own-uimessage-type) type as a generic parameter:

```typescript
import type { UIMessage, InferUITools } from 'ai';

type MyUIMessageMetadata = {};
type MyDataPart = {};
type MyTools = InferUITools<typeof tools>;

type MyUIMessage = UIMessage<
  MyUIMessageMetadata,
  MyDataPart,
  MyTools
>;

// Use MyUIMessage type when creating the UI message stream
const uiStream = result.toUIMessageStream<MyUIMessage>();

// Type-safe filtering with autocomplete
const stream = filterUIMessageStream(
  uiStream,
  includeParts(['text', 'tool-weather']) // Autocomplete works!
);

// Type-safe chunk mapping
const stream = mapUIMessageStream(
  uiStream,
  ({ chunk, part }) => {
    // part.type is typed based on MyUIMessage
    return chunk;
  }
);
```

## Client-Side Usage

The transformed stream has the same type as the original UI message stream. You can consume it with [`useChat()`](https://ai-sdk.dev/docs/reference/ai-sdk-ui/use-chat) or [`readUIMessageStream()`](https://ai-sdk.dev/docs/reference/ai-sdk-ui/read-ui-message-stream).

Since message parts may be different on the client vs. the server, you may need to reconcile message parts when the client sends messages back to the server.

If you save messages to a database and configure `useChat()` to [only send the last message](https://ai-sdk.dev/docs/ai-sdk-ui/chatbot-message-persistence#sending-only-the-last-message), you can read existing messages from the database. This means the model will have access to all message parts, including filtered parts not available on the client.

## Part Type Mapping

The transformations operate on [UIMessagePart](https://ai-sdk.dev/docs/reference/ai-sdk-core/ui-message#uimessagepart-types) types, which are derived from [UIMessageChunk](https://github.com/vercel/ai/blob/main/packages/ai/src/ui-message-stream/ui-message-chunks.ts) types:

| Part Type         | Chunk Types                           |
| ----------------- | ------------------------------------- |
| [`text`](https://ai-sdk.dev/docs/reference/ai-sdk-core/ui-message#textuipart)            | `text-start`, `text-delta`, `text-end` |
| [`reasoning`](https://ai-sdk.dev/docs/reference/ai-sdk-core/ui-message#reasoninguipart)       | `reasoning-start`, `reasoning-delta`, `reasoning-end` |
| [`tool-{name}`](https://ai-sdk.dev/docs/reference/ai-sdk-core/ui-message#tooluipart)     | `tool-input-start`, `tool-input-delta`, `tool-input-available`, `tool-input-error`, `tool-output-available`, `tool-output-error` |
| [`data-{name}`](https://ai-sdk.dev/docs/reference/ai-sdk-core/ui-message#datauipart)     | `data-{name}` |
| [`step-start`](https://ai-sdk.dev/docs/reference/ai-sdk-core/ui-message#stepstartuipart)      | `start-step` |
| [`file`](https://ai-sdk.dev/docs/reference/ai-sdk-core/ui-message#fileuipart)            | `file` |
| [`source-url`](https://ai-sdk.dev/docs/reference/ai-sdk-core/ui-message#sourceurluipart)      | `source-url` |
| [`source-document`](https://ai-sdk.dev/docs/reference/ai-sdk-core/ui-message#sourcedocumentuipart) | `source-document` |

### Control Chunks

[Control chunks](https://github.com/vercel/ai/blob/main/packages/ai/src/ui-message-stream/ui-message-chunks.ts#L278-L293) always pass through regardless of filter/transform settings:

- `start`: Stream start marker
- `finish`: Stream finish marker
- `abort`: Stream abort marker
- `message-metadata`: Message metadata updates
- `error`: Error messages

### Step Boundary Handling

Step boundaries are handled automatically:

1. `start-step` is buffered until the first content chunk is encountered
2. If the first content chunk passes through, `start-step` is included
3. If the first content chunk is filtered out, `start-step` is also filtered out
4. `finish-step` is only included if the corresponding `start-step` was included

## API Reference

### `mapUIMessageStream`

```typescript
function mapUIMessageStream<UI_MESSAGE extends UIMessage>(
  stream: ReadableStream<UIMessageChunk>,
  mapFn: MapUIMessageStreamFn<UI_MESSAGE>,
): AsyncIterableStream<InferUIMessageChunk<UI_MESSAGE>>

type MapUIMessageStreamFn<UI_MESSAGE extends UIMessage> = (
  input: MapInput<UI_MESSAGE>,
  context: MapContext<UI_MESSAGE>,
) => InferUIMessageChunk<UI_MESSAGE> | null;

type MapInput<UI_MESSAGE extends UIMessage> = {
  chunk: InferUIMessageChunk<UI_MESSAGE>;
  part: InferPartialUIMessagePart<UI_MESSAGE>;
};

type MapContext<UI_MESSAGE extends UIMessage> = {
  index: number;
  chunks: InferUIMessageChunk<UI_MESSAGE>[];
};
```

### `flatMapUIMessageStream`

```typescript
// Without predicate - buffer all parts
function flatMapUIMessageStream<UI_MESSAGE extends UIMessage>(
  stream: ReadableStream<UIMessageChunk>,
  flatMapFn: FlatMapUIMessageStreamFn<UI_MESSAGE>,
): AsyncIterableStream<InferUIMessageChunk<UI_MESSAGE>>

// With predicate - buffer only matching parts, pass through others
function flatMapUIMessageStream<UI_MESSAGE extends UIMessage, PART extends InferUIMessagePart<UI_MESSAGE>>(
  stream: ReadableStream<UIMessageChunk>,
  predicate: FlatMapUIMessageStreamPredicate<UI_MESSAGE, PART>,
  flatMapFn: FlatMapUIMessageStreamFn<UI_MESSAGE, PART>,
): AsyncIterableStream<InferUIMessageChunk<UI_MESSAGE>>

type FlatMapUIMessageStreamFn<UI_MESSAGE extends UIMessage, PART = InferUIMessagePart<UI_MESSAGE>> = (
  input: FlatMapInput<UI_MESSAGE, PART>,
  context: FlatMapContext<UI_MESSAGE>,
) => PART | null;

type FlatMapInput<UI_MESSAGE extends UIMessage, PART = InferUIMessagePart<UI_MESSAGE>> = {
  part: PART;
};

type FlatMapContext<UI_MESSAGE extends UIMessage> = {
  index: number;
  parts: InferUIMessagePart<UI_MESSAGE>[];
};
```

#### `partTypeIs`

```typescript
function partTypeIs<UI_MESSAGE extends UIMessage, T extends InferUIMessagePartType<UI_MESSAGE>>(
  type: T | T[],
): FlatMapUIMessageStreamPredicate<UI_MESSAGE, Extract<InferUIMessagePart<UI_MESSAGE>, { type: T }>>

type FlatMapUIMessageStreamPredicate<UI_MESSAGE extends UIMessage, PART extends InferUIMessagePart<UI_MESSAGE>> = 
  (part: InferPartialUIMessagePart<UI_MESSAGE>) => boolean;
```

### `filterUIMessageStream`

```typescript
function filterUIMessageStream<UI_MESSAGE extends UIMessage>(
  stream: ReadableStream<UIMessageChunk>,
  filterFn: FilterUIMessageStreamPredicate<UI_MESSAGE>,
): AsyncIterableStream<InferUIMessageChunk<UI_MESSAGE>>

type FilterUIMessageStreamPredicate<UI_MESSAGE extends UIMessage> = (
  input: MapInput<UI_MESSAGE>,
  context: MapContext<UI_MESSAGE>,
) => boolean;
```

#### `includeParts`

```typescript
function includeParts<UI_MESSAGE extends UIMessage>(
  partTypes: Array<InferUIMessagePartType<UI_MESSAGE>>,
): FilterUIMessageStreamPredicate<UI_MESSAGE>
```

#### `excludeParts`

```typescript
function excludeParts<UI_MESSAGE extends UIMessage>(
  partTypes: Array<InferUIMessagePartType<UI_MESSAGE>>,
): FilterUIMessageStreamPredicate<UI_MESSAGE>
```
