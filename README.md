<div align='center'>

# ai-filter-stream

<p align="center">Filter UI message streams to the client</p>
<p align="center">
  <a href="https://www.npmjs.com/package/ai-filter-stream" alt="ai-filter-stream"><img src="https://img.shields.io/npm/dt/ai-filter-stream?label=ai-filter-stream"></a> <a href="https://github.com/zirkelc/ai-filter-stream/actions/workflows/ci.yml" alt="CI"><img src="https://img.shields.io/github/actions/workflow/status/zirkelc/ai-filter-stream/ci.yml?branch=main"></a>
</p>

</div>

This library allows you filter UI message chunks returned from [`streamText()`](https://ai-sdk.dev/docs/reference/ai-sdk-core/stream-text) by their corresponding UI message part type. 

### Why?

The AI SDK UI message stream created by [`toUIMessageStream()`](https://ai-sdk.dev/docs/reference/ai-sdk-core/stream-text#to-ui-message-stream) streams all parts (text, tools, data, etc.) to the client. 
Tool calls, like database queries often contain large amounts of data or sensitive information that should not be visible on the client. 
This library provides a type-safe filter to apply selective streaming of certain message parts.

### Installation

This library only supports AI SDK v5.

```bash
npm install ai-filter-stream
```

### Usage

Use the `filterUIMessageStream` function to wrap the UI message stream from [`result.toUIMessageStream()`](https://ai-sdk.dev/docs/reference/ai-sdk-core/stream-text#to-ui-message-stream) and provide a filter to include or exclude certain UI message parts:

> [!NOTE]  
> Providing a [`MyUIMessage`](https://ai-sdk.dev/docs/reference/ai-sdk-core/ui-message#creating-your-own-uimessage-type) type to `filterUIMessageStream<MyMessage>()` is optional and only required for type-safety so that the part type is inferred based on your tools and data parts.

```typescript
import { streamText } from 'ai';
import { filterUIMessageStream } from 'ai-filter-stream';
import type { UIMessage, InferUITools } from 'ai';

type MyUIMessageMetadata = {};

type MyDataPart = {};

type MyTools = InferUITools<typeof tools>;

// Define your UI message type for type safety
// See: https://ai-sdk.dev/docs/reference/ai-sdk-core/ui-message
type MyUIMessage = UIMessage<
  MyUIMessageMetadata, // or unknown
  MyDataPart, // or unknown
  MyTools,
>;

const tools = {
  weather: tool({
    description: 'Get the weather in a location',
    inputSchema: z.object({
      location: z.string().describe('The location to get the weather for'),
    }),
    execute: ({ location }) => ({
      location,
      temperature: 72 + Math.floor(Math.random() * 21) - 10,
    }),
  }),
};

const result = streamText({
  model,
  prompt: 'What is the weather in Tokyo?',
  tools,
});

// Inclusive filtering: include only `text` parts
const stream = filterUIMessageStream<MyMessage>(result.toUIMessageStream(), {
  includeParts: ['text'], // Autocomplete works here!
});

// Exclusive filtering: exclude only `tool-weather` parts
const stream = filterUIMessageStream<MyMessage>(result.toUIMessageStream(), {
  excludeParts: ['reasoning', 'tool-calculator'], // Autocomplete works here!
});

// Dynamic filtering: apply filter function for each chunk
const stream = filterUIMessageStream<MyMessage>(result.toUIMessageStream(), {
  filterParts: ({ partType }) => {
    // Always include text
    if (partType === 'text') return true;

    // Only include tools that start with 'weather'
    if (partType.startsWith('tool-weather')) return true;

    // Exclude everything else
    return false;
  },
});
```

## Part Type Mapping

The filter operates on [UIMessagePart](https://ai-sdk.dev/docs/reference/ai-sdk-core/ui-message#uimessagepart-types) part types, which are derived from [UIMessageChunk](https://github.com/vercel/ai/blob/main/packages/ai/src/ui-message-stream/ui-message-chunks.ts) chunk types:

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

Controls chunks are always passed through regardless of filter settings:

- `start`: Stream start marker
- `finish`: Stream finish marker
- `abort`: Stream abort marker
- `message-metadata`: Message metadata updates
- `error`: Error messages

### Start-Step Filtering

The filter automatically handles step boundaries, that means a [`start-step`](https://ai-sdk.dev/docs/reference/ai-sdk-core/ui-message#stepstartuipart) is only emitted if the actual content is not filtered:

1. `start-step` is buffered until the first content chunk is encountered
2. If the first content chunk passes the filter, `start-step` is included
3. If the first content chunk is filtered out, `start-step` is also filtered out
4. `finish-step` is only included if the corresponding `start-step` was included


## Type Safety

The [`toUIMessageStream()`](https://ai-sdk.dev/docs/reference/ai-sdk-core/stream-text#to-ui-message-stream) from [`streamText()`](https://ai-sdk.dev/docs/reference/ai-sdk-core/stream-text) returns a generic stream `ReadableStream<UIMessageChunk>` which means that the original `UIMessage` cannot be inferred automatically. To enable autocomplete and type-safety for filtering parts by type, we need to pass our own [`UIMessage`](https://ai-sdk.dev/docs/reference/ai-sdk-core/ui-message#creating-your-own-uimessage-type) as generic param to `filterUIMessageStream()`:

```typescript
type MyMessage = UIMessage<MyMetadata, MyData, MyTools>;

const stream = filterUIMessageStream<MyMessage>(
  result.toUIMessageStream(), // returns generic ReadableStream<UIMessageChunk>
  {
    includeParts: ['text', 'tool-weather'] }, // type-safe through MyMessage
  }
);
```

See [UIMessage](https://ai-sdk.dev/docs/reference/ai-sdk-core/ui-message) in the AI SDK docs to create your own `UIMessage` type.

## API Reference

### `filterUIMessageStream`

```typescript
function filterUIMessageStream<UI_MESSAGE extends UIMessage>(
  stream: ReadableStream<UIMessageChunk>,
  options: FilterUIMessageStreamOptions<UI_MESSAGE>,
): AsyncIterableStream<InferUIMessageChunk<UI_MESSAGE>>
```

### `FilterUIMessageStreamOptions`

```typescript
type FilterUIMessageStreamOptions<UI_MESSAGE extends UIMessage> =
  | {
      filterParts: (options: { partType: InferUIMessagePartType<UI_MESSAGE> }) => boolean;
    }
  | {
      includeParts: Array<InferUIMessagePartType<UI_MESSAGE>>;
    }
  | {
      excludeParts: Array<InferUIMessagePartType<UI_MESSAGE>>;
    };
```
