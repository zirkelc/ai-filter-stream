import { convertAsyncIteratorToReadableStream } from '@ai-sdk/provider-utils';
import type {
  AsyncIterableStream,
  FileUIPart,
  InferUIMessageChunk,
  ReasoningUIPart,
  SourceDocumentUIPart,
  SourceUrlUIPart,
  TextUIPart,
  UIMessage,
} from 'ai';
import {
  getToolOrDynamicToolName,
  isDataUIPart,
  isToolOrDynamicToolUIPart,
} from 'ai';
import { createAsyncIterableStream } from './create-async-iterable-stream.js';
import {
  isMetaChunk,
  isStepEndChunk,
  isStepStartChunk,
} from './stream-utils.js';
import type {
  InferPartialUIMessagePart,
  InferUIMessagePart,
  InferUIMessagePartType,
} from './types.js';
import { createUIMessageStreamReader } from './ui-message-stream-reader.js';

/**
 * Input object provided to the part flatMap function.
 */
export type FlatMapInput<
  UI_MESSAGE extends UIMessage,
  PART extends InferUIMessagePart<UI_MESSAGE> = InferUIMessagePart<UI_MESSAGE>,
> = {
  /** The reconstructed part */
  part: PART;
};

/**
 * Context provided to the part flatMap function.
 */
export type FlatMapContext<UI_MESSAGE extends UIMessage> = {
  /** The index of the current part in the stream (0-based) */
  index: number;
  /** All parts seen so far (including the current one) */
  parts: InferUIMessagePart<UI_MESSAGE>[];
};

/**
 * FlatMap function for part-level transformation.
 * Return:
 * - The part (possibly transformed) to include it
 * - null to filter out the part
 */
export type FlatMapUIMessageStreamFn<
  UI_MESSAGE extends UIMessage,
  PART extends InferUIMessagePart<UI_MESSAGE> = InferUIMessagePart<UI_MESSAGE>,
> = (
  input: FlatMapInput<UI_MESSAGE, PART>,
  context: FlatMapContext<UI_MESSAGE>,
) => PART | null;

/**
 * Predicate function to determine which parts should be buffered.
 * Receives the partial part from readUIMessageStream.
 * Returns true to buffer the part for transformation, false to pass through immediately.
 */
export type FlatMapUIMessageStreamPredicate<
  UI_MESSAGE extends UIMessage,
  PART extends InferUIMessagePart<UI_MESSAGE> = InferUIMessagePart<UI_MESSAGE>,
> = (part: InferPartialUIMessagePart<UI_MESSAGE>) => boolean;

/**
 * Creates a predicate that matches parts by their type.
 */
export function partTypeIs<
  UI_MESSAGE extends UIMessage,
  PART_TYPE extends InferUIMessagePartType<UI_MESSAGE>,
>(
  type: PART_TYPE | PART_TYPE[],
): FlatMapUIMessageStreamPredicate<
  UI_MESSAGE,
  Extract<InferUIMessagePart<UI_MESSAGE>, { type: PART_TYPE }>
> {
  const partTypes = Array.isArray(type) ? type : [type];
  // Cast through unknown since part.type may not exactly overlap with PART_TYPE
  return (part: InferPartialUIMessagePart<UI_MESSAGE>): boolean =>
    partTypes.includes(part.type as unknown as PART_TYPE);
}

/**
 * Checks if a part is complete based on its state.
 */
function isPartComplete<UI_MESSAGE extends UIMessage>(
  part: InferUIMessagePart<UI_MESSAGE>,
): boolean {
  if (part.type === 'step-start') return false;
  if (!('state' in part)) return true; // Single-chunk parts (file, source-url, etc.)
  return (
    part.state === 'done' ||
    part.state === 'output-available' ||
    part.state === 'output-error'
  );
}

/**
 * FlatMaps a UIMessageStream at the part level using readUIMessageStream.
 *
 * This implementation uses:
 * - readUIMessageStream for both part assembly AND predicate checking
 * - An async generator for cleaner control flow and natural backpressure
 * - convertAsyncIteratorToReadableStream for stream conversion
 *
 * No manual tool state tracking or partial part building needed.
 */
export function flatMapUIMessageStream<
  UI_MESSAGE extends UIMessage,
  PART extends InferUIMessagePart<UI_MESSAGE>,
>(
  stream: ReadableStream<InferUIMessageChunk<UI_MESSAGE>>,
  predicate: FlatMapUIMessageStreamPredicate<UI_MESSAGE, PART>,
  flatMapFn: FlatMapUIMessageStreamFn<UI_MESSAGE, PART>,
): AsyncIterableStream<InferUIMessageChunk<UI_MESSAGE>>;
export function flatMapUIMessageStream<UI_MESSAGE extends UIMessage>(
  stream: ReadableStream<InferUIMessageChunk<UI_MESSAGE>>,
  flatMapFn: FlatMapUIMessageStreamFn<UI_MESSAGE>,
): AsyncIterableStream<InferUIMessageChunk<UI_MESSAGE>>;

// Implementation
export function flatMapUIMessageStream<
  UI_MESSAGE extends UIMessage,
  PART extends InferUIMessagePart<UI_MESSAGE>,
>(
  ...args:
    | [
        ReadableStream<InferUIMessageChunk<UI_MESSAGE>>,
        FlatMapUIMessageStreamFn<UI_MESSAGE, PART>,
      ]
    | [
        ReadableStream<InferUIMessageChunk<UI_MESSAGE>>,
        FlatMapUIMessageStreamPredicate<UI_MESSAGE, PART>,
        FlatMapUIMessageStreamFn<UI_MESSAGE, PART>,
      ]
): AsyncIterableStream<InferUIMessageChunk<UI_MESSAGE>> {
  const [inputStream, predicate, flatMapFn] =
    args.length === 2
      ? [args[0], undefined, args[1]]
      : [args[0], args[1], args[2]];

  const streamReader = createUIMessageStreamReader<UI_MESSAGE>(inputStream);

  // State for tracking parts
  let lastPartCount = 0;
  let isBufferingCurrentPart = false;
  let isStreamingCurrentPart = false;
  let bufferedChunks: InferUIMessageChunk<UI_MESSAGE>[] = [];
  const allParts: InferUIMessagePart<UI_MESSAGE>[] = [];

  // State for step boundary handling
  let bufferedStartStep: InferUIMessageChunk<UI_MESSAGE> | undefined;
  let stepStartEmitted = false;
  let stepHasContent = false;

  /**
   * Generator that yields chunks with step boundary handling.
   */
  async function* emitChunks(
    chunk: InferUIMessageChunk<UI_MESSAGE>,
  ): AsyncGenerator<InferUIMessageChunk<UI_MESSAGE>> {
    if (bufferedStartStep && !stepHasContent) {
      stepHasContent = true;
      yield bufferedStartStep;
      stepStartEmitted = true;
      bufferedStartStep = undefined;
    }
    yield chunk;
  }

  /**
   * Flush buffered part: apply flatMapFn and yield chunks.
   */
  async function* flushBufferedPart(
    completedPart: InferUIMessagePart<UI_MESSAGE>,
  ): AsyncGenerator<InferUIMessageChunk<UI_MESSAGE>> {
    isBufferingCurrentPart = false;
    allParts.push(completedPart);

    const result = flatMapFn(
      { part: completedPart as PART },
      { index: allParts.length - 1, parts: allParts },
    );

    if (result !== null) {
      const chunksToEmit =
        result === completedPart
          ? bufferedChunks
          : serializePartToChunks(result, bufferedChunks);

      for (const chunk of chunksToEmit) {
        yield* emitChunks(chunk);
      }
    }

    bufferedChunks = [];
  }

  /**
   * Main processing generator.
   */
  async function* processChunks(): AsyncGenerator<
    InferUIMessageChunk<UI_MESSAGE>
  > {
    try {
      while (true) {
        const { done, value: chunk } = await streamReader.read();
        if (done) {
          streamReader.close();
          break;
        }

        // Handle meta chunks - pass through immediately
        if (isMetaChunk(chunk)) {
          yield chunk;
          await streamReader.enqueue(chunk);
          continue;
        }

        // Handle step boundaries specially
        if (isStepStartChunk(chunk)) {
          bufferedStartStep = chunk;
          stepHasContent = false;
          await streamReader.enqueue(chunk);
          continue;
        }

        if (isStepEndChunk(chunk)) {
          if (stepStartEmitted) {
            yield chunk;
            stepStartEmitted = false;
          }
          bufferedStartStep = undefined;
          await streamReader.enqueue(chunk);
          continue;
        }

        // For content chunks: feed to stream reader and get updated message
        const message = await streamReader.enqueue(chunk);

        if (!message) {
          break;
        }

        // Get the current part (last part)
        const currentPart = message.parts[
          message.parts.length - 1
        ] as InferUIMessagePart<UI_MESSAGE>;

        // Detect new part (part count increased)
        if (message.parts.length > lastPartCount) {
          // Check predicate on the partial part from AI SDK
          const shouldBuffer =
            !predicate ||
            predicate(currentPart as InferPartialUIMessagePart<UI_MESSAGE>);

          if (shouldBuffer) {
            isBufferingCurrentPart = true;
            isStreamingCurrentPart = false;
            bufferedChunks = [chunk];

            // Single-chunk parts are complete immediately
            if (isPartComplete(currentPart)) {
              yield* flushBufferedPart(currentPart);
            }
          } else {
            isBufferingCurrentPart = false;
            isStreamingCurrentPart = true;
            yield* emitChunks(chunk);

            // Single-chunk parts complete immediately
            if (isPartComplete(currentPart)) {
              isStreamingCurrentPart = false;
              allParts.push(currentPart); // Track for context
            }
          }

          lastPartCount = message.parts.length;
        } else if (isBufferingCurrentPart) {
          // Continue buffering current part
          bufferedChunks.push(chunk);

          if (isPartComplete(currentPart)) {
            yield* flushBufferedPart(currentPart);
          }
        } else if (isStreamingCurrentPart) {
          // Continue streaming current part
          yield* emitChunks(chunk);

          if (isPartComplete(currentPart)) {
            isStreamingCurrentPart = false;
            allParts.push(currentPart);
          }
        }
      }
    } finally {
      await streamReader.release();
    }
  }

  const outputStream = convertAsyncIteratorToReadableStream(processChunks());
  return createAsyncIterableStream(outputStream);
}

/**
 * Extracts the part ID from a list of chunks belonging to the same part.
 */
function getPartId<UI_MESSAGE extends UIMessage>(
  chunks: InferUIMessageChunk<UI_MESSAGE>[],
): string {
  const chunk = chunks[0];
  if (!chunk) return 'unknown';
  if ('id' in chunk && chunk.id) return chunk.id;
  if ('toolCallId' in chunk && chunk.toolCallId) return chunk.toolCallId;
  return 'unknown';
}

/**
 * Serializes a UIMessagePart back to chunks.
 */
function serializePartToChunks<UI_MESSAGE extends UIMessage>(
  part: InferUIMessagePart<UI_MESSAGE>,
  originalChunks: InferUIMessageChunk<UI_MESSAGE>[],
): InferUIMessageChunk<UI_MESSAGE>[] {
  if (part.type === 'file') {
    const filePart = part as FileUIPart;
    return [
      {
        type: 'file',
        mediaType: filePart.mediaType,
        url: filePart.url,
        providerMetadata: filePart.providerMetadata,
      } as InferUIMessageChunk<UI_MESSAGE>,
    ];
  }

  if (part.type === 'source-url') {
    const sourceUrlPart = part as SourceUrlUIPart;
    return [
      {
        type: 'source-url',
        sourceId: sourceUrlPart.sourceId,
        url: sourceUrlPart.url,
        title: sourceUrlPart.title,
        providerMetadata: sourceUrlPart.providerMetadata,
      } as InferUIMessageChunk<UI_MESSAGE>,
    ];
  }

  if (part.type === 'source-document') {
    const sourceDocumentPart = part as SourceDocumentUIPart;
    return [
      {
        type: 'source-document',
        sourceId: sourceDocumentPart.sourceId,
        mediaType: sourceDocumentPart.mediaType,
        title: sourceDocumentPart.title,
        filename: sourceDocumentPart.filename,
        providerMetadata: sourceDocumentPart.providerMetadata,
      } as InferUIMessageChunk<UI_MESSAGE>,
    ];
  }

  if (isDataUIPart(part)) {
    return [
      { type: part.type, data: part.data } as InferUIMessageChunk<UI_MESSAGE>,
    ];
  }

  const id = getPartId(originalChunks);

  if (part.type === 'text') {
    const textPart = part as TextUIPart;
    return [
      { type: 'text-start', id, providerMetadata: textPart.providerMetadata },
      { type: 'text-delta', id, delta: textPart.text },
      { type: 'text-end', id, providerMetadata: textPart.providerMetadata },
    ] as InferUIMessageChunk<UI_MESSAGE>[];
  }

  if (part.type === 'reasoning') {
    const reasoningPart = part as ReasoningUIPart;
    return [
      {
        type: 'reasoning-start',
        id,
        providerMetadata: reasoningPart.providerMetadata,
      },
      { type: 'reasoning-delta', id, delta: reasoningPart.text },
      {
        type: 'reasoning-end',
        id,
        providerMetadata: reasoningPart.providerMetadata,
      },
    ] as InferUIMessageChunk<UI_MESSAGE>[];
  }

  if (isToolOrDynamicToolUIPart(part)) {
    const dynamic = part.type === 'dynamic-tool';

    const chunks: InferUIMessageChunk<UI_MESSAGE>[] = [
      {
        type: 'tool-input-start',
        toolCallId: part.toolCallId,
        toolName: getToolOrDynamicToolName(part),
        dynamic,
        providerExecuted: part.providerExecuted,
      } as InferUIMessageChunk<UI_MESSAGE>,
    ];

    if (part.state === 'input-available' || part.state === 'output-available') {
      chunks.push({
        type: 'tool-input-available',
        toolCallId: part.toolCallId,
        toolName: getToolOrDynamicToolName(part),
        input: part.input,
        dynamic,
        providerExecuted: part.providerExecuted,
        providerMetadata: part.callProviderMetadata,
      } as InferUIMessageChunk<UI_MESSAGE>);
    }

    if (part.state === 'output-available') {
      chunks.push({
        type: 'tool-output-available',
        toolCallId: part.toolCallId,
        output: part.output,
        dynamic,
        providerExecuted: part.providerExecuted,
        preliminary: part.preliminary,
      } as InferUIMessageChunk<UI_MESSAGE>);
    } else if (part.state === 'output-error') {
      chunks.push({
        type: 'tool-output-error',
        toolCallId: part.toolCallId,
        errorText: part.errorText,
        dynamic,
        providerExecuted: part.providerExecuted,
      } as InferUIMessageChunk<UI_MESSAGE>);
    }

    return chunks;
  }

  return originalChunks;
}
