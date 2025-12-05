import { convertAsyncIteratorToReadableStream } from '@ai-sdk/provider-utils';
import type { AsyncIterableStream, InferUIMessageChunk, UIMessage } from 'ai';
import type { InferUIMessagePart, InferUIMessagePartType } from './types.js';
import { createAsyncIterableStream } from './utils/create-async-iterable-stream.js';
import { createUIMessageStreamReader } from './utils/create-ui-message-stream-reader.js';
import { serializePartToChunks } from './utils/serialize-part-to-chunks.js';
import {
  isMetaChunk,
  isPartComplete,
  isStepEndChunk,
  isStepStartChunk,
} from './utils/stream-utils.js';

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
 * Receives the part from readUIMessageStream (may still be streaming).
 * Returns true to buffer the part for transformation, false to pass through immediately.
 */
export type FlatMapUIMessageStreamPredicate<
  UI_MESSAGE extends UIMessage,
  PART extends InferUIMessagePart<UI_MESSAGE> = InferUIMessagePart<UI_MESSAGE>,
> = (part: InferUIMessagePart<UI_MESSAGE>) => boolean;

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
  return (part: InferUIMessagePart<UI_MESSAGE>): boolean =>
    partTypes.includes(part.type as unknown as PART_TYPE);
}

/**
 * FlatMaps a UIMessageStream at the part level using readUIMessageStream.
 *
 * This function buffers all chunks for a part until it's complete, then allows
 * you to transform the complete part. This is useful when you need access to
 * the full part content before deciding how to transform it.
 *
 * When a predicate is provided (e.g., `partTypeIs('text')`), only matching parts
 * are buffered for transformation. Non-matching parts stream through immediately
 * without buffering, preserving real-time streaming behavior.
 *
 * Meta chunks (start, finish, abort, message-metadata, error) always pass through.
 * Step boundaries (start-step, finish-step) are handled automatically:
 * - start-step is buffered and only emitted if subsequent content is included
 * - finish-step is only emitted if the corresponding start-step was emitted
 *
 * @example
 * ```typescript
 * // Filter out reasoning parts
 * const stream = flatMapUIMessageStream(
 *   inputStream,
 *   ({ part }) => part.type === 'reasoning' ? null : part
 * );
 *
 * // Transform text content
 * const stream = flatMapUIMessageStream(
 *   inputStream,
 *   ({ part }) => {
 *     if (part.type === 'text') {
 *       return { ...part, text: part.text.toUpperCase() };
 *     }
 *     return part;
 *   }
 * );
 *
 * // Buffer only specific parts, pass through others immediately
 * const stream = flatMapUIMessageStream(
 *   inputStream,
 *   partTypeIs('text'),
 *   ({ part }) => ({ ...part, text: part.text.toUpperCase() })
 * );
 *
 * // Access previous parts and index
 * const stream = flatMapUIMessageStream(
 *   inputStream,
 *   ({ part }, { index, parts }) => {
 *     console.log(`Processing part ${index}, previous parts: ${parts.length - 1}`);
 *     return part;
 *   }
 * );
 * ```
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

  // State for tracking parts
  let lastPartCount = 0;
  let isBufferingCurrentPart = false;
  let isStreamingCurrentPart = false;
  let bufferedChunks: InferUIMessageChunk<UI_MESSAGE>[] = [];
  const allParts: InferUIMessagePart<UI_MESSAGE>[] = [];

  // State for step boundary handling
  let bufferedStartStep: InferUIMessageChunk<UI_MESSAGE> | undefined;
  let stepStartEmitted = false;

  /**
   * Generator that yields chunks with step boundary handling.
   */
  async function* emitChunks(
    chunk: InferUIMessageChunk<UI_MESSAGE>,
  ): AsyncGenerator<InferUIMessageChunk<UI_MESSAGE>> {
    if (bufferedStartStep) {
      yield bufferedStartStep;
      stepStartEmitted = true;
      bufferedStartStep = undefined;
    }
    yield chunk;
  }

  /**
   * Flush buffered part: apply flatMapFn and yield chunks.
   * Always uses serializePartToChunks which includes step boundaries.
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
      // Always use serializePartToChunks - it includes step boundaries
      const chunksToEmit = serializePartToChunks(result, bufferedChunks);

      for (const chunk of chunksToEmit) {
        yield chunk;
      }
    }

    // Clear step state - serializePartToChunks provides its own boundaries,
    // so we suppress the input stream's finish-step
    bufferedStartStep = undefined;
    stepStartEmitted = false;
    bufferedChunks = [];
  }

  /**
   * Main processing generator.
   */
  async function* processChunks(): AsyncGenerator<
    InferUIMessageChunk<UI_MESSAGE>
  > {
    for await (const {
      chunk,
      message,
    } of createUIMessageStreamReader<UI_MESSAGE>(inputStream)) {
      // Handle meta chunks - pass through immediately
      if (isMetaChunk(chunk)) {
        yield chunk;
        continue;
      }

      // Handle step boundaries specially
      if (isStepStartChunk(chunk)) {
        bufferedStartStep = chunk;
        continue;
      }

      if (isStepEndChunk(chunk)) {
        if (stepStartEmitted) {
          yield chunk;
          stepStartEmitted = false;
        }
        bufferedStartStep = undefined;
        continue;
      }

      // Content chunks - message should always be defined here
      if (!message) {
        break;
      }

      // Get the current part (last part)
      const currentPart = message.parts[message.parts.length - 1]!;

      // Detect new part (part count increased)
      if (message.parts.length > lastPartCount) {
        // Check predicate on the part from AI SDK
        const shouldBuffer = !predicate || predicate(currentPart);

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
  }

  const outputStream = convertAsyncIteratorToReadableStream(processChunks());
  return createAsyncIterableStream(outputStream);
}
