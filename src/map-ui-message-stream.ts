import { convertAsyncIteratorToReadableStream } from '@ai-sdk/provider-utils';
import type { AsyncIterableStream, InferUIMessageChunk, UIMessage } from 'ai';
import type { InferUIMessagePart } from './types.js';
import { createAsyncIterableStream } from './utils/create-async-iterable-stream.js';
import { createUIMessageStreamReader } from './utils/create-ui-message-stream-reader.js';
import {
  isMetaChunk,
  isStepEndChunk,
  isStepStartChunk,
} from './utils/stream-utils.js';

/**
 * Input object provided to the chunk map function.
 */
export type MapInput<UI_MESSAGE extends UIMessage> = {
  /** The current chunk */
  chunk: InferUIMessageChunk<UI_MESSAGE>;
  /**
   * The assembled part this chunk belongs to (from readUIMessageStream).
   * Use `part.type` to determine the part type.
   */
  part: InferUIMessagePart<UI_MESSAGE>;
};

/**
 * Context provided to the chunk map function (similar to Array.map callback).
 */
export type MapContext<UI_MESSAGE extends UIMessage> = {
  /** The index of the current chunk in the stream (0-based) */
  index: number;
  /** All chunks seen so far (including the current one) */
  chunks: InferUIMessageChunk<UI_MESSAGE>[];
};

/**
 * Map function for chunk-level transformation.
 * Similar to Array.map, receives the input object and context.
 * Return the chunk (possibly transformed) to include it, or null to filter it out.
 */
export type MapUIMessageStreamFn<UI_MESSAGE extends UIMessage> = (
  input: MapInput<UI_MESSAGE>,
  context: MapContext<UI_MESSAGE>,
) => InferUIMessageChunk<UI_MESSAGE> | null;

/**
 * Maps/filters a UIMessageStream at the chunk level using readUIMessageStream.
 *
 * This function processes each chunk as it arrives and allows you to:
 * - Transform chunks by returning a modified chunk
 * - Filter out chunks by returning null
 *
 * Meta chunks (start, finish, abort, message-metadata, error) always pass through.
 * Step boundaries (start-step, finish-step) are handled automatically:
 * - start-step is buffered and only emitted if subsequent content is included
 * - finish-step is only emitted if the corresponding start-step was emitted
 *
 * @example
 * ```typescript
 * // Filter out reasoning chunks using part.type
 * const stream = mapUIMessageStream(
 *   inputStream,
 *   ({ chunk, part }) => part.type === 'reasoning' ? null : chunk
 * );
 *
 * // Transform text chunks
 * const stream = mapUIMessageStream(
 *   inputStream,
 *   ({ chunk, part }) => {
 *     if (chunk.type === 'text-delta') {
 *       return { ...chunk, delta: chunk.delta.toUpperCase() };
 *     }
 *     return chunk;
 *   }
 * );
 *
 * // Access previous chunks and index
 * const stream = mapUIMessageStream(
 *   inputStream,
 *   ({ chunk }, { index, chunks }) => {
 *     console.log(`Processing chunk ${index} of ${chunks.length}`);
 *     return chunk;
 *   }
 * );
 * ```
 */
export function mapUIMessageStream<UI_MESSAGE extends UIMessage>(
  stream: ReadableStream<InferUIMessageChunk<UI_MESSAGE>>,
  mapFn: MapUIMessageStreamFn<UI_MESSAGE>,
): AsyncIterableStream<InferUIMessageChunk<UI_MESSAGE>> {
  // State for step boundary handling
  let bufferedStartStep: InferUIMessageChunk<UI_MESSAGE> | undefined;
  let stepStartEmitted = false;

  // Track all chunks and current index for context
  const allChunks: InferUIMessageChunk<UI_MESSAGE>[] = [];
  let currentIndex = 0;

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
   * Main processing generator.
   */
  async function* processChunks(): AsyncGenerator<
    InferUIMessageChunk<UI_MESSAGE>
  > {
    for await (const {
      chunk,
      message,
    } of createUIMessageStreamReader<UI_MESSAGE>(stream)) {
      // Track chunks for context
      allChunks.push(chunk);
      const index = currentIndex++;

      // Meta chunks pass through immediately
      if (isMetaChunk(chunk)) {
        yield chunk;
        continue;
      }

      // Step boundaries - special handling
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

      // Get the current part from AI SDK (last part)
      const currentPart = message.parts[message.parts.length - 1]!;

      // Apply map function
      const result = mapFn(
        {
          chunk,
          part: currentPart,
        },
        { index, chunks: allChunks },
      );

      // If result is not null, emit with step handling
      if (result !== null) {
        yield* emitChunks(result);
      }
    }
  }

  const outputStream = convertAsyncIteratorToReadableStream(processChunks());
  return createAsyncIterableStream(outputStream);
}
