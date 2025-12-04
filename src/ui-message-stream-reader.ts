import type { InferUIMessageChunk, UIMessage } from 'ai';
import { readUIMessageStream } from 'ai';
import {
  isMetaChunk,
  isStepEndChunk,
  isStepStartChunk,
} from './stream-utils.js';

/**
 * A helper that wraps a UIMessageStream with readUIMessageStream for part assembly.
 * Provides methods to read chunks, enqueue them for processing, and get assembled messages.
 */
export type UIMessageStreamReader<UI_MESSAGE extends UIMessage> = {
  /** Read the next chunk from the input stream */
  read: ReadableStreamDefaultReader<InferUIMessageChunk<UI_MESSAGE>>['read'];
  /**
   * Enqueue a chunk to the internal stream and get the updated message.
   * For meta/step chunks, returns undefined since they don't emit messages.
   * For content chunks, returns the updated message with assembled parts.
   */
  enqueue: (
    chunk: InferUIMessageChunk<UI_MESSAGE>,
  ) => Promise<UI_MESSAGE | undefined>;
  /** Close the internal stream - call when input stream is done */
  close: () => void;
  /** Signal an error on the internal stream */
  error: (err: unknown) => void;
  /** Drain remaining messages and release the reader lock - call in finally block */
  release: () => Promise<void>;
};

/**
 * Creates a UIMessageStreamReader that wraps a stream with readUIMessageStream.
 *
 * This helper encapsulates the common pattern of:
 * 1. Creating a reader for the input stream
 * 2. Setting up an internal stream for readUIMessageStream
 * 3. Feeding chunks to readUIMessageStream for part assembly
 *
 * @example
 * ```typescript
 * const streamReader = createUIMessageStreamReader<UIMessage>(inputStream);
 *
 * try {
 *   while (true) {
 *     const { done, value: chunk } = await streamReader.read();
 *     if (done) {
 *       streamReader.close();
 *       break;
 *     }
 *
 *     // Feed chunk and get assembled message (undefined for meta/step chunks)
 *     const message = await streamReader.enqueue(chunk);
 *     if (message) {
 *       const currentPart = message.parts[message.parts.length - 1];
 *       // ... process part
 *     }
 *   }
 * } finally {
 *   // Drain remaining messages and release the reader lock
 *   await streamReader.release();
 * }
 * ```
 */
export function createUIMessageStreamReader<UI_MESSAGE extends UIMessage>(
  stream: ReadableStream<InferUIMessageChunk<UI_MESSAGE>>,
): UIMessageStreamReader<UI_MESSAGE> {
  // Reader for the input stream - used to read chunks one at a time
  const reader = stream.getReader();

  // Internal stream that we control - chunks are enqueued here and
  // readUIMessageStream consumes them to assemble parts into messages
  let controller: ReadableStreamDefaultController<
    InferUIMessageChunk<UI_MESSAGE>
  >;
  const internalStream = new ReadableStream<InferUIMessageChunk<UI_MESSAGE>>({
    start(c) {
      controller = c;
    },
  });

  // readUIMessageStream returns an async iterable of assembled messages.
  // Each time we enqueue a content chunk and call iterator.next(),
  // we get back the updated message with the current part state.
  const uiMessages = readUIMessageStream<UI_MESSAGE>({
    stream: internalStream,
  });
  const iterator = uiMessages[Symbol.asyncIterator]();

  return {
    // Read the next chunk from the input stream
    read() {
      return reader.read();
    },

    // Enqueue a chunk for processing and return the assembled message.
    // For content chunks, this returns the message with updated parts.
    // For meta/step chunks, returns undefined (they don't produce messages).
    async enqueue(chunk: InferUIMessageChunk<UI_MESSAGE>) {
      // Feed chunk to the internal stream for readUIMessageStream to process
      controller.enqueue(chunk);

      // Meta chunks (start, finish, error, etc.) and step chunks (start-step, finish-step)
      // don't emit messages in readUIMessageStream. Calling iterator.next() for these
      // would block waiting for the next content chunk that does emit.
      // We only need messages for content chunks to access the assembled parts.
      if (
        isMetaChunk(chunk) ||
        isStepStartChunk(chunk) ||
        isStepEndChunk(chunk)
      ) {
        return undefined;
      }

      // For content chunks, get the updated message with assembled parts
      const { done, value } = await iterator.next();
      return done ? undefined : value;
    },

    // Close the internal stream - call when input stream is done
    close() {
      controller.close();
    },

    // Signal an error on the internal stream
    error(err: unknown) {
      controller.error(err);
    },

    // Drain remaining messages and release the reader lock.
    // Call this in the finally block to ensure clean shutdown.
    async release() {
      // Drain any remaining messages from the iterator
      while (true) {
        const { done } = await iterator.next();
        if (done) break;
      }
      // Release the reader lock on the input stream
      reader.releaseLock();
    },
  };
}
