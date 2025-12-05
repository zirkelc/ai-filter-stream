import type { InferUIMessageChunk, UIMessage } from 'ai';
import { readUIMessageStream } from 'ai';
import {
  isMetaChunk,
  isStepEndChunk,
  isStepStartChunk,
} from './stream-utils.js';

/**
 * Value yielded by the UIMessageStreamReader async generator.
 */
export type UIMessageStreamReaderValue<UI_MESSAGE extends UIMessage> = {
  /** The current chunk from the input stream */
  chunk: InferUIMessageChunk<UI_MESSAGE>;
  /**
   * The assembled message with updated parts.
   * Undefined for meta chunks (start, finish, error, abort, message-metadata)
   * and step chunks (start-step, finish-step) since they don't produce messages.
   */
  message: UI_MESSAGE | undefined;
};

/**
 * Creates an async generator that wraps a UIMessageStream with readUIMessageStream.
 *
 * This helper encapsulates the common pattern of:
 * 1. Reading chunks from an input stream
 * 2. Feeding them to readUIMessageStream for part assembly
 * 3. Yielding both the chunk and the assembled message
 *
 * @example
 * ```typescript
 * for await (const { chunk, message } of createUIMessageStreamReader<UIMessage>(inputStream)) {
 *   if (message) {
 *     const currentPart = message.parts[message.parts.length - 1];
 *     // ... process part
 *   }
 * }
 * ```
 */
export async function* createUIMessageStreamReader<
  UI_MESSAGE extends UIMessage,
>(
  inputStream: ReadableStream<InferUIMessageChunk<UI_MESSAGE>>,
): AsyncGenerator<UIMessageStreamReaderValue<UI_MESSAGE>> {
  // Reader for the input stream - used to read chunks one at a time
  const inputReader = inputStream.getReader();

  // Internal stream that we control - chunks are enqueued here and
  // readUIMessageStream consumes them to assemble parts into messages
  let controller!: ReadableStreamDefaultController<
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

  try {
    while (true) {
      const { done, value: chunk } = await inputReader.read();
      if (done) {
        controller.close();
        break;
      }

      // Feed chunk to the internal stream for readUIMessageStream to process
      controller.enqueue(chunk);

      // Meta chunks (start, finish, error, etc.) and step chunks (start-step, finish-step)
      // don't emit messages in readUIMessageStream. Calling iterator.next() for these
      // would block waiting for the next content chunk that does emit.
      let message: UI_MESSAGE | undefined;
      if (
        !isMetaChunk(chunk) &&
        !isStepStartChunk(chunk) &&
        !isStepEndChunk(chunk)
      ) {
        // For content chunks, get the updated message with assembled parts
        const result = await iterator.next();
        message = result.done ? undefined : result.value;
      }

      yield { chunk, message };
    }
  } finally {
    // Drain any remaining messages from the iterator
    while (true) {
      const { done } = await iterator.next();
      if (done) break;
    }
    // Release the reader lock on the input stream
    inputReader.releaseLock();
  }
}
