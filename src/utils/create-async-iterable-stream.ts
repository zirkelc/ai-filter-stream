import type { AsyncIterableStream } from 'ai';

// Copied from https://github.com/vercel/ai/blob/main/packages/ai/src/util/async-iterable-stream.ts
export function createAsyncIterableStream<T>(
  source: ReadableStream<T>,
): AsyncIterableStream<T> {
  // Pipe through a TransformStream to ensure a fresh, unlocked stream.
  const stream = source.pipeThrough(new TransformStream<T, T>());

  /**
   * Implements the async iterator protocol for the stream.
   * Ensures proper cleanup (cancelling and releasing the reader) on completion, early exit, or error.
   */
  // @ts-expect-error
  (stream as AsyncIterableStream<T>)[Symbol.asyncIterator] = function (
    this: ReadableStream<T>,
  ): AsyncIterator<T> {
    const reader = this.getReader();

    let finished = false;

    /**
     * Cleans up the reader by cancelling and releasing the lock.
     */
    async function cleanup(cancelStream: boolean) {
      finished = true;
      try {
        if (cancelStream) {
          await reader.cancel?.();
        }
      } finally {
        try {
          reader.releaseLock();
        } catch {}
      }
    }

    return {
      /**
       * Reads the next chunk from the stream.
       * @returns A promise resolving to the next IteratorResult.
       */
      async next(): Promise<IteratorResult<T>> {
        if (finished) {
          return { done: true, value: undefined };
        }

        const { done, value } = await reader.read();

        if (done) {
          await cleanup(true);
          return { done: true, value: undefined };
        }

        return { done: false, value };
      },

      /**
       * Called on early exit (e.g., break from for-await).
       * Ensures the stream is cancelled and resources are released.
       * @returns A promise resolving to a completed IteratorResult.
       */
      async return(): Promise<IteratorResult<T>> {
        await cleanup(true);
        return { done: true, value: undefined };
      },

      /**
       * Called on early exit with error.
       * Ensures the stream is cancelled and resources are released, then rethrows the error.
       * @param err The error to throw.
       * @returns A promise that rejects with the provided error.
       */
      async throw(err: unknown): Promise<IteratorResult<T>> {
        await cleanup(true);
        throw err;
      },
    };
  };

  return stream as AsyncIterableStream<T>;
}
