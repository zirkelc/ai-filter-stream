import {
  convertArrayToReadableStream,
  convertAsyncIterableToArray,
} from '@ai-sdk/provider-utils/test';
import type { UIMessageChunk } from 'ai';
import { describe, expect, it } from 'vitest';
import { mapUIMessageStream } from './map-ui-message-stream.js';
import {
  ABORT_CHUNK,
  ERROR_CHUNK,
  FILE_CHUNKS,
  FINISH_CHUNK,
  MESSAGE_METADATA_CHUNK,
  REASONING_CHUNKS,
  START_CHUNK,
  TEXT_CHUNKS,
  TOOL_CHUNKS,
} from './utils/test-utils.js';

describe('mapUIMessageStream', () => {
  it('should pass through all chunks with identity map', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      ...TEXT_CHUNKS,
      FINISH_CHUNK,
    ]);

    const mappedStream = mapUIMessageStream(stream, ({ chunk }) => chunk);

    const result = await convertAsyncIterableToArray(mappedStream);

    expect(result).toEqual([START_CHUNK, ...TEXT_CHUNKS, FINISH_CHUNK]);
  });

  it('should filter out chunks by returning null', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      ...TEXT_CHUNKS,
      ...REASONING_CHUNKS,
      FINISH_CHUNK,
    ]);

    const mappedStream = mapUIMessageStream(stream, ({ chunk, part }) => {
      return part.type === 'reasoning' ? null : chunk;
    });

    const result = await convertAsyncIterableToArray(mappedStream);

    // Should not include reasoning chunks or the step that only contained reasoning
    expect(result).toEqual([START_CHUNK, ...TEXT_CHUNKS, FINISH_CHUNK]);
  });

  it('should transform chunks', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      ...TEXT_CHUNKS,
      FINISH_CHUNK,
    ]);

    const mappedStream = mapUIMessageStream(stream, ({ chunk }) => {
      if (chunk.type === 'text-delta') {
        return { ...chunk, delta: chunk.delta.toUpperCase() };
      }
      return chunk;
    });

    const result = await convertAsyncIterableToArray(mappedStream);

    const textDeltas = result.filter((c) => c.type === 'text-delta');
    expect(textDeltas).toEqual([
      { type: 'text-delta', id: '1', delta: 'HELLO' },
      { type: 'text-delta', id: '1', delta: ' WORLD' },
    ]);
  });

  it('should handle single-chunk parts (file)', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      ...FILE_CHUNKS,
      FINISH_CHUNK,
    ]);

    const mappedStream = mapUIMessageStream(stream, ({ chunk, part }) => {
      if (part.type === 'file') {
        expect(chunk.type).toBe('file');
      }
      return chunk;
    });

    const result = await convertAsyncIterableToArray(mappedStream);

    const fileChunks = result.filter((c) => c.type === 'file');
    expect(fileChunks.length).toBe(1);
  });

  it('should always pass through meta chunks', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      MESSAGE_METADATA_CHUNK,
      ERROR_CHUNK,
      ABORT_CHUNK,
      FINISH_CHUNK,
    ]);

    // Even when returning null for everything, meta chunks pass through
    const mappedStream = mapUIMessageStream(stream, () => null);

    const result = await convertAsyncIterableToArray(mappedStream);

    expect(result).toEqual([
      START_CHUNK,
      MESSAGE_METADATA_CHUNK,
      ERROR_CHUNK,
      ABORT_CHUNK,
      FINISH_CHUNK,
    ]);
  });

  it('should not emit start-step if all content is filtered out', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      ...REASONING_CHUNKS, // Will be filtered
      FINISH_CHUNK,
    ]);

    const mappedStream = mapUIMessageStream(stream, ({ chunk, part }) => {
      return part.type === 'reasoning' ? null : chunk;
    });

    const result = await convertAsyncIterableToArray(mappedStream);

    // Should not include start-step or finish-step since all content was filtered
    expect(result).toEqual([START_CHUNK, FINISH_CHUNK]);
  });

  it('should provide complete tool part', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      ...TOOL_CHUNKS,
      FINISH_CHUNK,
    ]);

    let capturedPart: unknown;
    const mappedStream = mapUIMessageStream(stream, ({ chunk, part }) => {
      if (part.type === 'tool-weather') {
        capturedPart = part;
      }
      return chunk;
    });

    await convertAsyncIterableToArray(mappedStream);

    // Part should have all tool properties populated
    expect(capturedPart).toMatchObject({
      type: 'tool-weather',
      toolCallId: '3',
      state: 'output-available',
      input: { location: 'NYC' },
      output: { temperature: 65 },
    });
  });

  it('should provide index and chunks array in context', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      ...TEXT_CHUNKS,
      FINISH_CHUNK,
    ]);

    const indices: number[] = [];
    const chunkCounts: number[] = [];
    const mappedStream = mapUIMessageStream(
      stream,
      ({ chunk }, { index, chunks }) => {
        indices.push(index);
        chunkCounts.push(chunks.length);
        return chunk;
      },
    );

    await convertAsyncIterableToArray(mappedStream);

    // The map function is only called for content chunks (text-start, 2x text-delta, text-end)
    // Meta chunks (start, finish) and step chunks (start-step, finish-step) pass through automatically
    // But the index and chunks array include ALL chunks
    // Stream: start(0), start-step(1), text-start(2), text-delta(3), text-delta(4), text-end(5), finish-step(6), finish(7)
    // Map called for indices: 2, 3, 4, 5
    expect(indices).toEqual([2, 3, 4, 5]);
    // Chunks array includes all chunks up to and including current
    expect(chunkCounts).toEqual([3, 4, 5, 6]);
  });

  it('should allow accessing previous chunks', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      ...TEXT_CHUNKS,
      FINISH_CHUNK,
    ]);

    let lastChunksSnapshot: UIMessageChunk[] = [];
    const mappedStream = mapUIMessageStream(stream, ({ chunk }, { chunks }) => {
      lastChunksSnapshot = [...chunks];
      return chunk;
    });

    await convertAsyncIterableToArray(mappedStream);

    // Last time map was called was for text-end (index 5)
    // At that point, chunks array has: start, start-step, text-start, text-delta, text-delta, text-end
    expect(lastChunksSnapshot.length).toBe(6);
    expect(lastChunksSnapshot[0]?.type).toBe('start');
    expect(lastChunksSnapshot[5]?.type).toBe('text-end');
  });

  it('should provide partial part with accumulated text', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      ...TEXT_CHUNKS,
      FINISH_CHUNK,
    ]);

    const textContents: (string | undefined)[] = [];
    const mappedStream = mapUIMessageStream(stream, ({ chunk, part }) => {
      if (part.type === 'text') {
        // AI SDK's readUIMessageStream accumulates text in part.text
        textContents.push((part as { text?: string }).text);
      }
      return chunk;
    });

    await convertAsyncIterableToArray(mappedStream);

    // AI SDK behavior: text is accumulated, not delta-based
    // text-start: '' (empty), text-delta: 'Hello', text-delta: 'Hello World', text-end: 'Hello World'
    expect(textContents).toEqual(['', 'Hello', 'Hello World', 'Hello World']);
  });
});
