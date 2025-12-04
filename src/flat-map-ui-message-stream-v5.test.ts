import {
  convertArrayToReadableStream,
  convertAsyncIterableToArray,
} from '@ai-sdk/provider-utils/test';
import type { UIMessageChunk } from 'ai';
import { describe, expect, it } from 'vitest';
import {
  flatMapUIMessageStream,
  partTypeIs,
} from './flat-map-ui-message-stream-v4.js';
import {
  // flatMapUIMessageStream,
  // partTypeIs,
} from './flat-map-ui-message-stream-v5.js';
import {
  FILE_CHUNKS,
  FINISH_CHUNK,
  type MyUIMessage,
  type MyUIMessageChunk,
  REASONING_CHUNKS,
  START_CHUNK,
  TEXT_CHUNKS,
  TOOL_CHUNKS,
} from './test-utils.js';

describe('flatMapUIMessageStream', () => {
  it('should pass through all parts with identity flatMap', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      ...TEXT_CHUNKS,
      FINISH_CHUNK,
    ]);

    const mappedStream = flatMapUIMessageStream(stream, ({ part }) => part);

    const result = await convertAsyncIterableToArray(mappedStream);

    expect(result).toEqual([START_CHUNK, ...TEXT_CHUNKS, FINISH_CHUNK]);
  });

  it('should filter out parts by returning null', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      ...TEXT_CHUNKS,
      ...REASONING_CHUNKS,
      FINISH_CHUNK,
    ]);

    const mappedStream = flatMapUIMessageStream(stream, ({ part }) => {
      return part.type === 'reasoning' ? null : part;
    });

    const result = await convertAsyncIterableToArray(mappedStream);

    // Should not include reasoning chunks
    expect(result).toEqual([START_CHUNK, ...TEXT_CHUNKS, FINISH_CHUNK]);
  });

  it('should transform parts', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      ...TEXT_CHUNKS,
      FINISH_CHUNK,
    ]);

    const mappedStream = flatMapUIMessageStream(stream, ({ part }) => {
      if (part.type === 'text') {
        const textPart = part;
        return { ...textPart, text: textPart.text.toUpperCase() };
      }
      return part;
    });

    const result = await convertAsyncIterableToArray(mappedStream);

    // Find the text delta - it should be uppercase
    const textDeltas = result.filter((c) => c.type === 'text-delta');
    expect(textDeltas.length).toBe(1);
    expect(textDeltas[0]!.delta).toBe('HELLO WORLD');
  });

  it('should handle single-chunk parts (file)', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      ...FILE_CHUNKS,
      FINISH_CHUNK,
    ]);

    const mappedStream = flatMapUIMessageStream(stream, ({ part }) => {
      expect(part.type).toBe('file');
      return part;
    });

    const result = await convertAsyncIterableToArray(mappedStream);

    const fileChunks = result.filter((c) => c.type === 'file');
    expect(fileChunks.length).toBe(1);
  });

  it('should reconstruct tool parts correctly', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      ...TOOL_CHUNKS,
      FINISH_CHUNK,
    ]);

    let capturedPart: unknown;
    const mappedStream = flatMapUIMessageStream(stream, ({ part }) => {
      if (part.type === 'tool-weather') {
        capturedPart = part;
      }
      return part;
    });

    await convertAsyncIterableToArray(mappedStream);

    expect(capturedPart).toMatchObject({
      type: 'tool-weather',
      toolCallId: '3',
      state: 'output-available',
      input: { location: 'NYC' },
      output: { temperature: 65 },
    });
  });

  it('should always pass through meta chunks immediately', async () => {
    const ERROR_CHUNK: MyUIMessageChunk = {
      type: 'error',
      errorText: 'Test error',
    };

    const stream = convertArrayToReadableStream([
      START_CHUNK,
      ERROR_CHUNK,
      FINISH_CHUNK,
    ]);

    const mappedStream = flatMapUIMessageStream(stream, () => null);

    const result = await convertAsyncIterableToArray(mappedStream);

    expect(result).toEqual([START_CHUNK, ERROR_CHUNK, FINISH_CHUNK]);
  });

  it('should not emit start-step if all content is filtered out', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      ...REASONING_CHUNKS,
      FINISH_CHUNK,
    ]);

    const mappedStream = flatMapUIMessageStream(stream, () => null);

    const result = await convertAsyncIterableToArray(mappedStream);

    // Should not include start-step or finish-step
    expect(result).toEqual([START_CHUNK, FINISH_CHUNK]);
  });

  it('should provide index and parts array in context', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      ...TEXT_CHUNKS,
      ...REASONING_CHUNKS,
      FINISH_CHUNK,
    ]);

    const indices: number[] = [];
    const partCounts: number[] = [];
    const mappedStream = flatMapUIMessageStream(
      stream,
      ({ part }, { index, parts }) => {
        indices.push(index);
        partCounts.push(parts.length);
        return part;
      },
    );

    await convertAsyncIterableToArray(mappedStream);

    // Index should increment for each part
    expect(indices).toEqual([0, 1]);
    // Parts array should grow with each part
    expect(partCounts).toEqual([1, 2]);
  });

  it('should allow accessing previous parts', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      ...TEXT_CHUNKS,
      ...REASONING_CHUNKS,
      FINISH_CHUNK,
    ]);

    let lastPartsSnapshot: MyUIMessage['parts'] = [];
    const mappedStream = flatMapUIMessageStream<MyUIMessage>(
      stream,
      ({ part }, { parts }) => {
        lastPartsSnapshot = [...parts];
        return part;
      },
    );

    await convertAsyncIterableToArray(mappedStream);

    // Should have all parts at the end
    expect(lastPartsSnapshot.length).toBe(2);
    expect(lastPartsSnapshot[0]?.type).toBe('text');
    expect(lastPartsSnapshot[1]?.type).toBe('reasoning');
  });

  describe('predicate', () => {
    it('should buffer only matching parts and pass through others', async () => {
      const stream = convertArrayToReadableStream([
        START_CHUNK,
        ...TEXT_CHUNKS,
        ...REASONING_CHUNKS,
        FINISH_CHUNK,
      ]);

      const processedTypes: string[] = [];
      const mappedStream = flatMapUIMessageStream(
        stream,
        partTypeIs('text'),
        ({ part }) => {
          processedTypes.push(part.type);
          return { ...part, text: part.text.toUpperCase() };
        },
      );

      const result = await convertAsyncIterableToArray(mappedStream);

      // Only text should have been processed by flatMap
      expect(processedTypes).toEqual(['text']);

      // Text should be transformed
      const textDeltas = result.filter((c) => c.type === 'text-delta');
      expect(textDeltas.length).toBe(1);
      expect(textDeltas[0]!.delta).toBe('HELLO WORLD');

      // Reasoning should pass through unchanged
      const reasoningDeltas = result.filter(
        (c) => c.type === 'reasoning-delta',
      );
      expect(reasoningDeltas.length).toBe(2);
      expect(reasoningDeltas[0]!.delta).toBe('Think');
      expect(reasoningDeltas[1]!.delta).toBe('ing...');
    });

    it('should support array of types', async () => {
      const stream = convertArrayToReadableStream([
        START_CHUNK,
        ...TEXT_CHUNKS,
        ...REASONING_CHUNKS,
        ...FILE_CHUNKS,
        FINISH_CHUNK,
      ]);

      const processedTypes: string[] = [];
      const mappedStream = flatMapUIMessageStream(
        stream,
        partTypeIs(['text', 'reasoning']),
        ({ part }) => {
          processedTypes.push(part.type);
          return part;
        },
      );

      const result = await convertAsyncIterableToArray(mappedStream);

      // Both text and reasoning should be processed
      expect(processedTypes).toEqual(['text', 'reasoning']);

      // All parts should be present (file passed through)
      const fileChunks = result.filter((c) => c.type === 'file');
      expect(fileChunks.length).toBe(1);
    });

    it('should filter matching parts when returning null', async () => {
      const stream = convertArrayToReadableStream([
        START_CHUNK,
        ...TEXT_CHUNKS,
        ...REASONING_CHUNKS,
        FINISH_CHUNK,
      ]);

      const mappedStream = flatMapUIMessageStream(
        stream,
        partTypeIs('text'),
        ({ part }) => null, // Filter out text parts
      );

      const result = await convertAsyncIterableToArray(mappedStream);

      // Text should be filtered out
      const textChunks = result.filter((c) => c.type.startsWith('text'));
      expect(textChunks.length).toBe(0);

      // Reasoning should pass through (not matched by predicate)
      const reasoningChunks = result.filter((c) =>
        c.type.startsWith('reasoning'),
      );
      expect(reasoningChunks.length).toBe(4); // start, delta, delta, end
    });

    it('should pass through tool parts when predicate only matches text', async () => {
      const stream = convertArrayToReadableStream([
        START_CHUNK,
        ...TOOL_CHUNKS,
        ...TEXT_CHUNKS,
        FINISH_CHUNK,
      ]);

      const processedTypes: string[] = [];
      const mappedStream = flatMapUIMessageStream(
        stream,
        partTypeIs('text'),
        ({ part }) => {
          processedTypes.push(part.type);
          return part;
        },
      );

      const result = await convertAsyncIterableToArray(mappedStream);

      // Only text should be processed
      expect(processedTypes).toEqual(['text']);

      // Tool chunks should be present
      const toolChunks = result.filter((c) => c.type.startsWith('tool-'));
      expect(toolChunks.length).toBe(4);
    });

    it('should maintain step boundaries for passed-through parts', async () => {
      const stream = convertArrayToReadableStream([
        START_CHUNK,
        ...REASONING_CHUNKS,
        FINISH_CHUNK,
      ]);

      const mappedStream = flatMapUIMessageStream(
        stream,
        partTypeIs('text'), // Only match text (reasoning will pass through)
        ({ part }) => part,
      );

      const result = await convertAsyncIterableToArray(mappedStream);

      // Step boundaries should be present for reasoning
      expect(result).toContainEqual({ type: 'start-step' });
      expect(result).toContainEqual({ type: 'finish-step' });
    });

    it('should stream non-matching chunks immediately without buffering', async () => {
      // This test verifies that chunks for non-matching parts are emitted
      // as they arrive, not buffered until the part is complete
      const chunks: MyUIMessageChunk[] = [
        START_CHUNK,
        ...REASONING_CHUNKS,
        FINISH_CHUNK,
      ];

      const stream = convertArrayToReadableStream(chunks);
      const emittedChunks: UIMessageChunk[] = [];

      const mappedStream = flatMapUIMessageStream(
        stream,
        partTypeIs('text'), // Only buffer text, stream reasoning immediately
        ({ part }) => part,
      );

      // Collect chunks as they are emitted
      for await (const chunk of mappedStream) {
        emittedChunks.push(chunk);
      }

      // Verify all reasoning chunks are present and in order
      const reasoningChunks = emittedChunks.filter(
        (c) =>
          c.type === 'reasoning-start' ||
          c.type === 'reasoning-delta' ||
          c.type === 'reasoning-end',
      );
      expect(reasoningChunks.length).toBe(4);
      expect(reasoningChunks[0]?.type).toBe('reasoning-start');
      expect(reasoningChunks[1]?.type).toBe('reasoning-delta');
      expect(reasoningChunks[2]?.type).toBe('reasoning-delta');
      expect(reasoningChunks[3]?.type).toBe('reasoning-end');
    });

    it('should buffer matching parts and stream non-matching parts in interleaved stream', async () => {
      // Interleaved stream: reasoning (stream) -> text (buffer) -> reasoning (stream)
      const chunks: MyUIMessageChunk[] = [
        START_CHUNK,
        { type: 'start-step' },
        // First reasoning part (should stream immediately)
        { type: 'reasoning-start', id: '1' },
        { type: 'reasoning-delta', id: '1', delta: 'First thought' },
        { type: 'reasoning-end', id: '1' },
        // Text part (should be buffered and transformed)
        { type: 'text-start', id: '2' },
        { type: 'text-delta', id: '2', delta: 'hello' },
        { type: 'text-end', id: '2' },
        // Second reasoning part (should stream immediately)
        { type: 'reasoning-start', id: '3' },
        { type: 'reasoning-delta', id: '3', delta: 'Second thought' },
        { type: 'reasoning-end', id: '3' },
        { type: 'finish-step' },
        FINISH_CHUNK,
      ];

      const stream = convertArrayToReadableStream(chunks);

      const mappedStream = flatMapUIMessageStream(
        stream,
        partTypeIs('text'),
        ({ part }) => ({ ...part, text: part.text.toUpperCase() }),
      );

      const result = await convertAsyncIterableToArray(mappedStream);

      // Reasoning should be unchanged (streamed through)
      const reasoningDeltas = result.filter(
        (c) => c.type === 'reasoning-delta',
      );
      expect(reasoningDeltas.length).toBe(2);
      expect((reasoningDeltas[0] as { delta: string }).delta).toBe(
        'First thought',
      );
      expect((reasoningDeltas[1] as { delta: string }).delta).toBe(
        'Second thought',
      );

      // Text should be transformed (buffered)
      const textDeltas = result.filter((c) => c.type === 'text-delta');
      expect(textDeltas.length).toBe(1);
      expect((textDeltas[0] as { delta: string }).delta).toBe('HELLO');
    });
  });
});
