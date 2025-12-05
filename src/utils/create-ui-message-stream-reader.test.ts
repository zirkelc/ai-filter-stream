import { convertArrayToReadableStream } from '@ai-sdk/provider-utils/test';
import { describe, expect, it } from 'vitest';
import { createUIMessageStreamReader } from './create-ui-message-stream-reader.js';
import {
  ABORT_CHUNK,
  ERROR_CHUNK,
  FINISH_CHUNK,
  MESSAGE_METADATA_CHUNK,
  type MyUIMessage,
  type MyUIMessageChunk,
  REASONING_CHUNKS,
  START_CHUNK,
  TEXT_CHUNKS,
  TOOL_CHUNKS,
} from './test-utils.js';

describe('createUIMessageStreamReader', () => {
  it('should yield all chunks from the input stream', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      ...TEXT_CHUNKS,
      FINISH_CHUNK,
    ]);

    const chunks: MyUIMessageChunk[] = [];
    for await (const { chunk } of createUIMessageStreamReader<MyUIMessage>(
      stream,
    )) {
      chunks.push(chunk);
    }

    expect(chunks).toEqual([START_CHUNK, ...TEXT_CHUNKS, FINISH_CHUNK]);
  });

  it('should yield assembled message for content chunks', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      ...TEXT_CHUNKS,
      FINISH_CHUNK,
    ]);

    const messages: MyUIMessage[] = [];
    for await (const { message } of createUIMessageStreamReader<MyUIMessage>(
      stream,
    )) {
      if (message) {
        messages.push(message);
      }
    }

    // TEXT_CHUNKS has: start-step, text-start, text-delta, text-delta, text-end, finish-step
    // Content chunks are: text-start, text-delta, text-delta, text-end (4 messages)
    expect(messages.length).toBe(4);

    // Each message should have parts
    for (const message of messages) {
      expect(message.parts.length).toBeGreaterThan(0);
    }

    // Last message should have accumulated text
    const lastMessage = messages[messages.length - 1];
    const textPart = lastMessage?.parts.find((p) => p.type === 'text');
    expect(textPart).toBeDefined();
    expect(textPart?.text).toBe('Hello World');
  });

  it('should yield undefined message for meta chunks (start, finish, error, abort, message-metadata)', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      MESSAGE_METADATA_CHUNK,
      ERROR_CHUNK,
      ABORT_CHUNK,
      FINISH_CHUNK,
    ]);

    const results: Array<{
      chunk: MyUIMessageChunk;
      message: MyUIMessage | undefined;
    }> = [];
    for await (const {
      chunk,
      message,
    } of createUIMessageStreamReader<MyUIMessage>(stream)) {
      results.push({ chunk, message });
    }

    // All meta chunks should have undefined message
    expect(results.every((r) => r.message === undefined)).toBe(true);
  });

  it('should yield undefined message for step-start chunks', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      { type: 'start-step' as const },
      { type: 'text-start' as const, id: '1' },
      { type: 'text-end' as const, id: '1' },
      { type: 'finish-step' as const },
      FINISH_CHUNK,
    ]);

    const stepStartResults: Array<MyUIMessage | undefined> = [];
    for await (const {
      chunk,
      message,
    } of createUIMessageStreamReader<MyUIMessage>(stream)) {
      if (chunk.type === 'start-step') {
        stepStartResults.push(message);
      }
    }

    // step-start should have undefined message
    expect(stepStartResults.length).toBe(1);
    expect(stepStartResults[0]).toBeUndefined();
  });

  it('should yield undefined message for finish-step chunks', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      { type: 'start-step' as const },
      { type: 'text-start' as const, id: '1' },
      { type: 'text-end' as const, id: '1' },
      { type: 'finish-step' as const },
      FINISH_CHUNK,
    ]);

    const finishStepResults: Array<MyUIMessage | undefined> = [];
    for await (const {
      chunk,
      message,
    } of createUIMessageStreamReader<MyUIMessage>(stream)) {
      if (chunk.type === 'finish-step') {
        finishStepResults.push(message);
      }
    }

    // finish-step should have undefined message
    expect(finishStepResults.length).toBe(1);
    expect(finishStepResults[0]).toBeUndefined();
  });

  it('should accumulate text in parts across text-delta chunks', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      ...TEXT_CHUNKS,
      FINISH_CHUNK,
    ]);

    const textContents: string[] = [];
    for await (const { message } of createUIMessageStreamReader<MyUIMessage>(
      stream,
    )) {
      if (message) {
        const textPart = message.parts.find((p) => p.type === 'text');
        if (textPart) {
          textContents.push(textPart?.text);
        }
      }
    }

    // AI SDK accumulates text: '', 'Hello', 'Hello World', 'Hello World'
    expect(textContents).toEqual(['', 'Hello', 'Hello World', 'Hello World']);
  });

  it('should provide correct part type for tool chunks', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      ...TOOL_CHUNKS,
      FINISH_CHUNK,
    ]);

    const partTypes: string[] = [];
    for await (const { message } of createUIMessageStreamReader<MyUIMessage>(
      stream,
    )) {
      if (message) {
        const lastPart = message.parts[message.parts.length - 1];
        if (lastPart) {
          partTypes.push(lastPart.type);
        }
      }
    }

    // AI SDK uses 'tool-weather' as the part type (tool-{toolName})
    expect(partTypes.every((t) => t === 'tool-weather')).toBe(true);
  });

  it('should handle reasoning chunks', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      ...REASONING_CHUNKS,
      FINISH_CHUNK,
    ]);

    const reasoningTexts: string[] = [];
    for await (const { message } of createUIMessageStreamReader<MyUIMessage>(
      stream,
    )) {
      if (message) {
        const reasoningPart = message.parts.find((p) => p.type === 'reasoning');
        if (reasoningPart) {
          reasoningTexts.push((reasoningPart as { text: string }).text);
        }
      }
    }

    // Reasoning text should accumulate
    expect(reasoningTexts).toEqual(['', 'Think', 'Thinking...', 'Thinking...']);
  });

  it('should release reader lock after iteration completes', async () => {
    const stream = convertArrayToReadableStream([START_CHUNK, FINISH_CHUNK]);

    // Consume all chunks
    for await (const _ of createUIMessageStreamReader<MyUIMessage>(stream)) {
      // Just iterate through
    }

    // After iteration, we should be able to get a new reader from the stream
    // (though it will be exhausted)
    const newReader = stream.getReader();
    const { done } = await newReader.read();
    expect(done).toBe(true);
    newReader.releaseLock();
  });

  it('should handle a complete stream lifecycle', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      ...TEXT_CHUNKS,
      ...REASONING_CHUNKS,
      FINISH_CHUNK,
    ]);

    const allChunks: MyUIMessageChunk[] = [];
    const allMessages: MyUIMessage[] = [];

    for await (const {
      chunk,
      message,
    } of createUIMessageStreamReader<MyUIMessage>(stream)) {
      allChunks.push(chunk);
      if (message) {
        allMessages.push(message);
      }
    }

    // Should have read all chunks
    expect(allChunks).toEqual([
      START_CHUNK,
      ...TEXT_CHUNKS,
      ...REASONING_CHUNKS,
      FINISH_CHUNK,
    ]);

    // Should have received messages for content chunks only
    // TEXT_CHUNKS: start-step, text-start, text-delta, text-delta, text-end, finish-step (4 content)
    // REASONING_CHUNKS: start-step, reasoning-start, reasoning-delta, reasoning-delta, reasoning-end, finish-step (4 content)
    expect(allMessages.length).toBe(8);

    // Final message should have both text and reasoning parts
    const finalMessage = allMessages[allMessages.length - 1];
    expect(finalMessage?.parts.some((p) => p.type === 'text')).toBe(true);
    expect(finalMessage?.parts.some((p) => p.type === 'reasoning')).toBe(true);
  });

  it('should yield chunk and message together', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      ...TEXT_CHUNKS,
      FINISH_CHUNK,
    ]);

    const results: Array<{ chunkType: string; hasMessage: boolean }> = [];
    for await (const {
      chunk,
      message,
    } of createUIMessageStreamReader<MyUIMessage>(stream)) {
      results.push({
        chunkType: chunk.type,
        hasMessage: message !== undefined,
      });
    }

    // Meta chunks and step chunks should not have messages
    // Content chunks should have messages
    expect(results).toEqual([
      { chunkType: 'start', hasMessage: false },
      { chunkType: 'start-step', hasMessage: false },
      { chunkType: 'text-start', hasMessage: true },
      { chunkType: 'text-delta', hasMessage: true },
      { chunkType: 'text-delta', hasMessage: true },
      { chunkType: 'text-end', hasMessage: true },
      { chunkType: 'finish-step', hasMessage: false },
      { chunkType: 'finish', hasMessage: false },
    ]);
  });
});
