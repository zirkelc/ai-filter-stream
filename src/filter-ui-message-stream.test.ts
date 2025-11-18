import type { UIMessageChunk } from 'ai';
import { describe, expect, it } from 'vitest';
import { createAsyncIterableStream } from './create-async-iterable-stream.js';
import { filterUIMessageStream } from './filter-ui-message-stream.js';

describe('filterUIMessageStream', () => {
  it('should filter chunks using include', async () => {
    const chunks: UIMessageChunk[] = [
      { type: 'start' },
      { type: 'text-start', id: '1' },
      { type: 'text-delta', id: '1', delta: 'Hello' },
      { type: 'text-end', id: '1' },
      { type: 'reasoning-start', id: '2' },
      { type: 'reasoning-delta', id: '2', delta: 'Thinking...' },
      { type: 'reasoning-end', id: '2' },
      { type: 'finish' },
    ];

    const stream = createAsyncIterableStream(
      new ReadableStream({
        start(controller) {
          for (const chunk of chunks) {
            controller.enqueue(chunk);
          }
          controller.close();
        },
      }),
    );

    const filtered = filterUIMessageStream(stream, {
      includeParts: ['text'],
    });

    const result: UIMessageChunk[] = [];
    for await (const chunk of filtered) {
      result.push(chunk);
    }

    expect(result).toEqual([
      { type: 'start' },
      { type: 'text-start', id: '1' },
      { type: 'text-delta', id: '1', delta: 'Hello' },
      { type: 'text-end', id: '1' },
      { type: 'finish' },
    ]);
  });

  it('should filter chunks using exclude', async () => {
    const chunks: UIMessageChunk[] = [
      { type: 'start' },
      { type: 'text-start', id: '1' },
      { type: 'text-delta', id: '1', delta: 'Hello' },
      { type: 'text-end', id: '1' },
      { type: 'reasoning-start', id: '2' },
      { type: 'reasoning-delta', id: '2', delta: 'Thinking...' },
      { type: 'reasoning-end', id: '2' },
      { type: 'finish' },
    ];

    const stream = createAsyncIterableStream(
      new ReadableStream({
        start(controller) {
          for (const chunk of chunks) {
            controller.enqueue(chunk);
          }
          controller.close();
        },
      }),
    );

    const filtered = filterUIMessageStream(stream, {
      excludeParts: ['reasoning'],
    });

    const result: UIMessageChunk[] = [];
    for await (const chunk of filtered) {
      result.push(chunk);
    }

    expect(result).toEqual([
      { type: 'start' },
      { type: 'text-start', id: '1' },
      { type: 'text-delta', id: '1', delta: 'Hello' },
      { type: 'text-end', id: '1' },
      { type: 'finish' },
    ]);
  });

  it('should filter chunks using filter function', async () => {
    const chunks: UIMessageChunk[] = [
      { type: 'start' },
      { type: 'text-start', id: '1' },
      { type: 'text-delta', id: '1', delta: 'Hello' },
      { type: 'text-end', id: '1' },
      {
        type: 'tool-input-start',
        toolCallId: '1',
        toolName: 'weather-current',
      },
      {
        type: 'tool-input-available',
        toolCallId: '1',
        toolName: 'weather-current',
        input: {},
      },
      {
        type: 'tool-input-start',
        toolCallId: '2',
        toolName: 'calculator',
      },
      { type: 'finish' },
    ];

    const stream = createAsyncIterableStream(
      new ReadableStream({
        start(controller) {
          for (const chunk of chunks) {
            controller.enqueue(chunk);
          }
          controller.close();
        },
      }),
    );

    const filtered = filterUIMessageStream(stream, {
      filterParts: ({ partType }) => {
        // Include text and any tool that starts with 'tool-weather'
        return partType === 'text' || partType.startsWith('tool-weather');
      },
    });

    const result: UIMessageChunk[] = [];
    for await (const chunk of filtered) {
      result.push(chunk);
    }

    expect(result).toEqual([
      { type: 'start' },
      { type: 'text-start', id: '1' },
      { type: 'text-delta', id: '1', delta: 'Hello' },
      { type: 'text-end', id: '1' },
      {
        type: 'tool-input-start',
        toolCallId: '1',
        toolName: 'weather-current',
      },
      {
        type: 'tool-input-available',
        toolCallId: '1',
        toolName: 'weather-current',
        input: {},
      },
      { type: 'finish' },
    ]);
  });

  it('should buffer start-step and only include if subsequent content passes filter', async () => {
    const chunks: UIMessageChunk[] = [
      { type: 'start' },
      { type: 'start-step' },
      { type: 'text-start', id: '1' },
      { type: 'text-delta', id: '1', delta: 'Hello' },
      { type: 'text-end', id: '1' },
      { type: 'finish-step' },
      { type: 'finish' },
    ];

    const stream = createAsyncIterableStream(
      new ReadableStream({
        start(controller) {
          for (const chunk of chunks) {
            controller.enqueue(chunk);
          }
          controller.close();
        },
      }),
    );

    const filtered = filterUIMessageStream(stream, {
      includeParts: ['text', 'step-start'],
    });

    const result: UIMessageChunk[] = [];
    for await (const chunk of filtered) {
      result.push(chunk);
    }

    expect(result).toEqual([
      { type: 'start' },
      { type: 'start-step' },
      { type: 'text-start', id: '1' },
      { type: 'text-delta', id: '1', delta: 'Hello' },
      { type: 'text-end', id: '1' },
      { type: 'finish-step' },
      { type: 'finish' },
    ]);
  });

  it('should not include start-step if subsequent content is filtered out', async () => {
    const chunks: UIMessageChunk[] = [
      { type: 'start' },
      { type: 'start-step' },
      { type: 'reasoning-start', id: '1' },
      { type: 'reasoning-delta', id: '1', delta: 'Thinking...' },
      { type: 'reasoning-end', id: '1' },
      { type: 'finish-step' },
      { type: 'finish' },
    ];

    const stream = createAsyncIterableStream(
      new ReadableStream({
        start(controller) {
          for (const chunk of chunks) {
            controller.enqueue(chunk);
          }
          controller.close();
        },
      }),
    );

    const filtered = filterUIMessageStream(stream, {
      includeParts: ['text'],
    });

    const result: UIMessageChunk[] = [];
    for await (const chunk of filtered) {
      result.push(chunk);
    }

    expect(result).toEqual([{ type: 'start' }, { type: 'finish' }]);
  });

  it('should handle dynamic tools', async () => {
    const chunks: UIMessageChunk[] = [
      { type: 'start' },
      {
        type: 'tool-input-start',
        toolCallId: '1',
        toolName: 'dynamic-weather',
        dynamic: true,
      },
      {
        type: 'tool-input-available',
        toolCallId: '1',
        toolName: 'dynamic-weather',
        input: { location: 'NYC' },
        dynamic: true,
      },
      {
        type: 'tool-output-available',
        toolCallId: '1',
        output: { temperature: 65 },
        dynamic: true,
      },
      { type: 'finish' },
    ];

    const stream = createAsyncIterableStream(
      new ReadableStream({
        start(controller) {
          for (const chunk of chunks) {
            controller.enqueue(chunk);
          }
          controller.close();
        },
      }),
    );

    const filtered = filterUIMessageStream(stream, {
      includeParts: ['dynamic-tool'],
    });

    const result: UIMessageChunk[] = [];
    for await (const chunk of filtered) {
      result.push(chunk);
    }

    expect(result).toEqual([
      { type: 'start' },
      {
        type: 'tool-input-start',
        toolCallId: '1',
        toolName: 'dynamic-weather',
        dynamic: true,
      },
      {
        type: 'tool-input-available',
        toolCallId: '1',
        toolName: 'dynamic-weather',
        input: { location: 'NYC' },
        dynamic: true,
      },
      {
        type: 'tool-output-available',
        toolCallId: '1',
        output: { temperature: 65 },
        dynamic: true,
      },
      { type: 'finish' },
    ]);
  });
});
