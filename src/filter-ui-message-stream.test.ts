import {
  convertArrayToReadableStream,
  convertAsyncIterableToArray,
} from '@ai-sdk/provider-utils/test';
import {
  dynamicTool,
  type InferUIMessageChunk,
  type InferUITools,
  readUIMessageStream,
  tool,
  type UIMessage,
} from 'ai';
import { describe, expect, it } from 'vitest';
import { z } from 'zod';
import { filterUIMessageStream } from './filter-ui-message-stream.js';
import type { InferUIMessagePart } from './types.js';

type MyMetadata = { id: string };

type MyDataPart = { weather: { location: string; temperature: number } };

type MyTools = InferUITools<typeof tools>;

type MyUIMessage = UIMessage<MyMetadata, MyDataPart, MyTools>;

type MyUIMessageChunk = InferUIMessageChunk<MyUIMessage>;

type MyUIMessagePart = InferUIMessagePart<MyUIMessage>;

const tools = {
  weather: tool({
    description: 'Get the weather in a location',
    inputSchema: z.object({
      location: z.string().describe('The location to get the weather for'),
    }),
    execute: ({ location }) => ({
      location,
      temperature: 72 + Math.floor(Math.random() * 21) - 10,
    }),
  }),
  calculator: dynamicTool({
    description: 'Calculate a mathematical expression',
    execute: async ({ expression }) => ({
      result: eval(expression),
    }),
  }),
};

const START_CHUNK: MyUIMessageChunk = { type: 'start' };
const FINISH_CHUNK: MyUIMessageChunk = { type: 'finish' };
const ABORT_CHUNK: MyUIMessageChunk = { type: 'abort' };
const MESSAGE_METADATA_CHUNK: MyUIMessageChunk = {
  type: 'message-metadata',
  messageMetadata: { id: '1' },
};
const ERROR_CHUNK: MyUIMessageChunk = {
  type: 'error',
  errorText: 'Test error',
};

const TEXT_CHUNKS: MyUIMessageChunk[] = [
  { type: 'start-step' },
  { type: 'text-start', id: '1' },
  { type: 'text-delta', id: '1', delta: 'Hello' },
  { type: 'text-end', id: '1' },
  { type: 'finish-step' },
];

const REASONING_CHUNKS: MyUIMessageChunk[] = [
  { type: 'start-step' },
  { type: 'reasoning-start', id: '2' },
  { type: 'reasoning-delta', id: '2', delta: 'Thinking...' },
  { type: 'reasoning-end', id: '2' },
  { type: 'finish-step' },
];

const TOOL_CHUNKS: MyUIMessageChunk[] = [
  { type: 'start-step' },
  {
    type: 'tool-input-start',
    toolCallId: '3',
    toolName: 'weather',
  },
  {
    type: 'tool-input-delta',
    toolCallId: '3',
    inputTextDelta: '{"location"',
  },
  {
    type: 'tool-input-available',
    toolCallId: '3',
    toolName: 'weather',
    input: { location: 'NYC' },
  },
  {
    type: 'tool-output-available',
    toolCallId: '3',
    output: { temperature: 65 },
  },
  { type: 'finish-step' },
];

const DYNAMIC_TOOL_CHUNKS: MyUIMessageChunk[] = [
  { type: 'start-step' },
  {
    type: 'tool-input-start',
    toolCallId: '4',
    toolName: 'calculator',
    dynamic: true,
  },
  {
    type: 'tool-input-available',
    toolCallId: '4',
    toolName: 'calculator',
    input: { expression: '2+2' },
    dynamic: true,
  },
  {
    type: 'tool-output-available',
    toolCallId: '4',
    output: { result: 4 },
    dynamic: true,
  },
  { type: 'finish-step' },
];

const TOOL_ERROR_CHUNKS: MyUIMessageChunk[] = [
  { type: 'start-step' },
  {
    type: 'tool-input-error',
    toolCallId: '5',
    toolName: 'failed',
    input: {},
    errorText: 'Invalid input',
  },
  {
    type: 'tool-output-error',
    toolCallId: '5',
    errorText: 'Execution failed',
  },
  { type: 'finish-step' },
];

const SOURCE_CHUNKS: MyUIMessageChunk[] = [
  { type: 'start-step' },
  {
    type: 'source-url',
    sourceId: 'source-1',
    url: 'https://example.com',
    title: 'Example Source',
  },
  {
    type: 'source-document',
    sourceId: 'source-2',
    mediaType: 'application/pdf',
    title: 'Document Title',
  },
  { type: 'finish-step' },
];

const FILE_CHUNKS: MyUIMessageChunk[] = [
  { type: 'start-step' },
  {
    type: 'file',
    url: 'https://example.com/file.pdf',
    mediaType: 'application/pdf',
  },
  { type: 'finish-step' },
];

const DATA_CHUNKS: MyUIMessageChunk[] = [
  { type: 'start-step' },
  {
    type: 'data-weather',
    data: { location: 'NYC', temperature: 72 },
  },
  { type: 'finish-step' },
];

describe('filterUIMessageStream', () => {
  it('should filter chunks using include', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      ...REASONING_CHUNKS,
      ...TEXT_CHUNKS,
      FINISH_CHUNK,
    ]);

    const filteredStream = filterUIMessageStream(stream, {
      includeParts: ['text'],
    });

    const result = await convertAsyncIterableToArray(filteredStream);

    expect(result).toMatchInlineSnapshot(`
      [
        {
          "type": "start",
        },
        {
          "type": "start-step",
        },
        {
          "id": "1",
          "type": "text-start",
        },
        {
          "delta": "Hello",
          "id": "1",
          "type": "text-delta",
        },
        {
          "id": "1",
          "type": "text-end",
        },
        {
          "type": "finish-step",
        },
        {
          "type": "finish",
        },
      ]
    `);
  });

  it('should filter chunks using exclude', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      ...REASONING_CHUNKS,
      ...TEXT_CHUNKS,
      FINISH_CHUNK,
    ]);

    const filteredStream = filterUIMessageStream(stream, {
      excludeParts: ['reasoning'],
    });

    const result = await convertAsyncIterableToArray(filteredStream);

    expect(result).toMatchInlineSnapshot(`
      [
        {
          "type": "start",
        },
        {
          "type": "start-step",
        },
        {
          "id": "1",
          "type": "text-start",
        },
        {
          "delta": "Hello",
          "id": "1",
          "type": "text-delta",
        },
        {
          "id": "1",
          "type": "text-end",
        },
        {
          "type": "finish-step",
        },
        {
          "type": "finish",
        },
      ]
    `);
  });

  it('should filter chunks using filter function', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      ...TOOL_CHUNKS,
      ...TEXT_CHUNKS,
      FINISH_CHUNK,
    ]);

    const filteredStream = filterUIMessageStream(stream, {
      filterParts: ({ partType }) => {
        // Include text and any tool that starts with 'tool-weather'
        return partType.startsWith('tool-weather');
      },
    });

    const result = await convertAsyncIterableToArray(filteredStream);

    expect(result).toMatchInlineSnapshot(`
      [
        {
          "type": "start",
        },
        {
          "type": "start-step",
        },
        {
          "toolCallId": "3",
          "toolName": "weather",
          "type": "tool-input-start",
        },
        {
          "inputTextDelta": "{"location"",
          "toolCallId": "3",
          "type": "tool-input-delta",
        },
        {
          "input": {
            "location": "NYC",
          },
          "toolCallId": "3",
          "toolName": "weather",
          "type": "tool-input-available",
        },
        {
          "output": {
            "temperature": 65,
          },
          "toolCallId": "3",
          "type": "tool-output-available",
        },
        {
          "type": "finish-step",
        },
        {
          "type": "finish",
        },
      ]
    `);
  });

  it('should not include start-step if subsequent content is filtered out', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      ...TEXT_CHUNKS,
      FINISH_CHUNK,
    ]);

    const filteredStream = filterUIMessageStream(stream, {
      includeParts: ['reasoning'],
    });

    const result = await convertAsyncIterableToArray(filteredStream);

    expect(result).toEqual([{ type: 'start' }, { type: 'finish' }]);
  });

  it('should always pass through controls chunks', async () => {
    const stream = convertArrayToReadableStream([
      START_CHUNK,
      ABORT_CHUNK,
      MESSAGE_METADATA_CHUNK,
      ERROR_CHUNK,
      FINISH_CHUNK,
    ]);

    const filteredStream = filterUIMessageStream(stream, {
      includeParts: [],
    });

    const result = await convertAsyncIterableToArray(filteredStream);

    expect(result).toEqual([
      { type: 'start' },
      { type: 'abort' },
      { type: 'message-metadata', messageMetadata: { id: '1' } },
      { type: 'error', errorText: 'Test error' },
      { type: 'finish' },
    ]);
  });

  describe('should handle each part type', () => {
    describe.each<MyUIMessagePart['type']>([
      'text',
      'reasoning',
      'tool-weather',
      'dynamic-tool',
      'source-url',
      'source-document',
      'data-weather',
      'file',
    ])(' %s', async (partType) => {
      it('includeParts', async () => {
        const stream = convertArrayToReadableStream([
          START_CHUNK,
          ...TEXT_CHUNKS,
          ...REASONING_CHUNKS,
          ...TOOL_CHUNKS,
          ...DYNAMIC_TOOL_CHUNKS,
          ...SOURCE_CHUNKS,
          ...FILE_CHUNKS,
          ...DATA_CHUNKS,
          FINISH_CHUNK,
        ]);

        const filteredStream = filterUIMessageStream(stream, {
          includeParts: [partType],
        });

        const result = await convertAsyncIterableToArray(
          readUIMessageStream({ stream: filteredStream }),
        );

        const parts = result.flatMap((message) => message.parts);
        const partsByType = parts.filter((part) => part.type === partType);
        expect(partsByType.length).toBeGreaterThan(0);
      });

      it('excludeParts', async () => {
        const stream = convertArrayToReadableStream([
          START_CHUNK,
          ...TEXT_CHUNKS,
          ...REASONING_CHUNKS,
          ...TOOL_CHUNKS,
          ...DYNAMIC_TOOL_CHUNKS,
          ...SOURCE_CHUNKS,
          ...FILE_CHUNKS,
          ...DATA_CHUNKS,
          FINISH_CHUNK,
        ]);

        const filteredStream = filterUIMessageStream(stream, {
          excludeParts: [partType],
        });

        const result = await convertAsyncIterableToArray(
          readUIMessageStream({ stream: filteredStream }),
        );

        const parts = result.flatMap((message) => message.parts);
        const partsByType = parts.filter((part) => part.type === partType);
        expect(partsByType.length).toBe(0);
      });

      it('filterParts', async () => {
        const stream = convertArrayToReadableStream([
          START_CHUNK,
          ...TEXT_CHUNKS,
          ...REASONING_CHUNKS,
          ...TOOL_CHUNKS,
          ...DYNAMIC_TOOL_CHUNKS,
          ...SOURCE_CHUNKS,
          ...FILE_CHUNKS,
          ...DATA_CHUNKS,
          FINISH_CHUNK,
        ]);

        const filteredStream = filterUIMessageStream(stream, {
          filterParts: (opts) => opts.partType === partType,
        });

        const result = await convertAsyncIterableToArray(
          readUIMessageStream({ stream: filteredStream }),
        );

        const parts = result.flatMap((message) => message.parts);
        const partsByType = parts.filter((part) => part.type === partType);
        expect(partsByType.length).toBeGreaterThan(0);
      });
    });
  });
});
