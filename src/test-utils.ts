import {
  type InferUIMessageChunk,
  type InferUITools,
  tool,
  type UIMessage,
} from 'ai';
import { z } from 'zod';
import type { InferUIMessagePart } from './types.js';

export type MyMetadata = { id: string };

export type MyDataPart = { weather: { location: string; temperature: number } };

export type MyTools = InferUITools<typeof tools>;

export type MyUIMessage = UIMessage<MyMetadata, MyDataPart, MyTools>;

export type MyUIMessageChunk = InferUIMessageChunk<MyUIMessage>;

export type MyUIMessagePart = InferUIMessagePart<MyUIMessage>;

const weatherTool = tool({
  description: 'Get the weather in a location',
  inputSchema: z.object({
    location: z.string().describe('The location to get the weather for'),
  }),
  execute: ({ location }) => ({
    location,
    temperature: 72 + Math.floor(Math.random() * 21) - 10,
  }),
});

// const calculatorTool = dynamicTool({
//   description: 'Calculate a mathematical expression',
//   inputSchema: z.object({}),
//   execute: (input) => {
//     const { expression } = input as { expression: string };
//     return {
//       result: `Result: ${expression}`,
//     };
//   },
// });

const tools = {
  weather: weatherTool,
  // calculator: calculatorTool,
};

export const START_CHUNK: MyUIMessageChunk = { type: 'start' };
export const FINISH_CHUNK: MyUIMessageChunk = { type: 'finish' };
export const ABORT_CHUNK: MyUIMessageChunk = { type: 'abort' };
export const MESSAGE_METADATA_CHUNK: MyUIMessageChunk = {
  type: 'message-metadata',
  messageMetadata: { id: '1' },
};
export const ERROR_CHUNK: MyUIMessageChunk = {
  type: 'error',
  errorText: 'Test error',
};

export const TEXT_CHUNKS: MyUIMessageChunk[] = [
  { type: 'start-step' },
  { type: 'text-start', id: '1' },
  { type: 'text-delta', id: '1', delta: 'Hello' },
  { type: 'text-delta', id: '1', delta: ' World' },
  { type: 'text-end', id: '1' },
  { type: 'finish-step' },
];

export const REASONING_CHUNKS: MyUIMessageChunk[] = [
  { type: 'start-step' },
  { type: 'reasoning-start', id: '2' },
  { type: 'reasoning-delta', id: '2', delta: 'Think' },
  { type: 'reasoning-delta', id: '2', delta: 'ing...' },
  { type: 'reasoning-end', id: '2' },
  { type: 'finish-step' },
];

export const TOOL_CHUNKS: MyUIMessageChunk[] = [
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

export const DYNAMIC_TOOL_CHUNKS: MyUIMessageChunk[] = [
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

export const TOOL_ERROR_CHUNKS: MyUIMessageChunk[] = [
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

export const SOURCE_CHUNKS: MyUIMessageChunk[] = [
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

export const DATA_CHUNKS: MyUIMessageChunk[] = [
  { type: 'start-step' },
  {
    type: 'data-weather',
    data: { location: 'NYC', temperature: 65 },
  },
  { type: 'finish-step' },
];

export const FILE_CHUNKS: MyUIMessageChunk[] = [
  { type: 'start-step' },
  {
    type: 'file',
    url: 'https://example.com/file.pdf',
    mediaType: 'application/pdf',
  },
  { type: 'finish-step' },
];
