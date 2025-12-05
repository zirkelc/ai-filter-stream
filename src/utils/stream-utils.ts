import type { UIMessage, UIMessageChunk } from 'ai';
import type { InferUIMessagePart } from '../types.js';

/**
 * Checks if a chunk is a control/meta chunk that should always pass through.
 */
export function isMetaChunk(chunk: UIMessageChunk): boolean {
  return (
    chunk.type === 'start' ||
    chunk.type === 'finish' ||
    chunk.type === 'abort' ||
    chunk.type === 'message-metadata' ||
    chunk.type === 'error'
  );
}

export function isMessageDataChunk(chunk: UIMessageChunk): boolean {
  return chunk.type.startsWith('data-');
}

/**
 * Checks if a chunk marks the start of a message.
 */
export function isMessageStartChunk(chunk: UIMessageChunk): boolean {
  return chunk.type === 'start';
}

/**
 * Checks if a chunk marks the end of a message.
 */
export function isMessageEndChunk(chunk: UIMessageChunk): boolean {
  return chunk.type === 'finish' || chunk.type === 'abort';
}

/**
 * Checks if a chunk marks the start of a step.
 */
export function isStepStartChunk(chunk: UIMessageChunk): boolean {
  return chunk.type === 'start-step';
}

/**
 * Checks if a chunk marks the end of a step.
 */
export function isStepEndChunk(chunk: UIMessageChunk): boolean {
  return chunk.type === 'finish-step';
}

/**
 * Checks if a part is complete based on its state.
 * Single-chunk parts (file, source-url, etc.) are always complete.
 * Multi-chunk parts (text, reasoning, tool) are complete when their state is terminal.
 */
export function isPartComplete<UI_MESSAGE extends UIMessage>(
  part: InferUIMessagePart<UI_MESSAGE>,
): boolean {
  if (part.type === 'step-start') return false;
  if (!('state' in part)) return true; // Single-chunk parts (file, source-url, etc.)
  return (
    part.state === 'done' ||
    part.state === 'output-available' ||
    part.state === 'output-error'
  );
}
