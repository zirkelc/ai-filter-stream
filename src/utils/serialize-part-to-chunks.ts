import type {
  FileUIPart,
  InferUIMessageChunk,
  ReasoningUIPart,
  SourceDocumentUIPart,
  SourceUrlUIPart,
  TextUIPart,
  UIMessage,
} from 'ai';
import {
  getToolOrDynamicToolName,
  isDataUIPart,
  isToolOrDynamicToolUIPart,
} from 'ai';
import type { InferUIMessagePart } from '../types.js';

/**
 * Extracts the part ID from a list of chunks belonging to the same part.
 */
function getPartId<UI_MESSAGE extends UIMessage>(
  chunks: InferUIMessageChunk<UI_MESSAGE>[],
): string {
  for (const chunk of chunks) {
    if ('id' in chunk && chunk.id) return chunk.id;
    if ('toolCallId' in chunk && chunk.toolCallId) return chunk.toolCallId;
  }
  return 'unknown';
}

/**
 * Serializes a UIMessagePart back to chunks, including step boundaries.
 *
 * This function converts a complete part (e.g., TextUIPart, ToolUIPart)
 * back into the chunk format used by UIMessageStream, wrapped with
 * start-step and finish-step boundaries.
 *
 * @param part - The part to serialize
 * @param originalChunks - Original chunks for extracting IDs (for text/reasoning parts)
 * @returns Array of chunks representing the part, including step boundaries
 */
export function serializePartToChunks<UI_MESSAGE extends UIMessage>(
  part: InferUIMessagePart<UI_MESSAGE>,
  originalChunks: InferUIMessageChunk<UI_MESSAGE>[],
): InferUIMessageChunk<UI_MESSAGE>[] {
  const contentChunks = serializePartContentChunks(part, originalChunks);

  return [
    { type: 'start-step' } as InferUIMessageChunk<UI_MESSAGE>,
    ...contentChunks,
    { type: 'finish-step' } as InferUIMessageChunk<UI_MESSAGE>,
  ];
}

/**
 * Serializes a UIMessagePart to content chunks (without step boundaries).
 */
function serializePartContentChunks<UI_MESSAGE extends UIMessage>(
  part: InferUIMessagePart<UI_MESSAGE>,
  originalChunks: InferUIMessageChunk<UI_MESSAGE>[],
): InferUIMessageChunk<UI_MESSAGE>[] {
  if (part.type === 'file') {
    const filePart = part as FileUIPart;
    return [
      {
        type: 'file',
        mediaType: filePart.mediaType,
        url: filePart.url,
        providerMetadata: filePart.providerMetadata,
      } as InferUIMessageChunk<UI_MESSAGE>,
    ];
  }

  if (part.type === 'source-url') {
    const sourceUrlPart = part as SourceUrlUIPart;
    return [
      {
        type: 'source-url',
        sourceId: sourceUrlPart.sourceId,
        url: sourceUrlPart.url,
        title: sourceUrlPart.title,
        providerMetadata: sourceUrlPart.providerMetadata,
      } as InferUIMessageChunk<UI_MESSAGE>,
    ];
  }

  if (part.type === 'source-document') {
    const sourceDocumentPart = part as SourceDocumentUIPart;
    return [
      {
        type: 'source-document',
        sourceId: sourceDocumentPart.sourceId,
        mediaType: sourceDocumentPart.mediaType,
        title: sourceDocumentPart.title,
        filename: sourceDocumentPart.filename,
        providerMetadata: sourceDocumentPart.providerMetadata,
      } as InferUIMessageChunk<UI_MESSAGE>,
    ];
  }

  if (isDataUIPart(part)) {
    return [
      { type: part.type, data: part.data } as InferUIMessageChunk<UI_MESSAGE>,
    ];
  }

  const id = getPartId(originalChunks);

  if (part.type === 'text') {
    const textPart = part as TextUIPart;
    return [
      { type: 'text-start', id, providerMetadata: textPart.providerMetadata },
      { type: 'text-delta', id, delta: textPart.text },
      { type: 'text-end', id, providerMetadata: textPart.providerMetadata },
    ] as InferUIMessageChunk<UI_MESSAGE>[];
  }

  if (part.type === 'reasoning') {
    const reasoningPart = part as ReasoningUIPart;
    return [
      {
        type: 'reasoning-start',
        id,
        providerMetadata: reasoningPart.providerMetadata,
      },
      { type: 'reasoning-delta', id, delta: reasoningPart.text },
      {
        type: 'reasoning-end',
        id,
        providerMetadata: reasoningPart.providerMetadata,
      },
    ] as InferUIMessageChunk<UI_MESSAGE>[];
  }

  if (isToolOrDynamicToolUIPart(part)) {
    const dynamic = part.type === 'dynamic-tool';

    const chunks: InferUIMessageChunk<UI_MESSAGE>[] = [
      {
        type: 'tool-input-start',
        toolCallId: part.toolCallId,
        toolName: getToolOrDynamicToolName(part),
        dynamic,
        providerExecuted: part.providerExecuted,
      } as InferUIMessageChunk<UI_MESSAGE>,
    ];

    if (part.state === 'input-available' || part.state === 'output-available') {
      chunks.push({
        type: 'tool-input-available',
        toolCallId: part.toolCallId,
        toolName: getToolOrDynamicToolName(part),
        input: part.input,
        dynamic,
        providerExecuted: part.providerExecuted,
        providerMetadata: part.callProviderMetadata,
      } as InferUIMessageChunk<UI_MESSAGE>);
    }

    if (part.state === 'output-available') {
      chunks.push({
        type: 'tool-output-available',
        toolCallId: part.toolCallId,
        output: part.output,
        dynamic,
        providerExecuted: part.providerExecuted,
        preliminary: part.preliminary,
      } as InferUIMessageChunk<UI_MESSAGE>);
    } else if (part.state === 'output-error') {
      chunks.push({
        type: 'tool-output-error',
        toolCallId: part.toolCallId,
        errorText: part.errorText,
        dynamic,
        providerExecuted: part.providerExecuted,
      } as InferUIMessageChunk<UI_MESSAGE>);
    }

    return chunks;
  }

  return originalChunks;
}
