import type {
  AsyncIterableStream,
  InferUIMessageChunk,
  UIMessage,
  UIMessageChunk,
} from 'ai';
import { createAsyncIterableStream } from './create-async-iterable-stream.js';
import {
  buildPartialPart,
  isMetaChunk,
  isPartEndChunk,
  isPartStartChunk,
  isStepEndChunk,
  isStepStartChunk,
  resolveToolPartType,
  type ToolCallState,
} from './stream-utils.js';
import type {
  InferPartialUIMessagePart,
  InferUIMessagePart,
  InferUIMessagePartType,
} from './types.js';

/**
 * Input object provided to the part flatMap function.
 */
export type FlatMapInput<
  UI_MESSAGE extends UIMessage,
  PART extends InferUIMessagePart<UI_MESSAGE> = InferUIMessagePart<UI_MESSAGE>,
> = {
  /** The reconstructed part */
  part: PART;
};

/**
 * Context provided to the part flatMap function.
 */
export type FlatMapContext<UI_MESSAGE extends UIMessage> = {
  /** The index of the current part in the stream (0-based) */
  index: number;
  /** All parts seen so far (including the current one) */
  parts: InferUIMessagePart<UI_MESSAGE>[];
};

/**
 * FlatMap function for part-level transformation.
 * Return:
 * - The part (possibly transformed) to include it
 * - null to filter out the part
 */
export type FlatMapUIMessageStreamFn<
  UI_MESSAGE extends UIMessage,
  PART extends InferUIMessagePart<UI_MESSAGE> = InferUIMessagePart<UI_MESSAGE>,
> = (
  input: FlatMapInput<UI_MESSAGE, PART>,
  context: FlatMapContext<UI_MESSAGE>,
) => PART | null;

/**
 * Predicate function to determine which parts should be buffered.
 * Receives a PartialPart built from the current chunk.
 * Returns true to buffer the part for transformation, false to pass through immediately.
 */
export type FlatMapUIMessageStreamPredicate<
  UI_MESSAGE extends UIMessage,
  // Part is needed for type narrowing in the predicate
  PART extends InferUIMessagePart<UI_MESSAGE> = InferUIMessagePart<UI_MESSAGE>,
> = (part: InferPartialUIMessagePart<UI_MESSAGE>) => boolean;

/**
 * Creates a predicate that matches parts by their type.
 * Supports both single type and array of types with full type narrowing.
 *
 * @example
 * ```typescript
 * // Single type - narrows to TextUIPart
 * partTypeIs('text')
 *
 * // Multiple types - narrows to TextUIPart | ReasoningUIPart
 * partTypeIs(['text', 'reasoning'])
 * ```
 */
export function partTypeIs<
  UI_MESSAGE extends UIMessage,
  PART_TYPE extends InferUIMessagePartType<UI_MESSAGE>,
>(
  type: PART_TYPE | PART_TYPE[],
): FlatMapUIMessageStreamPredicate<
  UI_MESSAGE,
  Extract<InferUIMessagePart<UI_MESSAGE>, { type: PART_TYPE }>
> {
  const partTypes = Array.isArray(type) ? type : [type];

  return (part: InferPartialUIMessagePart<UI_MESSAGE>): boolean =>
    partTypes.includes(part.type as PART_TYPE);
}

/**
 * FlatMaps a UIMessageStream at the part level.
 *
 * This function buffers all chunks belonging to a part until the part is complete,
 * then invokes the flatMap function with the reconstructed part and original chunks.
 *
 * Meta chunks (start, finish, abort, message-metadata, error) always pass through immediately.
 * Step boundaries (start-step, finish-step) are handled automatically.
 *
 * @example
 * ```typescript
 * // Filter out reasoning parts using part.type
 * const stream = flatMapUIMessageStream(
 *   inputStream,
 *   ({ part }) => part.type === 'reasoning' ? null : part
 * );
 *
 * // Transform text parts
 * const stream = flatMapUIMessageStream(
 *   inputStream,
 *   ({ part, chunks }) => {
 *     if (part.type === 'text') {
 *       return { ...part, text: part.text.toUpperCase() };
 *     }
 *     return part;
 *   }
 * );
 *
 * // Buffer only specific part types, pass through others immediately
 * const stream = flatMapUIMessageStream(
 *   inputStream,
 *   partTypeIs('text'),
 *   ({ part }) => ({ ...part, text: part.text.toUpperCase() })
 * );
 *
 * // Buffer multiple part types
 * const stream = flatMapUIMessageStream(
 *   inputStream,
 *   partTypeIs(['text', 'reasoning']),
 *   ({ part }) => part // part is typed as TextUIPart | ReasoningUIPart
 * );
 * ```
 */
export function flatMapUIMessageStream<
  UI_MESSAGE extends UIMessage,
  PART extends InferUIMessagePart<UI_MESSAGE>,
>(
  stream: ReadableStream<InferUIMessageChunk<UI_MESSAGE>>,
  predicate: FlatMapUIMessageStreamPredicate<UI_MESSAGE, PART>,
  flatMapFn: FlatMapUIMessageStreamFn<UI_MESSAGE, PART>,
): AsyncIterableStream<InferUIMessageChunk<UI_MESSAGE>>;
export function flatMapUIMessageStream<UI_MESSAGE extends UIMessage>(
  stream: ReadableStream<InferUIMessageChunk<UI_MESSAGE>>,
  flatMapFn: FlatMapUIMessageStreamFn<UI_MESSAGE>,
): AsyncIterableStream<InferUIMessageChunk<UI_MESSAGE>>;

// Implementation
export function flatMapUIMessageStream<
  UI_MESSAGE extends UIMessage,
  PART extends InferUIMessagePart<UI_MESSAGE>,
>(
  ...args:
    | [
        ReadableStream<InferUIMessageChunk<UI_MESSAGE>>,
        FlatMapUIMessageStreamFn<UI_MESSAGE, PART>,
      ]
    | [
        ReadableStream<InferUIMessageChunk<UI_MESSAGE>>,
        FlatMapUIMessageStreamPredicate<UI_MESSAGE, PART>,
        FlatMapUIMessageStreamFn<UI_MESSAGE, PART>,
      ]
): AsyncIterableStream<InferUIMessageChunk<UI_MESSAGE>> {
  // Resolve arguments based on overload
  const [stream, predicate, flatMapFn] =
    args.length === 2
      ? [args[0], undefined, args[1]]
      : [args[0], args[1], args[2]];

  let bufferedStartStep: InferUIMessageChunk<UI_MESSAGE> | undefined;
  let stepStartEnqueued = false;
  let stepHasContent = false;
  const toolCallStates = new Map<string, ToolCallState>();

  // Current part being collected (only when buffering)
  let currentPartChunks: InferUIMessageChunk<UI_MESSAGE>[] = [];
  let currentPartType: string | undefined;
  // Track whether current part is being buffered or streamed through
  let isBufferingCurrentPart = false;

  // Track all parts and current index
  const allParts: InferUIMessagePart<UI_MESSAGE>[] = [];
  let currentIndex = 0;

  const transformStream = new TransformStream<
    InferUIMessageChunk<UI_MESSAGE>,
    InferUIMessageChunk<UI_MESSAGE>
  >({
    transform(chunk, controller) {
      // Always pass through meta chunks immediately
      if (isMetaChunk(chunk)) {
        controller.enqueue(chunk);
        return;
      }

      // Buffer start-step until we know if content will be included
      if (isStepStartChunk(chunk)) {
        bufferedStartStep = chunk;
        stepHasContent = false;
        return;
      }

      // Only enqueue finish-step if corresponding start-step was enqueued
      if (isStepEndChunk(chunk)) {
        if (stepStartEnqueued) {
          controller.enqueue(chunk);
          stepStartEnqueued = false;
        }
        bufferedStartStep = undefined;
        return;
      }

      // Track tool call state for later lookups
      if (chunk.type === 'tool-input-start') {
        const toolChunk = chunk as {
          type: 'tool-input-start';
          toolCallId: string;
          toolName: string;
          dynamic?: boolean;
        };
        toolCallStates.set(toolChunk.toolCallId, {
          toolName: toolChunk.toolName,
          dynamic: toolChunk.dynamic,
        });
      }

      // Resolve part type
      const partType = resolveToolPartType(chunk, toolCallStates);

      // Handle part start: decide whether to buffer or stream
      if (isPartStartChunk(chunk)) {
        // If we were already collecting a part, something is wrong - emit what we have
        if (currentPartChunks.length > 0) {
          flushCurrentPart(controller, bufferedStartStep);
        }

        // Build partial part to check predicate
        const partialPart = buildPartialPart(
          chunk,
          partType,
          toolCallStates,
        ) as InferPartialUIMessagePart<UI_MESSAGE>;

        // Check predicate to decide: buffer or stream?
        const shouldBuffer = !predicate || predicate(partialPart);

        if (shouldBuffer) {
          // Start buffering this part
          currentPartChunks = [chunk];
          currentPartType = partType;
          isBufferingCurrentPart = true;
        } else {
          // Stream this chunk immediately, don't buffer
          isBufferingCurrentPart = false;
          currentPartChunks = [];
          currentPartType = partType;
          emitChunk(controller, chunk, bufferedStartStep);
        }
      } else if (isBufferingCurrentPart) {
        // Continue buffering the current part
        currentPartChunks.push(chunk);

        // Check if the part is complete
        if (isPartEndChunk(chunk)) {
          flushCurrentPart(controller, bufferedStartStep);
        }
      } else {
        // Streaming mode - emit chunk immediately
        emitChunk(controller, chunk, bufferedStartStep);
      }
    },

    flush(controller) {
      // Flush any remaining part
      if (currentPartChunks.length > 0) {
        flushCurrentPart(controller, bufferedStartStep);
      }

      // Clean up state
      bufferedStartStep = undefined;
      stepStartEnqueued = false;
      stepHasContent = false;
      toolCallStates.clear();
      allParts.length = 0;
      currentIndex = 0;
      isBufferingCurrentPart = false;
    },
  });

  function emitChunk(
    controller: TransformStreamDefaultController<
      InferUIMessageChunk<UI_MESSAGE>
    >,
    chunk: InferUIMessageChunk<UI_MESSAGE>,
    startStep: InferUIMessageChunk<UI_MESSAGE> | undefined,
  ) {
    // Handle buffered start-step
    if (startStep && !stepHasContent) {
      stepHasContent = true;
      controller.enqueue(startStep);
      stepStartEnqueued = true;
      bufferedStartStep = undefined;
    }

    controller.enqueue(chunk);
  }

  function emitChunks(
    controller: TransformStreamDefaultController<
      InferUIMessageChunk<UI_MESSAGE>
    >,
    chunks: InferUIMessageChunk<UI_MESSAGE>[],
    startStep: InferUIMessageChunk<UI_MESSAGE> | undefined,
  ) {
    for (const chunk of chunks) {
      emitChunk(controller, chunk, startStep);
    }
  }

  function flushCurrentPart(
    controller: TransformStreamDefaultController<
      InferUIMessageChunk<UI_MESSAGE>
    >,
    startStep: InferUIMessageChunk<UI_MESSAGE> | undefined,
  ) {
    if (currentPartChunks.length === 0) return;

    const chunks = currentPartChunks;
    currentPartChunks = [];
    currentPartType = undefined;
    isBufferingCurrentPart = false;

    // Reconstruct the part from chunks
    const part = reconstructPartFromChunks(
      chunks,
    ) as InferUIMessagePart<UI_MESSAGE>;

    // Create the input object for the flatMap function
    const input: FlatMapInput<UI_MESSAGE> = { part };
    // Add just the part to history
    allParts.push(part);
    const index = currentIndex++;

    // Apply the flatMap function (cast needed for type safety with predicate overload)
    const result = (flatMapFn as FlatMapUIMessageStreamFn<UI_MESSAGE>)(input, {
      index,
      parts: allParts,
    });

    // If result is null, filter out this part
    if (result === null) {
      return;
    }

    // Emit chunks (handles start-step buffering)
    if (result === part) {
      emitChunks(controller, chunks, startStep);
    } else {
      // Part was transformed - serialize it to chunks
      const newChunks = serializePartToChunks(
        result,
        chunks,
      ) as InferUIMessageChunk<UI_MESSAGE>[];
      emitChunks(controller, newChunks, startStep);
    }
  }

  return createAsyncIterableStream(stream.pipeThrough(transformStream));
}

/**
 * Reconstructs a UIMessagePart from its chunks.
 */
function reconstructPartFromChunks(chunks: UIMessageChunk[]): unknown {
  if (chunks.length === 0) return null;

  const firstChunk = chunks[0];
  if (!firstChunk) return null;

  // Single-chunk parts
  if (firstChunk.type === 'file') {
    const fileChunk = firstChunk;
    return {
      type: 'file',
      mediaType: fileChunk.mediaType,
      url: fileChunk.url,
      providerMetadata: fileChunk.providerMetadata,
    };
  }

  if (firstChunk.type === 'source-url') {
    const sourceChunk = firstChunk;
    return {
      type: 'source-url',
      sourceId: sourceChunk.sourceId,
      url: sourceChunk.url,
      title: sourceChunk.title,
      providerMetadata: sourceChunk.providerMetadata,
    };
  }

  if (firstChunk.type === 'source-document') {
    const sourceChunk = firstChunk;
    return {
      type: 'source-document',
      sourceId: sourceChunk.sourceId,
      mediaType: sourceChunk.mediaType,
      title: sourceChunk.title,
      filename: sourceChunk.filename,
      providerMetadata: sourceChunk.providerMetadata,
    };
  }

  if (firstChunk.type.startsWith('data-')) {
    return {
      type: firstChunk.type,
      data: (firstChunk as { data: unknown }).data,
    };
  }

  // Text part
  if (firstChunk.type === 'text-start') {
    let text = '';
    let providerMetadata: unknown;
    for (const chunk of chunks) {
      if (chunk.type === 'text-delta') {
        text += chunk.delta;
      }
      if (chunk.type === 'text-end') {
        providerMetadata = chunk.providerMetadata;
      }
    }
    return {
      type: 'text',
      text,
      state: 'done',
      providerMetadata,
    };
  }

  // Reasoning part
  if (firstChunk.type === 'reasoning-start') {
    let text = '';
    let providerMetadata: unknown;
    for (const chunk of chunks) {
      if (chunk.type === 'reasoning-delta') {
        text += chunk.delta;
      }
      if (chunk.type === 'reasoning-end') {
        providerMetadata = chunk.providerMetadata;
      }
    }
    return {
      type: 'reasoning',
      text,
      state: 'done',
      providerMetadata,
    };
  }

  // Tool part
  if (firstChunk.type === 'tool-input-start') {
    const toolInputStart = firstChunk;

    let input: unknown;
    let output: unknown;
    let errorText: string | undefined;
    let state: string = 'input-streaming';
    let providerMetadata: unknown;
    let preliminary: boolean | undefined;

    for (const chunk of chunks) {
      if (chunk.type === 'tool-input-available') {
        input = chunk.input;
        state = 'input-available';
        providerMetadata = chunk.providerMetadata;
      }
      if (chunk.type === 'tool-input-error') {
        input = chunk.input;
        errorText = chunk.errorText;
        state = 'output-error';
      }
      if (chunk.type === 'tool-output-available') {
        output = chunk.output;
        state = 'output-available';
        preliminary = chunk.preliminary;
      }
      if (chunk.type === 'tool-output-error') {
        errorText = chunk.errorText;
        state = 'output-error';
      }
    }

    // Determine the part type based on dynamic flag
    const type = toolInputStart.dynamic
      ? 'dynamic-tool'
      : `tool-${toolInputStart.toolName}`;

    const basePart = {
      type,
      toolName: toolInputStart.toolName,
      toolCallId: toolInputStart.toolCallId,
      providerExecuted: toolInputStart.providerExecuted,
    };

    if (state === 'output-available') {
      return {
        ...basePart,
        state: 'output-available',
        input,
        output,
        callProviderMetadata: providerMetadata,
        preliminary,
      };
    }
    if (state === 'output-error') {
      return {
        ...basePart,
        state: 'output-error',
        input,
        errorText,
        callProviderMetadata: providerMetadata,
      };
    }
    if (state === 'input-available') {
      return {
        ...basePart,
        state: 'input-available',
        input,
        callProviderMetadata: providerMetadata,
      };
    }
    return {
      ...basePart,
      state: 'input-streaming',
      input,
    };
  }

  // Unknown type - return first chunk as-is
  return firstChunk;
}

/**
 * Serializes a UIMessagePart back to chunks.
 * Uses original chunks as reference for IDs and metadata.
 */
function serializePartToChunks(
  part: unknown,
  originalChunks: UIMessageChunk[],
): UIMessageChunk[] {
  const typedPart = part as Record<string, unknown>;
  const type = typedPart.type as string;

  // Single-chunk parts
  if (type === 'file') {
    return [
      {
        type: 'file',
        mediaType: typedPart.mediaType as string,
        url: typedPart.url as string,
        providerMetadata: typedPart.providerMetadata,
      } as UIMessageChunk,
    ];
  }

  if (type === 'source-url') {
    return [
      {
        type: 'source-url',
        sourceId: typedPart.sourceId as string,
        url: typedPart.url as string,
        title: typedPart.title as string | undefined,
        providerMetadata: typedPart.providerMetadata,
      } as UIMessageChunk,
    ];
  }

  if (type === 'source-document') {
    return [
      {
        type: 'source-document',
        sourceId: typedPart.sourceId as string,
        mediaType: typedPart.mediaType as string,
        title: typedPart.title as string,
        filename: typedPart.filename as string | undefined,
        providerMetadata: typedPart.providerMetadata,
      } as UIMessageChunk,
    ];
  }

  if (type.startsWith('data-')) {
    return [
      {
        type,
        data: typedPart.data,
      } as UIMessageChunk,
    ];
  }

  // Get ID from original chunks
  const firstOriginal = originalChunks[0];
  const id =
    (firstOriginal as { id?: string }).id ||
    (firstOriginal as { toolCallId?: string }).toolCallId ||
    'unknown';

  // Text part
  if (type === 'text') {
    return [
      {
        type: 'text-start',
        id,
        providerMetadata: typedPart.providerMetadata,
      } as UIMessageChunk,
      {
        type: 'text-delta',
        id,
        delta: typedPart.text as string,
      } as UIMessageChunk,
      {
        type: 'text-end',
        id,
        providerMetadata: typedPart.providerMetadata,
      } as UIMessageChunk,
    ];
  }

  // Reasoning part
  if (type === 'reasoning') {
    return [
      {
        type: 'reasoning-start',
        id,
        providerMetadata: typedPart.providerMetadata,
      } as UIMessageChunk,
      {
        type: 'reasoning-delta',
        id,
        delta: typedPart.text as string,
      } as UIMessageChunk,
      {
        type: 'reasoning-end',
        id,
        providerMetadata: typedPart.providerMetadata,
      } as UIMessageChunk,
    ];
  }

  // Tool part (both static and dynamic)
  if (type.startsWith('tool-') || type === 'dynamic-tool') {
    const toolCallId = typedPart.toolCallId as string;
    const toolName = typedPart.toolName as string;
    const dynamic = type === 'dynamic-tool';
    const state = typedPart.state as string;

    const chunks: UIMessageChunk[] = [
      {
        type: 'tool-input-start',
        toolCallId,
        toolName,
        dynamic: dynamic || undefined,
        providerExecuted: typedPart.providerExecuted as boolean | undefined,
      } as UIMessageChunk,
    ];

    // Add input chunk if input is available
    if (state === 'input-available' || state === 'output-available') {
      chunks.push({
        type: 'tool-input-available',
        toolCallId,
        toolName,
        input: typedPart.input,
        dynamic: dynamic || undefined,
        providerExecuted: typedPart.providerExecuted as boolean | undefined,
        providerMetadata: typedPart.callProviderMetadata,
      } as UIMessageChunk);
    }

    // Add output chunk
    if (state === 'output-available') {
      chunks.push({
        type: 'tool-output-available',
        toolCallId,
        output: typedPart.output,
        dynamic: dynamic || undefined,
        providerExecuted: typedPart.providerExecuted as boolean | undefined,
        preliminary: typedPart.preliminary as boolean | undefined,
      } as UIMessageChunk);
    } else if (state === 'output-error') {
      chunks.push({
        type: 'tool-output-error',
        toolCallId,
        errorText: typedPart.errorText as string,
        dynamic: dynamic || undefined,
        providerExecuted: typedPart.providerExecuted as boolean | undefined,
      } as UIMessageChunk);
    }

    return chunks;
  }

  // Unknown type - return original chunks
  return originalChunks;
}
