import type {
  AsyncIterableStream,
  InferUIMessageChunk,
  UIMessage,
  UIMessageChunk,
} from 'ai';
import {
  type MapContext,
  type MapInput,
  mapUIMessageStream,
} from './map-ui-message-stream.js';
import type { InferUIMessagePartType } from './types.js';

/**
 * Filter function that receives the same input/context as mapUIMessageStream.
 * Return true to include the chunk, false to filter it out.
 */
export type FilterUIMessageStreamPredicate<UI_MESSAGE extends UIMessage> = (
  input: MapInput<UI_MESSAGE>,
  context: MapContext<UI_MESSAGE>,
) => boolean;

/**
 * Creates a filter predicate that includes only the specified part types.
 *
 * @example
 * ```typescript
 * filterUIMessageStream(stream, includeParts(['text', 'tool-weather']));
 * ```
 */
export function includeParts<UI_MESSAGE extends UIMessage>(
  includePartTypes: Array<InferUIMessagePartType<UI_MESSAGE>>,
): FilterUIMessageStreamPredicate<UI_MESSAGE> {
  return ({ part }) => {
    const partType = part.type as InferUIMessagePartType<UI_MESSAGE>;
    return includePartTypes.includes(partType);
  };
}

/**
 * Creates a filter predicate that excludes the specified part types.
 *
 * @example
 * ```typescript
 * filterUIMessageStream(stream, excludeParts(['reasoning', 'tool-calculator']));
 * ```
 */
export function excludeParts<UI_MESSAGE extends UIMessage>(
  excludePartTypes: Array<InferUIMessagePartType<UI_MESSAGE>>,
): FilterUIMessageStreamPredicate<UI_MESSAGE> {
  return ({ part }) => {
    const partType = part.type as InferUIMessagePartType<UI_MESSAGE>;
    return !excludePartTypes.includes(partType);
  };
}

/**
 * Filters a UIMessageStream to include or exclude specific chunks.
 *
 * This is a convenience wrapper around `mapUIMessageStream` that provides
 * a simpler API for filtering chunks.
 *
 * The filter function receives `{ chunk, part }` and `{ index, chunks }` and returns
 * a boolean indicating whether to include the chunk.
 *
 * Use the `includeParts()` and `excludeParts()` helper functions for common filtering patterns.
 *
 * Meta chunks (start, finish, abort, message-metadata, error) always pass through.
 * Step boundaries (start-step, finish-step) are handled automatically.
 *
 * @example
 * ```typescript
 * // Custom filter function - include only text parts
 * const stream = filterUIMessageStream(
 *   result.toUIMessageStream(),
 *   ({ part }) => part.type === 'text'
 * );
 *
 * // Using includeParts helper
 * const stream = filterUIMessageStream(
 *   result.toUIMessageStream(),
 *   includeParts(['text', 'tool-weather'])
 * );
 *
 * // Using excludeParts helper
 * const stream = filterUIMessageStream(
 *   result.toUIMessageStream(),
 *   excludeParts(['reasoning', 'tool-calculator'])
 * );
 *
 * // Filter with context - access index and previous chunks
 * const stream = filterUIMessageStream(
 *   result.toUIMessageStream(),
 *   ({ part }, { index }) => part.type === 'text' && index < 10
 * );
 * ```
 */
export function filterUIMessageStream<UI_MESSAGE extends UIMessage>(
  stream: ReadableStream<InferUIMessageChunk<UI_MESSAGE>>,
  predicate: FilterUIMessageStreamPredicate<UI_MESSAGE>,
): AsyncIterableStream<InferUIMessageChunk<UI_MESSAGE>> {
  return mapUIMessageStream(stream, (input, context) => {
    return predicate(input, context) ? input.chunk : null;
  });
}
