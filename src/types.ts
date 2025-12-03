import type { UIDataTypes, UIMessage, UIMessagePart, UITools } from 'ai';

export type InferUIMessageMetadata<UI_MESSAGE extends UIMessage> =
  UI_MESSAGE extends UIMessage<infer METADATA> ? METADATA : unknown;

export type InferUIMessageData<UI_MESSAGE extends UIMessage> =
  UI_MESSAGE extends UIMessage<unknown, infer DATA_TYPES>
    ? DATA_TYPES
    : UIDataTypes;

export type InferUIMessageTools<UI_MESSAGE extends UIMessage> =
  UI_MESSAGE extends UIMessage<unknown, UIDataTypes, infer TOOLS>
    ? TOOLS
    : UITools;

export type InferUIMessagePart<UI_MESSAGE extends UIMessage> = UIMessagePart<
  InferUIMessageData<UI_MESSAGE>,
  InferUIMessageTools<UI_MESSAGE>
>;

export type InferUIMessagePartType<UI_MESSAGE extends UIMessage> =
  InferUIMessagePart<UI_MESSAGE>['type'];

/**
 * A partial part reconstructed from the current chunk.
 * Contains the part type and any available data from the chunk.
 */
export type InferPartialUIMessagePart<UI_MESSAGE extends UIMessage> = {
  /** The part type (e.g., 'text', 'reasoning', 'tool-weather', 'file') */
  type: InferUIMessagePartType<UI_MESSAGE>;
} & Partial<InferUIMessagePart<UI_MESSAGE>>;
