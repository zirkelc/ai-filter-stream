import type { UIMessage } from 'ai';

export type InferUIMessagePart<UI_MESSAGE extends UIMessage> =
  UI_MESSAGE['parts'][number];

export type InferUIMessagePartType<UI_MESSAGE extends UIMessage> =
  InferUIMessagePart<UI_MESSAGE>['type'];
