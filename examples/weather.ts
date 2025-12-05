import { openai } from '@ai-sdk/openai';
import {
  type InferUITools,
  readUIMessageStream,
  stepCountIs,
  streamText,
  tool,
  type UIMessage,
} from 'ai';
import { z } from 'zod';
import {
  excludeParts,
  filterUIMessageStream,
} from '../src/filter-ui-message-stream.js';
import {
  flatMapUIMessageStream,
  partTypeIs,
} from '../src/flat-map-ui-message-stream.js';
import { mapUIMessageStream } from '../src/map-ui-message-stream.js';

process.env.OPENAI_API_KEY = '';

export type MyMetadata = { id: string };
export type MyDataPart = { weather: { location: string; temperature: number } };
export type MyTools = InferUITools<typeof tools>;
export type MyUIMessage = UIMessage<MyMetadata, MyDataPart, MyTools>;

const toFahrenheit = (temperature: number) => {
  return (temperature * 9) / 5 + 32;
};

const tools = {
  weather: tool({
    description: 'Get the weather in a location',
    inputSchema: z.object({
      location: z.string().describe('The location to get the weather for'),
    }),
    execute: ({ location }) => ({
      location,
      temperature: 72 + Math.floor(Math.random() * 21) - 10,
      unit: 'C',
    }),
  }),
};

const result = streamText({
  model: openai('gpt-5'),
  prompt: 'What is the weather in Tokyo?',
  tools,
  stopWhen: stepCountIs(5),
});

const uiMessageStream = result.toUIMessageStream<MyUIMessage>();

const filteredStream = filterUIMessageStream(
  uiMessageStream,
  excludeParts(['reasoning']),
);

// Map: Transform text chunks to uppercase
const mappedStream = mapUIMessageStream(filteredStream, ({ chunk }) => {
  if (chunk.type === 'text-delta') {
    return { ...chunk, delta: chunk.delta.toUpperCase() };
  }
  return chunk;
});

// FlatMap: Buffer tool-call chunks until part is complete, then transform the part
const flatMappedStream = flatMapUIMessageStream(
  mappedStream,
  partTypeIs('tool-weather'),
  ({ part }) => {
    if (part.type === 'tool-weather' && part.state === 'output-available') {
      const { output } = part;
      return {
        ...part,
        output: {
          ...output,
          temperature: toFahrenheit(output.temperature),
          unit: 'F',
        },
      };
    }

    return part;
  },
);

for await (const message of readUIMessageStream({
  stream: flatMappedStream,
})) {
  for (const part of message.parts) {
    console.log(part);
  }
}
