---
layout: post
title: "A few simple tricks nobody tells you when you start building chatbots"
date: 2026-08-01
description: Everyone eventually figures them out on their own, right?
tags: llm chatbot
categories:
_styles: |
  #markdown-content > p > img {
    display: block;
    max-width: min(100%, 800px);
    height: auto;
    margin: 1.75rem auto;
  }
---

Thanks to LLMs capable of reasoning and using 1M+ tokens in a single prompt, the simple chatbot has grown up. Now it is a proper agent with virtually all the digital capabilities of its human counterpart.

That said, hooking up the OpenAI API to a simple back-and-forth conversational system is still the entry point for AI engineering.

Even though 99.9% of this system's complexity is abstracted away behind the inference API, there are a few concepts that I see people take a while to grasp as they learn how to build systems on top of AI.

This short note discusses a few of them. I haven't found good sources or references for these ideas elsewhere, so I hope this note is useful.

## Efficient citation via clipboarding
LLM-driven search à la Perplexity may seem like a 2024 thing, but it is probably still one of the top three most valuable use cases for AI. In this workstream, being able to reference sources and efficiently show them to the user is critical.

Many starter kits for chatbots (e.g., Chainlit and Open WebUI) can easily render Markdown. Markdown allows you to write hyperlinks with a string like [Please click me!](https://www.youtube.com/watch?v=dQw4w9WgXcQ), which is easy to write and simple to remember.

LLMs are perfectly capable of writing hyperlink strings like these and require only a few short prompt instructions to do so. However, asking the LLM to write the hyperlink comes with a few limitations:

- While the user is waiting for the text to finish streaming, they will see partial hyperlink strings like `[Please click me!](https://www.yo..`. Once the string is complete, it will be rendered, causing the partial text to disappear and the hyperlink to appear in its place. This jump is irritating and creates a poor user experience.
- As system complexity and scope grow, you will likely begin citing sources with very long URLs that take a long time (>100 ms) to write token by token.
- This approach is token-inefficient because the same result can be produced more cheaply.

The solution is simple. As the reader, you have probably already implemented it or reinvented it on the spot.

Regardless, here it is: instruct the LLM to write short citation keys like `@d3` or `#a1`. A postprocessor sitting on top of the text stream then captures these keys and deterministically replaces them with full hyperlinks. I personally call this the "clipboard" approach in all of my codebases.

Some caveats do apply. Namely, you need some mechanism to give the source a nice descriptor (like "Annual Shareholder Report, 2026"). The naive approach from earlier lets you prompt the LLM to create the hyperlinked text on the fly; this approach doesn't let you do that out of the box.

This approach also separates citation styling from prompting logic. That separation is a good idea (IMO), but it can be a pain if you'd rather just write prompts instead of extending application logic. Visually, here's what it looks like:

![Two citation approaches shown while streaming text, including a citation hover tooltip.]({{ site.baseurl }}/images/2026-08-01-simple-things-ai-engineering/streaming-citations.gif)

You can also generalize this approach to any string that the LLM might generate inefficiently. If your LLM is powering an agent with tools for accessing APIs or reading files in a structured format, users may ask questions whose answers require the model to regurgitate data nearly verbatim. If the source is a long CSV or JSON file, having the LLM copy its contents token by token is wasteful. Use the clipboard approach to copy and paste the output instead.

Your choice of placeholder token is mostly unimportant, so long as it is small and unlikely to show up in the generated text for unrelated reasons. Avoid using standard formats like `[1]` or `[2]`, since hallucinating LLMs will typically generate false citations in those formats. For richer styling, you could probably use XML or a similar format.

## Constructing follow-up questions or actions
Most domain-specific AI platforms (e.g., Microsoft Copilot, Perplexity, and Glean) now supply recommended or follow-up questions for users to click. This is partly a gimmick from product managers to juice usage numbers, but it can sometimes be genuinely useful.

The best follow-up questions (e.g., "Would you like me to also search for restaurants closer to you?") know both 1) what the user originally asked and 2) how the system responded in the most recent turn. The standard approach I saw in a number of systems from 2025 and earlier was to run a specific prompt to make the LLM generate these questions.

Again, this is wasteful and inefficient. Nowadays, using structured generation plus streaming, you can have an LLM answer the user's question and generate the follow-ups in a single prompt using a schema like the following:

(show a schema that uses the OpenAI API to answer and generate follow-ups)

![The naive approach makes separate API calls for the response and follow-ups, while the improved approach returns both over one continuous text stream.]({{ site.baseurl }}/images/2026-08-01-simple-things-ai-engineering/follow-up-api-calls.gif)

The combined approach can be cost-neutral, as most of the instruction tokens from the follow-up prompt are moved into the main response prompt. In either case, the instructions should remain in the prompt cache 100% of the time, so the prefill cost of additional instructions is low relative to the decode cost. Since both approaches emit the same number of tokens, the overall cost of generation is mostly unchanged.

From a user-experience perspective, emitting the follow-up questions from the same stream as the main response probably saves at least 100 milliseconds by avoiding the overhead of a separate API call.

The primary weakness of a combined response/follow-up prompt is that it requires the LLM to follow more instructions per prompt, which generally leads to poorer adherence and effectiveness as the prompt grows. Open-weight models released since late 2025 generally do not have this issue in standard workflows. It may still be a problem if you have a very complex prompt with many instructions.

## Using cheap LLMs to capture user attention
Every few months, a new high-speed LLM inference provider comes on the market, and the industry reacts by touting the advantages of zero-lag conversations, instant coding agents, and so on. I think the market for that is actually very small. When we have faster LLMs, we don't simply make existing workflows faster; we generally use the extra latency headroom to make those workflows better. In other words, long-running AI tasks are going to be a fact of life.

Consequently, we need to resort to all sorts of tricks to keep the user on our page before they're emotionally spent from waiting 800 milliseconds for the next dopamine hit.

From an AI workflow perspective, this means providing continuous state updates on long-running tasks. Perplexity was an early leader here and had a great design that ensured something interesting was always happening on screen, even if you had to wait for 15+ seconds.

My go-to method is to task a very small, cheap LLM, such as Llama 3.2 1B or Gemma 2B, with providing short commentary snippets using prompts like this:

```
Your role is to provide one or two sentences of brief commentary on the state of a running program. Write at a high level, use concise language, and use the present tense.

Original user query:
{query}

Recent actions:
{tool_calls}
```

The output should be something like `Searched Salesforce cases from the last two weeks`. This can be hugely beneficial for keeping users engaged.

Make sure to keep the tool calls ordered from oldest to newest so that you can leverage the prompt cache effectively here. Putting the newest calls first invalidates the prompt cache. If you run this every 3 seconds during a 60-second task, you can easily send this prompt 20+ times, making cache management important *even within the scope of a single turn*.

OpenAI did this quite a bit with ChatGPT, as you can see below:

![ChatGPT displaying live progress updates while searching for nearby restaurants.]({{ site.baseurl }}/images/chatgpt-progress.gif)

If you have any more ideas to include here, let me know!
