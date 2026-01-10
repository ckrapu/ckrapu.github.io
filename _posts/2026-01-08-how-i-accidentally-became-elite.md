---
title: How I accidentally became a power AI user in big tech
layout: post
date: 2026-01-08
description: On becoming 99.9th percentile in token consumption
tags: [ai, cursor, software-development]
---

> **TL;DR:** I'm in a non-engineering department and somehow became an extreme outlier for AI-assisted coding usage because I have to wear too many hats and I'm picky about our dependencies.

---

Last spring, my company onboarded onto Cursor and shortly after, it was gaining traction in virtually every department. I didn't see myself as the target audience—my department doesn't ship product code and, before Cursor, I generally had a low-tech approach: I had autocomplete turned off most of the time, and my AI assistance was mostly limited to ChatGPT buried 50 tabs deep in Chrome.

I didn't know Cursor stats were visible to most managers, and when I found the department-level reports, I was genuinely worried at first. I was one of the more aggressive users of Claude 4 Opus at the time, and I thought I might be reprimanded for incurring too many API fees. 

The usage dashboard showed that I was in the top 0.XX% of users for activity (as measured by **Agent** requests), but also that I had a very unusual activity pattern of roughly 4x as many **Ask** requests (LLM only explains, like ChatGPT) as compared to **Agent** requests (LLM actually writes code in the project directory structure). This didn't surprise me much; I enjoy having a back-and-forth dialogue about all the technologies I don't know yet. I find this almost as much fun as writing the code itself. Most users appeared to be showing a 0.3-0.5x ratio. They were less interested in having the AI explain it and simply wanted the code to be written. I will also note that this is all good-faith activity; I wasn't churning out slop or using it for non-work purposes.

Despite being an early adopter of some AI technology (I started building around GPT-3 in early 2021) I genuinely enjoy typing, writing out functions, scaffolding, and the minutiae of writing code. To me, it's the same feeling as watching TV or making small talk. Most of it doesn't require a huge amount of thinking, and it feels pretty nice when you're in the flow. I never thought of myself as an AI-first, 10x developer type of person.

Here's what drove my usage:

### 1. Documentation gaps, rich tooling

We have an internal environment rich with tools, but light on documentation and frequently requiring workarounds. Documentation is lacking at many companies, and it's super common for a service's API to drift out of alignment with docs, especially when it's just an outdated guide someone cobbled together.

I spent a *lot* of effort working around this. AI was a panacea for this problem. Lots of nominally-working but hard-to-use services were now perfectly usable, all of a sudden, because the LLM has the patience to poke around the API, try a dozen examples, read a few different conflicting sources, and figure out how to actually make it work.

### 2. No devops engineer (and that's fine)

I don't have a devops engineer, I will never get one, and that's okay with me. Most of my projects are deployed via managed cloud services instead of k8s. Early on, I made the judgment call that, even in the best circumstances, our traffic was never going to exceed $$10^4$$ active users and cloud costs wouldn't be an issue, even unoptimized.

This means that I can largely control the entire apparatus via Terraform without needing an additional teammate to manage a k8s cluster.

### 3. The technical lead does analytics

In this case, that was me. Many teams operate by having BA or junior DS-type employees perform analytics and reporting. They're cheaper, but less likely to automate their work given less coding experience—and I say this as someone with a DS background.

Since the reporting needs fell on my shoulders, it was definitely in my interest to vibe code as many SQL queries and dashboarding functions as possible to short-circuit the route from raw data to management decisions.

### 4. Not waiting for MCPs when a `.cursorrules` and some bash aliases would suffice
Pretty much all companies' IT departments have been talking about MCP servers for the last year. Some are further along than others. I, being inordinately impatient, just wrote a few bash aliases to make API calls and, for many operational tasks ("run a report against our sales data", "find all the newest companies we haven't reached out to yet", etc.), I could just tell Cursor that it had these command line tools for using Perplexity, Databricks, our company's Powerpoint template, etc. 

### 5. A great full-stack template

Many of the similar projects I'd seen at my company and elsewhere used software like Streamlit, Gradio, and OpenWebUI. These are fine, but they aren't ready out-of-the-box for going into production.

I invested a *lot* of time in early 2024 figuring out how we'd deploy an internal chatbot and eventually landed on [Chainlit](https://docs.chainlit.io/get-started/overview). This was a stroke of luck as it was a solid template with batteries included for OAuth, telemetry, human-in-the-loop, and other features that many teams were authoring from scratch at that point. It had quirks (the default schema tanks after 100k conversations) but otherwise was great to work with. It also pushed us to work mostly in FastAPI + LlamaIndex + SQLAlchemy, all of which turned out to be rock-solid in production.

Ironically, this OSS project was a much better base than all of the paid consulting/dev shop-produced products that I'd seen. 

By scaffolding around a quality codebase with sensible defaults, clean API structure, and a minimal but appealing UI, grounding on that foundation gave us much more effective LLM-produced code than if we had vibe coded from scratch.

These effects were compounding for me: it produced nice code which I enjoyed working on, which in turn made me want to do more of it instead of getting frustrated by opaque errors.

