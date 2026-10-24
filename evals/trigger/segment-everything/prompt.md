---
description: Triggering check (should-trigger, train split).
tags: [trigger, should-trigger, train]
runs: 3
max_turns: 10
timeout_seconds: 300
allowed_tools: [Skill, Read, Glob, Grep]
---

run segment-everything on kitchen.jpg, I want to see what objects the model finds on its own
