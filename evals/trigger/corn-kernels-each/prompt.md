---
description: Triggering check (should-trigger, train split).
tags: [trigger, should-trigger, train]
runs: 3
max_turns: 10
timeout_seconds: 300
allowed_tools: [Skill, Read, Glob, Grep]
---

I've got ~/data/field/IMG_2041.jpg of a corn ear. Can you outline every single kernel separately and give me masks I can load in Python?
