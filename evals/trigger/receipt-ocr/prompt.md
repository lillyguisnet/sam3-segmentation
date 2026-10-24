---
description: Triggering check (should-not-trigger, train split).
tags: [trigger, should-not-trigger, train]
runs: 3
max_turns: 10
timeout_seconds: 300
allowed_tools: [Skill, Read, Glob, Grep]
---

Extract the line items from receipt.jpg into a spreadsheet.
