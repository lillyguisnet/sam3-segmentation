---
description: Triggering check (should-trigger, train split).
tags: [trigger, should-trigger, train]
runs: 3
max_turns: 10
timeout_seconds: 300
allowed_tools: [Skill, Read, Glob, Grep]
---

We need COCO-format instance annotations for the jar photos in ./shelf_photos for our detector's training set. Can you generate first-pass masks that we then fix by hand?
