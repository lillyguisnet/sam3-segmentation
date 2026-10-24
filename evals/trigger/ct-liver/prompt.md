---
description: Triggering check (should-not-trigger, validation split).
tags: [trigger, should-not-trigger, validation]
runs: 3
max_turns: 10
timeout_seconds: 300
allowed_tools: [Skill, Read, Glob, Grep]
---

Segment the liver in this abdominal CT series (DICOM files in ./scan) for surgical planning.
