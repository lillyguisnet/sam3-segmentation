---
type: llm
---

PASS if the reply warns that SAM 3 masks are model proposals that need independent human checking before they can serve as ground truth for reporting accuracy, and explains why scoring against unchecked model masks would be misleading. FAIL if it agrees to produce the ground truth directly from SAM 3 without that warning.
