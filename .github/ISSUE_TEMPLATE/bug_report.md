---
name: Bug report
about: Report a problem with AmorphGen
title: "[BUG] "
labels: bug
assignees: ''
---

## Summary

A clear, one-sentence description of what's wrong.

## Steps to reproduce

Minimal example that triggers the bug. Please include:

1. The exact CLI command **or** Python snippet
2. The input file (or its composition / formula)
3. Any non-default config / YAML

```bash
# example
amorphgen --random-gen --composition "SiO2*16" --device cpu --model chgnet
```

## Expected behaviour

What you expected to happen.

## Actual behaviour

What actually happened. Paste the full traceback or error message if there was one (inside a ```` ``` ```` code block).

## Environment

- OS: <!-- e.g. macOS 14.5, Ubuntu 22.04, Rocky Linux 9 -->
- Python version: <!-- `python --version` -->
- AmorphGen version: <!-- `pip show amorphgen | grep Version` -->
- Calculator backend: <!-- mace-mpa-0 / chgnet / sevennet / classical -->
- Device: <!-- cpu / cuda / mps -->
- Relevant package versions: <!-- `pip show ase mace-torch chgnet | grep -E "Name|Version"` -->

## Additional context

Anything else that might help — log files, partial trajectory output, hardware (GPU model), etc.
