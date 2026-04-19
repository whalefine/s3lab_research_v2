---
name: pytorch-function-rewrite
description: >-
  Retrieve real PyTorch implementation details for a specific API (for example
  nn.LayerNorm), then create a Python reimplementation saved as lib/module/<function>.py.
  Use when the user asks to replace a PyTorch built-in with custom code, inspect
  internal implementation, port C++ kernels into Python, or compare behavior/risk
  versus original PyTorch functions.
---

# PyTorch Function Rewrite

Find how a target PyTorch function is actually implemented, then produce a local Python module under `lib/module`.

## Scope

Use this skill when the user asks things like:
- "把 `nn.xxx` 換成自己寫的"
- "查 `torch.xxx` 內部怎麼寫"
- "把 Python以外的程式語言 實作轉成 Python 版本"

## Mandatory workflow

### 1) Confirm target and output name

Extract:
- target API (for example `torch.nn.LayerNorm` or `torch.nn.functional.layer_norm`)
- expected file name

Default output path:
- `lib/module/<normalized_function_name>.py`

Name rules:
- use lowercase snake_case
- strip namespace prefixes (`torch.nn.` / `torch.`)
- examples:
  - `nn.LayerNorm` -> `lib/module/layer_norm.py`
  - `torch.nn.functional.gelu` -> `lib/module/gelu.py`

### 2) Locate official implementation from source

Always prioritize official upstream evidence in this order:
1. PyTorch GitHub source (`pytorch/pytorch`)
2. PyTorch official docs with `[source]` links
3. Maintainer discussions or RFC/PR notes only as supplemental context

Collect and cite:
- exact Python entry point (`nn.Module.forward` / `functional` API)
- whether core op dispatches to ATen/C++/CUDA kernel
- important defaults (`eps`, `unbiased`, shape rules, dtype behavior)

### 3) Determine implementation type

Classify target function:
- **Pure Python path**: logic is mostly Python and directly reproducible
- **Hybrid path**: Python wrapper + C++ kernel call
- **Kernel path**: nearly all logic in C++/CUDA

### 4) Create local Python module in `lib/module`

If source is Python:
- copy essential logic faithfully (not comments/license banner)
- keep API-compatible function/class signature when practical

If source is C++/CUDA/other language:
- implement equivalent Python version using `torch` tensor ops
- preserve semantics first, then readability

Module requirements:
- include a short top docstring with:
  - target original API
  - upstream reference URL(s)
  - PyTorch version/commit if identifiable
- include minimal self-check helper at bottom:
  - compare output with original API on random tensors
  - print max absolute error

### 5) Report differences and risks (required)

After writing module, always report:
1. **Behavior differences**  
   - numeric tolerance gaps
   - broadcast/shape corner cases
   - dtype/device differences (CPU/CUDA/bfloat16/float16)
2. **Performance differences**  
   - expected slowdown vs fused kernel
   - memory overhead
3. **Training/autograd risks**  
   - gradient stability
   - in-place behavior differences
4. **Deployment risks**  
   - TorchScript/ONNX export changes
   - quantization or mixed-precision compatibility

If no clear difference is found, explicitly state:
- "No functional mismatch found in current smoke tests, but kernel-level optimization is not preserved."

## Implementation checklist

Before final response, verify:
- [ ] file created under `lib/module`
- [ ] module name matches target function
- [ ] upstream source URL(s) recorded in docstring
- [ ] quick parity test included or executed
- [ ] explicit "differences + risks" section prepared

## Output template

Use this response structure:

1. Created file:
- `lib/module/<name>.py`

2. Upstream implementation summary:
- entry point
- actual backend (Python/C++/CUDA)

3. Differences from original:
- bullet list

4. Potential risks:
- bullet list

5. Suggested validation:
- forward parity test
- backward parity test
- edge-case inputs (shape/dtype/device)

## Guardrails

- Do not claim "equivalent" without at least one numerical parity check.
- Do not silently change default arguments from PyTorch.
- Do not remove `eps`-style stability terms.
- If original behavior depends on backend-specific kernels, clearly mark local version as "reference-compatible but not kernel-identical".
