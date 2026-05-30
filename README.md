# Compressed AI Is Not Automatically Greener — Figure Code

This repository contains the plotting code for Fig. 1 of the manuscript **“Compressed AI is not automatically greener.”** The figure summarizes deployment measurements showing that nominally low-bit inference can reduce memory footprint while increasing energy per completed query, latency, or reasoning error under specific workloads.

The script is for **figure reproduction**, not for re-running model inference or benchmarking from scratch.

## Files

- `sciencePolicyCode.py` — generates the six-panel quantization-trap figure from recorded measurement values.

## What the figure shows

Fig. 1 contains six panels:

- **A** — schematic of the autoregressive dequantization loop and the recurring deployment tax, `E_tax`.
- **B** — Qwen-2.5 energy per query under 16-bit and 4-bit deployment.
- **C** — Qwen-2.5 reasoning accuracy under 16-bit and 4-bit deployment.
- **D** — illustrative decomposition of total per-query energy into `E_mem`, `E_compute`, and `E_tax`.
- **E** — MedGo-32B medical-reasoning energy across 16-bit, AWQ-4bit, NF4-4bit, and GPTQ-Marlin where available.
- **F** — corresponding medical-reasoning accuracy.

The main empirical contrast highlighted in the figure is the PubMedQA result: production-grade methods labeled `w4-g128 4-bit` produce a **14.6× spread** in per-query energy on the same model/hardware setting, from 2832 J for AWQ-4bit to 194 J for GPTQ-Marlin-4bit, compared with 565 J for the 16-bit baseline.


## Citation

This code accompanies:

Han, H., Liu, X., Wang, X., Han, F., and Ling, Q. (2026). *Compressed AI is not automatically greener*. Manuscript.

BibTeX:

```bibtex
@misc{han2026compressedai,
  title  = {{Compressed AI is not automatically greener}},
  author = {Han, Henry and Liu, Xiyang and Wang, Xiaodong and Han, Fei and Ling, Qinghua},
  year   = {2026},
  note   = {Manuscript}
}
