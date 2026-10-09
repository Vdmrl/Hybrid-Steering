# Bundled benchmark inputs

This folder includes frozen copies of the IFEval input JSONL, HumanEval
`HumanEval.jsonl.gz`, and the Google Research IFEval evaluator modules. The
evaluator source files carry Apache-2.0 notices in their headers. The
100-prompt screen and 500-prompt confirmation sets are frozen neutral prompts
from the project; their SHA256 checks are enforced by the launcher.

The externally cloned `hybrid-steering-concepts` repository supplies only
training pair text at run time. Its files, model weights, and generated answers
are not committed here. Preserve its per-pair `source_license` fields and
source manifests when sharing derived artifacts.
