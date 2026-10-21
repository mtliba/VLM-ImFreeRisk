# Pre-registration — "Are medical-VLM errors visual?" (written 2026-09-13 before running the new generations)
H1 (uncertainty baseline): a question+initial-answer error predictor (frozen BiomedCLIP embeddings, logistic regression,
   image-grouped 5-fold CV) reaches higher error-detection AUROC than sampling-based self-consistency
   (5 samples, T=0.7, top-p 0.95; signal = fraction of samples agreeing with the greedy answer) on >= 3 of 4 settings.
H2 (shared errors): for two additional VLMs (KEEP prompt only, all splits) the error overlap with MedGemma-4B and
   Qwen2.5-VL-7B on the test sets has phi >= 0.4, and the Q+A predictor trained on one VLM's errors detects the other
   VLMs' errors at AUROC >= 0.75 (SLAKE) / >= 0.70 (VQA-RAD).
H3 (image irrelevance): adding the image embedding changes AUROC by <= 0.02 in every setting.
Decision rule: if H1 and H3 hold and H2 holds for at least one added VLM, write the analysis paper; otherwise report
the failures and do not claim "errors are non-visual".
