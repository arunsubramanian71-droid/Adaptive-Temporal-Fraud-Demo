# Note on the tau lock hashes (2026-10-12)

`scripts/run_phase9_e0_tau.py` wrote each tau JSON in text mode (Windows stores CRLF line endings) but hashed the LF-normalized string. The initially recorded hashes therefore did not match the file bytes. Verified before any change: the SHA-256 of each file with CRLF normalized to LF equals the initially recorded hash, so the content is exactly what was hashed at creation. Tau values were NOT recomputed. The `.sha256` files were then rewritten to the SHA-256 of the actual file bytes, which is what the E0 driver verifies.

* seed 42: initially recorded (LF-normalized) e9a5b9ee11fb5e2e8ea12959ed44736230056679fa106118663f27fc0ced0ffc; file-bytes hash now 3f73516c232363089d0492d0e99ee854b0401220a656ee3ef566f7d9e8ed6541
* seed 123: initially recorded (LF-normalized) b9722865504e239b532a5c3373ac73a751da409791a279b1c4781420509a45e6; file-bytes hash now f1961b990925e02cd0c13a9f2ed35c077c50d6e39fbd33f46999bc4d915364e3
* seed 2024: initially recorded (LF-normalized) 6d8e25a2b3ef2bd3ae7fc0cf47200a552ffc1ced9f70c6e6a9e24b55c292ed4f; file-bytes hash now 477d1b5243058a7431c107343f7da2064307b9fd79ddcdc7e5c80c1dd457b732
