# Third-party attribution

PRISM uses the benchmark and published experimental data distributed with [Microsoft CleaveNet](https://github.com/microsoft/cleavenet), pinned to commit `4dac67defc99ca35d967ddc76eca0fe8b74afdad`. The benchmark originated in Kukreja et al. (2015). Source files and hashes are listed in `data/source_manifest.json`.

CleaveNet-derived code retains its upstream MIT license, included at `third_party/cleavenet/LICENSE`, together with the recorded upstream commit. The generator implementation and sampling workflow build on that code and include PRISM-specific conditioning and training modifications.

Frozen ESM-2 features use the pretrained model distributed at [facebook/esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D). The encoder is downloaded separately and remains subject to its upstream terms. It is not duplicated in this release.
