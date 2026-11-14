# Third-party notices

The repository-level `LICENSE` covers only the original HealthRouter source code.
Datasets, pretrained weights, fine-tuned checkpoints, and imported packages retain
their own terms. None of those artifacts is redistributed in this release.

- **Ultralytics YOLO.** `ultralytics` and Ultralytics-trained YOLO weights are
  offered under AGPL-3.0 by default; proprietary use requires an applicable
  commercial license. Review the [current Ultralytics license terms](https://www.ultralytics.com/license)
  for the intended use before installing or using the segmentation lane.
- **RAD-DINO.** The upstream files carry an MIT license, while the model card says
  the model is for research and not clinical practice. Review the pinned
  [RAD-DINO model card](https://huggingface.co/microsoft/rad-dino/tree/110cbc18d5133582e320b43d53bf5c44e410c936)
  and its use limitations.
- **Benchmark data.** Shenzhen/Montgomery, Kvasir-SEG, CVC-ClinicDB,
  PatchCamelyon, APTOS, and CBIS-DDSM are not covered by this repository's MIT
  license. Obtain each dataset from its upstream host and follow its terms,
  attribution requirements, and any research-only or commercial-use restrictions.

This notice is a practical pointer, not legal advice. Upstream terms can change;
the linked terms control.
