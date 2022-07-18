## Image classification with vision Transformers

### Notes

`main.py` contains simple training code that enables either full fine-tuning (up to isolated embedding parameters) or
linear probing on CIFAR-10. For image classification, linear probing tends to perform generally better than full
fine-tuning. This is confirmed in my personal experiments and also recent works (e.g., [[1]](https://arxiv.org/pdf/2204.13650.pdf) [[2]](https://arxiv.org/pdf/2205.02973.pdf)).

[1] De, Soham, et al. "Unlocking high-accuracy differentially private image classification through scale." arXiv preprint arXiv:2204.13650 (2022).

[2] Mehta, Harsh, et al. "Large scale transfer learning for differentially private image classification." arXiv preprint arXiv:2205.02973 (2022).
