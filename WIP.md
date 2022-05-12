## Work-in-progress

While this codebase is born out of a specific research need, I plan to build off it to make it generally useful. This
doc details what's planned and work in progress.

- [ ] Rewrite privacy accounting utils.
- [ ] Merge `_ghost_step` and `_step`; lots of redundant code.
- [x] For non-prompt full fine-tuning on classification tasks, the eval loss first becomes really bad at the beginning
  of training. Performance becomes better all of a sudden.
- [ ] Support mixed-precision for ghost clipping
- [ ] Support additional HF models
    - [ ] BART
    - [ ] T5
- [ ] Release code for dialog generation experiments.
