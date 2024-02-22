# neural-machine-transalation-eng-vi
A simple Neural Machine Translation model using Transformer architecture to translate from English to Vietnamese

## Model architecture

This is using a simple Transformer Architecture with 6 encoder blocks, 6 decoder blocks. All of which uses Multi-head attention mechanism with scaled-dot product with 8 attention heads.

The model dimension is 512, position-wise feed forwarding dimension is 2048 and some dropout layers with the drop rate of 0.1.

<p align="center">
  <img src="transformer_architecture.png" alt = "UI" title = "Transformer architecture (source: https://arxiv.org/pdf/1706.03762.pdf)" width="270" height="400">
</p>

<p align="center">
   <em>Transformer architecture (source: https://arxiv.org/pdf/1706.03762.pdf)</em>
</p>
