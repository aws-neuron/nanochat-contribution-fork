"""NKI kernel code for the Trainium backend (no torch autograd, no nanochat model knowledge).

  attention_bthd.py  flash attention forward / backward on the native (B, T, H, D) layout, causal or sliding window

The autograd wrapper and the dispatch rules are in nanochat/trn/attention.py.
"""
