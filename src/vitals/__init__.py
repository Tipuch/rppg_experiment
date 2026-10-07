"""Respiratory rate from a 30 s PPG: the RR Mamba pipeline of run AH, frozen.

waves      each segment's wave (3 z-scored 10 s windows), and its `tach` and `ls`
           pulse-interval channels, the network's inputs
contact    BIDMC, PPG-DaLiA and CapnoBase recordings, and their breath labels
rr_table   the training table: all sources, shifted windows, a 90/5/5 subject
           split, train labels balanced (`python -m src.vitals.rr_table`)
rr_mamba   the network and its training (`cli vitals-rr-mamba`)
plot       its train-versus-dev figure
"""
