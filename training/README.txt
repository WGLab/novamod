data: IVT
feature: 11
model: CNN-Transformer

online-
test0: DNA, trained on HG002-WGA (random subset), small model (64,64), static beta = 1e-3
test1: RNA, trained on IVT, large model (256,128), static beta = 1e-4
test2: RNA, trained on IVT, large model (256,128), warmup beta = 1e-3 in 10 epochs
test3: DNA, trained on HG002-WGA, large model (256,128), warmup beta = 1e-3 in 10 epochs
test4: DNA, trained on unmodified DNA oligos, large model (256,128), warmup beta = 1e-3 in 10 epochs
test5: RNA, trained on unmodified RNA oligos, large model (256,128), warmup beta = 1e-3 in 10 epochs
test6: DNA, trained on unmodified DNA oligos, large model (256,128), warmup beta = 1e-3 in 10 epochs, increased sampling rate
test7: RNA, trained on unmodified RNA oligos, large model (256,128), warmup beta = 1e-3 in 10 epochs, increased sampling rate
test8: DNA, trained on unmodified DNA oligos, large model (256,128), warmup beta = 1e-4 in 10 epochs, increased sampling rate
test9: DNA, trained on unmodified DNA oligos (shuffled), large model (256,128), warmup beta = 1e-4 in 10 epochs

static-
test0: DNA, oligos, static, small model (64,64), static beta = 1e-3
test1: RNA, oligos, static, small model (64,64), static beta = 1e-3
