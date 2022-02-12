# MATH-412: Statistical Machine Learning
## Mariia Soroka, Tâm Johan Nguyên, Vireak Prou
This is our project for Statistical Machine Learning course. 
We have implemented different versions of NMF (non-negative matrix factorisation) and 
used this decomposition to analyse audio recordings.
### About the project
Non-negative matrix factorisation (NMF)
alogrithms allow to factorize matrices while enforcing
that the coefficients in the decomposition are non-negative.
In the case of spectogram factorisation this allows for
meaningful separation into sound components. Here we
consider multiplicative gradient and expectation-maximisation
(EM) algorithms for factorisation based on β-divergence
and study how they perform under different choices of
initialisation and β. In particular we show applications of
NMF to pitch estimation and denoising of audio recording
and study how those different methods lead to identification
of different sound features.
### What is in this repository
- Our report NMF.pdf, where we describe all the algorithms and 
analyse results
- Python script NMF.py, where our classes are implemented
- Python notebooks with examples on how to use classes, with benchmarking and with tests on synthetic and real data
- [Input audio data](/data/Chords.wav) and results of denoising using [Euclidian distance NMF](/data/denoised_Chords_EUC_7.wav) and [KL divergence NMF](/data/Chords_KL_denoised.wav) 

