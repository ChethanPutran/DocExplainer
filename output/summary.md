# State-of-the-Art   Generative   AI   Architectures   and Neural   Operators   (2025)

Abstract

Generative   AI   (GenAI)   is   advancing   rapidly,   driven   by   Transformers,   Diffusion   models, and   Neural   Operators.   This   report   surveys   state-of-the-art   generative   architectures   as   of late   2025,   explaining   their   working   principles,   strengths,   and   applications   in   scientific   and engineering   domains.

## 1 Overview   of   Generative   AI

Generative   AI   creates   new   content   such   as   text,   images,   code,   and   scientific   data   by   learning patterns from large-scale datasets.   These models are widely used in creative industries, software development,   and   scientific   research.

## 2 Core   Generative   Model   Families

Transformer-Based   Models

Transformers   dominate   text   and   code   generation.   They   rely   on   self-attention   mechanisms   to model   long-range   dependencies   in   sequences.   Transformers   are   highly   scalable   but   computa- tionally   expensive.   Representative   models   include   GPT-5 ,   Gemini-3 ,   and   Claude-4.5 .

Diffusion   Models

Diffusion   models   achieve   state-of-the-art   performance   in   image   and   video   generation. They gradually   denoise   random   noise   into   structured   data.   While   producing   high-quality   outputs, inference   can   be   slow.   Examples   include   Stable   Diffusion ,   Imagen , DALL-E   2   and   Mid- journey .

Generative   Adversarial   Networks

GANs consist of a generator and discriminator trained adversarially.   They are fast and suitable for   real-time   applications,   though   they   are   less   stable   than   diffusion   models.

Variational   Autoencoders

VAEs   encode   data   into   a   structured   latent   space   and   reconstruct   it   probabilistically.   They   are useful   for   representation   learning   but   often   produce   blurrier   outputs.

## 3 Neural   Operators

Traditional   neural   networks   learn   mappings   between   finite-dimensional   vectors.   In   contrast, many   scientific   problems   require   learning   mappings   between   functions,   known   as   operators . Neural   operators   enable   direct   learning   of   these   function-to-function   mappings.

1

DeepONet   (Deep   Operator   Network)

DeepONet was the first architecture to theoretically prove that neural networks can approximate any   continuous   nonlinear   operator.   It   consists   of   two   components:

•   Branch   Network :   Encodes   the   input   function   (e.g.,   initial   or   boundary   conditions).

•   Trunk   Network :   Encodes   the   spatial   coordinates   where   the   output   is   evaluated.

By   combining   these   representations,   DeepONet   can   approximate   solutions   to   differential   equa- tions   across   different   domains.

Fourier   Neural   Operator   (FNO)

The   Fourier   Neural   Operator   is   designed   for   efficiency   and   scalability.   It   operates   in   the   fre- quency   domain   using   Fourier   transforms.

•   Resolution   Invariance :   Models   trained   on   coarse   grids   generalize   to   finer   grids.

•   Efficiency :   Uses   Fast   Fourier   Transforms   (FFT),   making   it   significantly   faster   than DeepONet   for   PDE   problems.

## 4 Key   Comparisons

Table   1:   Comparison   of   generative   and   operator   learning   models

Feature Transformers Diffusion DeepONet FNO

Best   Use Text   /   Code Images   /   Video Physics   Theory Fluid   Dynamics Speed Medium Slow Medium Very   Fast Mathematical   Basis Attention Stochastic Universal   Approximation Fourier   Space Input   Type Sequences Pixels Functions Grids   /   Meshes

## 5 Trends   and   Challenges

Current   research   trends   include:

•   Physics-aware   and   constraint-based   generative   models

•   Improved   computational   efficiency   and   energy   usage

•   Increased   reliability   and   reduced   hallucinations   in   scientific   outputs

## 6 Conclusion

Transformers   and   diffusion   models   dominate   creative   and   multimodal   AI   tasks,   while   neural operators   enable   a   new   paradigm   for   scientific   computing.   DeepONet   provides   strong   theoret- ical   guarantees   for   operator   learning,   and   FNO   delivers   scalability   and   resolution   invariance. Together,   they   form   the   foundation   of   modern   scientific   machine   learning.

2

## References

[1]   Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, and A. Anandku- mar.   Fourier   Neural   Operator   for   Parametric   Partial   Differential   Equations.   arXiv   preprint arXiv:2010.08895 ,   2020.   Available   at:   https://arxiv.org/abs/2010.08895

[2]   L.   Lu,   P.   Jin,   G.   Pang,   Z.   Zhang,   and   G.   E.   Karniadakis.   DeepONet:   Learning   nonlinear operators for identifying differential equations based on the universal approximation theorem of   operators.   arXiv   preprint   arXiv:1910.03193 ,   2019.   Available   at:   https://arxiv.org/ abs/1910.03193

3

