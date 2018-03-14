# Bayesian estimation and hypothesis testing for Delayed and Risky Choice (DARC) experiments

## 🔥 This is _in progress_ Python implementation. For stable software see the Matlab  [delay-discounting-analysis](https://github.com/drbenvincent/delay-discounting-analysis) toolbox 🔥

## What is this?

This (will be) a Python based toolbox for the analysis of experimental data from Delayed and Risky Choice (DARC) experiments.

Experimental data could be obtained by any means, but the approach I take in my lab is to run adaptive experiments. And of course, we have a toolbox for that. This is currently implemented in Matlab (see the [darc-experiments-matlab](https://github.com/drbenvincent/darc-experiments-matlab) repo), which accompanies the paper by [Vincent & Rainforth (preprint)](https://psyarxiv.com/yehjb).

### Goals

1. Implement some/all of the ideas discussed in [Vincent (2016)](http://link.springer.com/article/10.3758%2Fs13428-015-0672-2). This was originally implemented in Matlab (see the [delay-discounting-analysis](https://github.com/drbenvincent/delay-discounting-analysis) repo) using JAGS. This implementation will be in Python, using [PyMC3](http://docs.pymc.io/index.html) to do the hard work.
2. Extend beyond the original scope of the paper (delay discounting tasks) to the more general case of Delayed and Risky Choice (DARC) tasks.

## References

Vincent, B. T. (2016) **[Hierarchical Bayesian estimation and hypothesis testing for delay discounting tasks](http://link.springer.com/article/10.3758%2Fs13428-015-0672-2)**, Behavior Research Methods. 48(4), 1608-1620. doi:10.3758/s13428-015-0672-2

Vincent, B. T., & Rainforth, T. (2017, October 20). **[The DARC Toolbox: automated, flexible, and efficient delayed and risky choice experiments using Bayesian adaptive design](https://psyarxiv.com/yehjb)**. Retrieved from [psyarxiv.com/yehjb](https://psyarxiv.com/yehjb)
