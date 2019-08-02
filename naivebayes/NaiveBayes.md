# Maximum Likelihood Estimation (MLE)

This note is from 周志华‘s book *Maching Learning*.

$D_c$ is the set of cluster $c$ in the training set $C$. Supposed that these samples are i.i.d, then the likelihood of the data set $D_c$ for parameter $\theta_c$ is


$$
P(D_c | \theta_c) = \prod_{\mathbf{x} \in D_c} P(\mathbf{x} | \theta_c). \tag{1}
$$
