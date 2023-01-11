# Code for numerical experiments in ``On the generalization of learning algorithms that do not converge''

The data plotted in the paper is under `outputs`. See also the `.txt` files for the filenames of the output files shown in the figures.
The main source file that trains neural networks is `zs_train_test.py`. Outputs of the training are processed to compute lower bounds on the stability 
coefficients and the loss auto-correlations in the python scripts under `utils`.

The paper appears in the proceedings of NeurIPS 2022. The bibtex of the arxiv version is here:
    
    @ARTICLE{2022arXiv220807951C,
       author = {{Chandramoorthy}, Nisha and {Loukas}, Andreas and {Gatmiry}, Khashayar and {Jegelka}, Stefanie},
        title = "{On the generalization of learning algorithms that do not converge}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Machine Learning, Mathematics - Dynamical Systems, Mathematics - Optimization and Control, Statistics - Machine Learning},
         year = 2022,
        month = aug,
          eid = {arXiv:2208.07951},
        pages = {arXiv:2208.07951},
     abstract = "{Generalization analyses of deep learning typically assume that the
        training converges to a fixed point. But, recent results
        indicate that in practice, the weights of deep neural networks
        optimized with stochastic gradient descent often oscillate
        indefinitely. To reduce this discrepancy between theory and
        practice, this paper focuses on the generalization of neural
        networks whose training dynamics do not necessarily converge to
        fixed points. Our main contribution is to propose a notion of
        statistical algorithmic stability (SAS) that extends classical
        algorithmic stability to non-convergent algorithms and to study
        its connection to generalization. This ergodic-theoretic
        approach leads to new insights when compared to the traditional
        optimization and learning theory perspectives. We prove that the
        stability of the time-asymptotic behavior of a learning
        algorithm relates to its generalization and empirically
        demonstrate how loss dynamics can provide clues to
        generalization performance. Our findings provide evidence that
        networks that ``train stably generalize better'' even when the
        training continues indefinitely and the weights do not converge.}",
        archivePrefix = {arXiv},
        eprint = {2208.07951},
        primaryClass = {cs.LG},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2022arXiv220807951C},
       adsnote = {Provided by the SAO/NASA Astrophysics Data System}
       }

