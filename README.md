


  # My gradient boosting


<div align="center">
<br />

[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

[![PRs welcome](https://img.shields.io/badge/PRs-welcome-ff69b4.svg?style=flat-square)](https://github.com/gorlov047/MyGradientBoosting/issues)
[![made with hearth by dec0dOS](https://img.shields.io/badge/made%20with%20%E2%99%A5%20by-gorlov047-red)](https://github.com/gorlov047)

</div>

<details open="open">
<summary>Table of Contents</summary>

- [About](#about)
  - [Built With](#built-with)
- [Support](#support)
- [Roadmap](#roadmap)
- [License](#license)
- [Sources of knowledge](#sources-of-knowledge)

</details>

---

## About

<table>
<tr>
<td>

This project is dedicated to the implementation of gradient boosting over decision trees.

Probably, I will add features to this implementation in the future/



<details open>
<summary>Additional info</summary>
<br>

 The Numpy package is used to vectorize calculations and avoid the use of loops.
- Decision tree regressor with the mean squared error, which is equal to variance reduction as feature selection criterion.
- Implementation of categorical feature processing. For a regression problem with the MSE loss function, the values can be
    ordered by the average value of the target on a subset. The split obtained in this way will be optimal.
- Implementation of regularization. trees are easily overfitted and the branching process must be stopped at some point. Implemented stop criteria:  
    - maximum depth of the tree;  
    - minimum number of samples in a leaf;  
    - maximum number of leaves in a tree;  
- Implementation of the search for the best split using dynamic programming. It is necessary and sufficient for each potential split value to know the number of elements in the right and left subtrees, their sum and the sum of their squares (in fact, all this needs to be known only for one of the halves of the split, and for the second it can be obtained by subtracting the values for the first of the full sums). This can be done in one pass through the array, accumulating just the values of partial sums (the operation is vectorized using numpy.cumsum)
- The implementation supports the possible training of boosting over other models.
- The implementation of boosting can optimize any differentiable loss function.
- For each basic model, based on the minimization of the loss function, a coefficient from 0 to 1 is selected with which it will enter the composition
- The implementation supports the separation of data into train and validation to stop the construction of the ensemble when overfitting is observed, i.e. increasing the loss on validation when adding new models.

</details>

</td>
</tr>
</table>

### Built With

- [NumPy](https://github.com/numpy/numpy)

<p align="right"><a href="#My-gradient-boosting">Back to top</a></p>

## Roadmap

A list of proposed features
- Decision tree regressor (Done)
- Gradient boosting regressor (Done)
- Gradient boosting classifier (in progress)
- Tests (in progress)

<p align="right"><a href="#My-gradient-boosting">Back to top</a></p>

## Support

Reach out to the maintainer at one of the following places:

- [GitHub discussions](https://github.com/gorlov047/MyGradientBoosting/discussions)
- The telegram(email) which is located [in GitHub profile](https://github.com/gorlov047)

<p align="right"><a href="#My-gradient-boosting">Back to top</a></p>

## License

This project is licensed under the **MIT license**. Feel free to edit and distribute this template as you like.

See [LICENSE](LICENSE) for more information.

<p align="right"><a href="#My-gradient-boosting">Back to top</a></p>

## Sources of knowledge
Sources that have been used to understand the gradient boosting model  
https://academy.yandex.ru/handbook/ml  
https://github.com/esokolov/ml-course-hse

<p align="right"><a href="#My-gradient-boosting">Back to top</a></p>
