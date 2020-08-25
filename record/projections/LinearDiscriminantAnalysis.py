""" Linear Discriminant Analysis
"""

# Author: René Chenard <rene.chenard.1@ulaval.ca>


import random
import numpy as np
from numpy.random import randn
from numpy.linalg import pinv, eig
from scipy.linalg import orth
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression


class LinearDiscriminantAnalysis(object):
    """Linear Discriminant Analysis

    A basis transformation tool using Linear discriminant analysis (LDA) in
    order to find a more informative representation.

    This tool finds the projection that maximizes the between class scatter to
    within class scatter ratio in order to maximize class separability along
    the first axes.

    If no labels are provided, Gaussian Mixture Model (GMM) is used in order to
    identify clusters that are more likely to belong to different classes.

    If n_components is specified, this tool can be used as a dimensionality
    reduction tool.

    The transformation matrix produced is a full rank matrix and is, therefore,
    invertible. In other words, this is an invertible LDA transformation. If no
    dimensionality reduction was applied to the transformed data, the original
    data should be recoverable by doing an inverse transformation.

    Parameters
    ----------
    n_components : int, default=None
        Number of components (<= n_features) for dimensionality reduction.
        If None, will be set to n_features. This parameter only affects the
        `transform` method.

    n_classes : int, default=None
        Number of classes (< 1) contained in the dataset. If no labels are
        given, unsupervised clustering via GMM will be used with this amount of
        clusters. If None, Akaike Information Criterion (AIC) is used to find
        the optimal amount of clusters within a specified range.

    predict_reduction : bool, default=True
        If True, use linear regression to predict the content of the eliminated
        dimensions when doing inverse transformations. If False, the mean value
        is used.

    eps : float, default=0.01
        Absolute threshold (<= 0) for a singular value of X to be considered
        significant. Eigenvectors whose singular values are non-significant are
        discarded and the rank of the transformation matrix is filled with
        non collinear randomly generated vectors.

    random_state : int, RandomState instance, default=None
        Pass an int for reproducible results across multiple function calls.

    covariance_type : {'full' (default), 'tied', 'diag', 'spherical'}
        String describing the type of covariance parameters to use for the
        unsupervised clustering with Gaussian Mixture Model.
        Must be one of:
        'full'
            each component has its own general covariance matrix
        'tied'
            all components share the same general covariance matrix
        'diag'
            each component has its own diagonal covariance matrix
        'spherical'
            each component has its own single variance

    Attributes
    ----------
    mu_ : array, shape (1, n_features)
        Centroid of the whole dataset. Essentially, the mean of all samples
        along each feature.

    mu_c_ : array, shape (n_classes, n_features)
        Centroid of every classes (or clusters, when unsupervised).

    S_within_ : array, shape (n_features, n_features)
        Within-class scatter matrix.

    S_between_ : array, shape (n_features, n_features)
        Between-class scatter matrix.

    W_ : array, shape (n_features, n_features)
        Transformation matrix.

    W_inverse_ : array, shape (n_features, n_features)
        Inverse transformation matrix.

    eig_pairs_ : tuple list
        Pairs of eigenvalues and eigenvectors from the linear discriminant
        analysis.

    dimensionality_: int
        Minimum amount of dimensions to preserve most of the information. This
        value is determined by using the amount of eigenvalues with a
        significant magnitude (>= eps).

    """

    def __init__(self, n_components=None, n_classes=None,
                 predict_reduction=True, eps=0.001, random_state=None,
                 covariance_type='spherical'):
        # Parameters:
        self.n_components = n_components
        self.n_classes = n_classes
        self.predict_reduction = predict_reduction
        self.eps = eps
        self.random_state = random_state
        self.covariance_type = covariance_type

        # Attributes:
        self.mu_ = None
        self.mu_c_ = None
        self.S_within_ = None
        self.S_between_ = None
        self.W_ = None
        self.W_inverse_ = None
        self.eig_pairs_ = None
        self.predictor_ = None
        self.clusters_ = None
        self.dimensionality_ = None

        # Random seed:
        np.random.seed(random_state)
        random.seed(random_state)

    def fit(self, X, y=None, n_clusters=None, min_clusters=2, max_clusters=40,
            class_clustering=False, verbose=False):
        """Fit LinearDiscriminantAnalysis model according to the given
                   training data and parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), default=None
            Target values. If not provided, GMM clustering is used to identify
            potential classes. The number of clusters that results in the
            smallest AIC is chosen.
        n_clusters : int, default=None
            The specific number of clusters to find.
            Supersedes min_clusters and max_clusters.
        min_clusters : int, default=2
            The minimal number of clusters to consider. If y isn't provided,
            every cluster is treated as a class. If y is provided, this
            corresponds to the minimal amount of clusters inside every classes.
        max_clusters : int, default=40
            The maximal number of clusters to consider. If y isn't provided,
            every cluster is treated as a class. If y is provided, this
            corresponds to the maximal amount of clusters inside every classes.
        class_clustering : bool, default=False
            If True, divides classes into a sum of clusters.
        verbose : bool, default=False
            If True, displays the results.
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]

        assert 0 < min_clusters <= max_clusters < n_samples * n_features
        assert 0 <= self.eps

        if y is None:
            self.clusters_ = self._clustering(X=X, min_clusters=min_clusters,
                                              max_clusters=max_clusters,
                                              n_clusters=n_clusters,
                                              verbose=verbose)

        elif class_clustering:
            y = y.flatten()
            self.clusters_ = - np.ones(y.shape, dtype=int)
            for target in np.unique(y):
                y_ = self._clustering(X=X[y == target], y=target,
                                      min_clusters=min_clusters,
                                      max_clusters=max_clusters,
                                      n_clusters=n_clusters,
                                      verbose=verbose)
                y_ += np.max(self.clusters_) + 1
                self.clusters_[y == target] = y_

            if verbose:
                print("Targets were provided: using the labeled data and "
                      "intra-class clustering.\n")
        else:
            self.clusters_ = y
            if verbose:
                print("Targets were provided: using the labeled data.\n")

        self.mu_, self.mu_c_ = self._means(X=X, y=self.clusters_,
                                           verbose=verbose)

        N_c, self.S_within_ = self._scatter_within(X=X, y=self.clusters_,
                                                   mu_c=self.mu_c_,
                                                   verbose=verbose)

        self.S_between_ = self._scatter_between(mu=self.mu_, mu_c=self.mu_c_,
                                                N_c=N_c, verbose=verbose)

        self.eig_pairs_ = self._eig_pairs(S_within=self.S_within_,
                                          S_between=self.S_between_)

        P = self._filter_eig_pairs(eig_pairs=self.eig_pairs_, eps=self.eps,
                                   verbose=verbose)

        self.W_ = self._fill_rank(P=P, verbose=verbose)

    def transform(self, X):
        """Project data to maximize class separation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        X_new = X @ self.W_

        if self.n_components is not None:
            n_features = X.shape[1]
            assert 0 < self.n_components <= n_features

            if self.predict_reduction:
                self.predictor_ = LinearRegression()
                self.predictor_.fit(X_new[:, :self.n_components], X)
            X_new = X_new[:, :self.n_components]

        return X_new

    def fit_transform(self, X, y=None, n_clusters=None,
                      min_clusters=2, max_clusters=40,
                      class_clustering=True,
                      verbose=False):
        """Fit the model with X (and y if provided) and apply the
        transformation on X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), default=None
            Target values. If not provided, GMM clustering is used to identify
            potential classes. The number of clusters that results in the
            smallest AIC is chosen.
        n_clusters : int, default=None
            The specific number of clusters to find.
            Supersedes min_clusters and max_clusters.
        min_clusters : int, default=2
            The minimal number of clusters to consider if y isn't provided.
        max_clusters : int, default=40
            The maximal number of clusters to consider if y isn't provided.
        class_clustering : bool, default=False
            If True, divides classes into a sum of clusters.
        verbose : bool, default=False
            If True, displays the results.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        self.fit(X=X, y=y, n_clusters=n_clusters,
                 min_clusters=min_clusters,
                 max_clusters=max_clusters,
                 class_clustering=class_clustering,
                 verbose=verbose)
        return self.transform(X)

    def inverse_transform(self, X, predict_reduction=True, verbose=False):
        """Transform data back to its original space.
        In other words, return an input X_original whose transform would be X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            New data, where n_samples is the number of samples
            and n_components is the number of components.
        predict_reduction : bool, default=True
            If True, use linear regression to predict the content of the
            eliminated dimensions when doing inverse transformations. If False,
            the mean value is used.
        verbose : bool, default=False
            If True, displays the results.

        Returns
        -------
        X_original : array-like, shape (n_samples, n_features)
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]

        if self.W_inverse_ is None:
            self.W_inverse_ = pinv(self.W_)

        if n_features != self.W_.shape[1]:
            assert n_features == self.n_components

            if verbose:
                print(f"Reverse tranformation after dimensionality reduction "
                      f"may yield unexpected results: "
                      f"{n_features} dim. expanded to {self.W_.shape[1]} dim.")

            if self.predict_reduction and predict_reduction:
                return self.predictor_.predict(X)
            else:
                mean = self.mu_.reshape(1, -1) @ self.W_
                X_ = np.repeat(mean.reshape(1, -1), n_samples, 0)
                X_[:, :n_features] = X[:, :n_features]

            X = X_

        return X @ self.W_inverse_

    def _clustering(self, X, min_clusters, max_clusters, n_clusters=None,
                    y=None, verbose=False):
        """Find the number of clusters that minimizes the AIC within the
        specified range.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        min_clusters : int
            The minimal number of clusters to consider if y isn't provided.
        max_clusters : int
            The maximal number of clusters to consider if y isn't provided.
        n_clusters : int, default=None
            The specific number of clusters to find.
            Supersedes min_clusters and max_clusters.
        y : array-like of shape (n_samples,), default=None
            Target values.
        verbose : bool, default=False
            If True, displays the results.

        Returns
        -------
        C : ndarray of shape (n_samples,)
            Returns predicted classes.
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]

        if verbose:
            print("No target is provided: using unsupervised clustering.\n")

        if self.n_classes is not None and y is None:
            assert 0 < self.n_classes < n_samples * n_features

            if verbose:
                print(f"Using the provided number of classes "
                      f"({self.n_classes}) for unsupervised clustering.\n")
            model = GaussianMixture(self.n_classes, covariance_type='full',
                                    random_state=self.random_state).fit(X)
        else:
            if verbose:
                if n_clusters is not None:
                    min_clusters = max_clusters = n_clusters
                    print(f"Using the provided number of clusters "
                          f"({n_clusters}) for unsupervised clustering.\n")
                elif y is not None:
                    print(f"Searching for an optimal number of clusters within"
                          f" class {y} for values between {min_clusters} and "
                          f"{max_clusters}...\n")
                else:
                    print("Predicting the classes from the clusters...\n")
                    print(f"Searching for an optimal number of clusters "
                          f"between {min_clusters} and {max_clusters}...\n")

            range_n = np.arange(min_clusters, max_clusters + 1)
            models = []
            for n in range_n:
                gmm = GaussianMixture(n, covariance_type=self.covariance_type,
                                      random_state=self.random_state).fit(X)
                models.append(gmm)
                if verbose:
                    print(f"{n} clusters: "
                          f"\tAIC: {gmm.aic(X):.3f}, "
                          f"\tBIC: {gmm.bic(X):.3f}  ")
            index = np.argmin(np.array([m.aic(X) for m in models]))
            model = models[index]
            self.n_classes = index + min_clusters
            if verbose and n_clusters is None:
                a = f" for class {y}" if y is not None else ""
                print(f"Optimal number of clusters{a} found: "
                      f"{self.n_classes}\n")

        return model.predict(X)

    @staticmethod
    def _means(X, y, verbose=False):
        """Evaluate the mean (centroid) of the whole dataset and the means of
        each classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        verbose : bool, default=False
            If True, displays the results.
        
        Returns
        -------
        mu, mu_c : pair of ndarrays of shape (n_features, 1) and (n_classes, n_features)
            Returns the evaluated means.
        """
        n_features = X.shape[1]
        n_classes = len(np.unique(y))

        mu = np.mean(X, axis=0).reshape(1, -1)
        if verbose:
            print(f"Mu:\n{mu}\n")

        mu_c = np.zeros((n_classes, n_features))
        for i, target in enumerate(np.unique(y)):
            mu_c[i] = np.mean(X[y == target], axis=0)
            if verbose:
                print(f"Mu_c[{i}]:\n{mu_c[i]}\n")

        return mu, mu_c

    @staticmethod
    def _scatter_within(X, y, mu_c, verbose=False):
        """Evaluate the scatter matrix of each classes (within classes).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        mu_c : array-like of shape (n_classes, n_features)
            Mean (centroid) of each classes.
        verbose : bool, default=False
            If True, displays the results.

        Returns
        -------
        N_c, S_within : pair of ndarrays of shape (n_classes, ) and (n_features, n_features)
            Returns the size of each classes and the corresponding scatter
            matrix (within classes).
        """
        n_classes = len(np.unique(y))

        data = []
        N_c = np.zeros(n_classes)
        for i, target in enumerate(np.unique(y)):
            delta = X[y == target] - mu_c[i]
            data.append(delta.T @ delta)
            N_c[i] = np.sum(y == target)

        S_within = np.sum(data, axis=0)
        if verbose:
            print(f"S_intra:\n{S_within}\n")

        return N_c, S_within

    @staticmethod
    def _scatter_between(mu, mu_c, N_c, verbose=False):
        """Evaluate the scatter matrix of the whole dataset (between classes).

        Parameters
        ----------
        mu : array-like of shape (1, n_features)
            Mean (centroid) of the whole dataset.
        mu_c : array-like of shape (n_classes, n_features)
            Mean (centroid) of each classes.
        N_c : array-like of shape (n_classes, )
            Size of each classes.
        verbose : bool, default=False
            If True, displays the results.

        Returns
        -------
        S_between : ndarray of shape (n_features, n_features)
            Returns the scatter matrix of the whole dataset (between classes).
        """
        delta = np.array(mu_c - mu)
        S_between = N_c * delta.T @ delta
        if verbose:
            print(f"S_inter:\n{S_between}\n")

        return S_between

    @staticmethod
    def _eig_pairs(S_within, S_between):
        """Evaluates the eigenvalues and the corresponding eigenvectors of the
        scatter matrices ratio.

        Parameters
        ----------
        S_within : ndarray of shape (n_features, n_features)
            Sum of the scatter matrix of each classes (within classes)
        S_between : ndarray of shape (n_features, n_features)
            Scatter matrix of the whole dataset (between classes).

        Returns
        -------
        eig_pairs : list of pairs
            A sorted list of the eigenvalues (in decreasing order by value) and
            their corresponding eigenvector.
        """
        A = pinv(S_within) @ S_between
        eig_val, eig_vec = eig(A)
        eig_val = np.abs(eig_val)

        return sorted(zip(eig_val, eig_vec.T), key=lambda k: k[0],
                      reverse=True)

    def _filter_eig_pairs(self, eig_pairs, eps, verbose=False):
        """Filters the (eigenvalue, eigenvector) pairs by their explained
        variance.

        Parameters
        ----------
        eig_pairs : list of pairs
        eps : float
        verbose : bool, default=False
            If True, displays the results.

        Returns
        -------
        P : ndarray of shape (?, n_features)
        """
        eig_vals, eig_vecs = zip(*eig_pairs)
        total = sum(eig_vals)
        eigenvectors = []

        if verbose:
            print("Singular values:")
        for i, v in enumerate(eig_pairs):
            if v[0] / total >= eps / 100:
                verdict = 'Accepted'
                eigenvectors.append(v[1])
            else:
                verdict = 'Rejected'
            if verbose:
                percentage = v[0] / total
                eigenvalue = v[0]
                print(f"Singular value {i + 1:}: "
                      f"\t{percentage:<8.2%} "
                      f"\t{eigenvalue:<8.6f} \t {verdict}")
        if verbose:
            print()

        self.dimensionality_ = len(eigenvectors)

        return np.vstack(eigenvectors).T.real

    @staticmethod
    def _fill_rank(P, verbose=False):
        """Fills the rank of the transformation matrix with linearly
        independent vectors in order to make the transformation invertible.

        Parameters
        ----------
        P : ndarray of shape (?, n_features)
        verbose : bool, default=False
            If True, displays the results.

        Returns
        -------
        W : ndarray of shape (n_features, n_features)
            The transformation matrix.
        """
        n_features = P.shape[0]

        while True:
            W = randn(n_features, n_features)
            W = orth(W)
            W[:P.shape[0], :P.shape[1]] = P
            if orth(W).shape == (n_features, n_features):
                break

        if verbose:
            print(f"W:\n{W}\n")

        return W
