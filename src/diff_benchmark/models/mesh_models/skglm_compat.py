from __future__ import annotations

from skglm import GeneralizedLinearEstimator

try:
    from sklearn.utils.validation import validate_data
except ImportError:  # sklearn<1.6 compatibility
    validate_data = None

from sklearn.utils.validation import check_array, check_X_y


class CompatGeneralizedLinearEstimator(GeneralizedLinearEstimator):
    """Compatibility shim for skglm on newer scikit-learn versions.

    scikit-learn >= 1.7 removed ``BaseEstimator._validate_data`` in favor of
    ``sklearn.utils.validation.validate_data``. Older versions of skglm still
    call ``self._validate_data`` inside ``fit``.

    This subclass restores that method so skglm estimators remain usable inside
    sklearn meta-estimators such as ``GridSearchCV``.
    """

    def _validate_data(self, X, y=None, reset=True, **check_params):
        if validate_data is not None:
            y_arg = y if y is not None else "no_validation"
            return validate_data(
                self,
                X=X,
                y=y_arg,
                reset=reset,
                **check_params,
            )

        if y is None:
            return check_array(X, **check_params)
        return check_X_y(X, y, **check_params)