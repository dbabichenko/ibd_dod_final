import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb


class ModelXGBoostMulticlass():
    def train_and_validate(self, x, y, random_seed):
        assert isinstance(x, pd.DataFrame)
        assert isinstance(y, pd.DataFrame)

        results = {}

        for target in y.columns:
            # split for training/test
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y[target],
                test_size=0.25,
                random_state=random_seed
            )

            # train
            model = xgb.XGBClassifier()
            model.fit(x_train, y_train)

            # evaluate
            y_preds = model.predict(x_test)

            # report
            results[target] = {
                'accuracy': metrics.accuracy_score(y_test, y_preds),
                'f1': metrics.f1_score(y_test, y_preds, average='weighted'),
                'roc_auc': metrics.roc_auc_score(y_test, y_preds),
                'feature_importances': model.feature_importances_,
                'features': x.columns
            }

        return results
