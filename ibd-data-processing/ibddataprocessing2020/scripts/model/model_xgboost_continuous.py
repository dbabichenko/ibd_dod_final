import pandas as pd
import numpy as np
import optuna
from sklearn import metrics
from sklearn.model_selection import cross_val_score, train_test_split
import xgboost as xgb


class ModelXGBoostContinuous():

    def objective(self, trial):
        x_train, x_test, y_train, y_test = train_test_split(
            self.x,
            self.current_y,
            test_size=0.25,
            random_state=self.random_state
        )

        d_train = xgb.DMatrix(x_train, label=y_train)
        d_test = xgb.DMatrix(x_test, y_test)

        xgb_config = {
            "verbosity": 0,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
            "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
        }

        if xgb_config["booster"] == "gbtree" or xgb_config["booster"] == "dart":
            xgb_config["max_depth"] = trial.suggest_int("max_depth", 1, 9)
            xgb_config["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
            xgb_config["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
            xgb_config["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        if xgb_config["booster"] == "dart":
            xgb_config["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            xgb_config["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            xgb_config["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
            xgb_config["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)

        # Add a callback for pruning
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-rmse")
        bst = xgb.train(xgb_config, d_train, evals=[(d_test, "validation")], callbacks=[pruning_callback])
        preds = bst.predict(d_test)
        pred_labels = np.rint(preds)
        accuracy = metrics.accuracy_score(y_test, pred_labels)

        return accuracy

    def tune_model(self, y):
        self.current_y = y
        optuna.logging.set_verbosity(optuna.logging.FATAL)
        study = optuna.study.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=100)

        return study.best_params

    def train_and_validate(self, x, df_targets, random_state):
        assert isinstance(x, pd.DataFrame)
        assert isinstance(df_targets, pd.DataFrame)

        self.x = x
        self.df_targets = df_targets
        self.random_state = random_state

        model_results = {}
        for target in df_targets.columns:
            best_hypers = self.tune_model(df_targets[target])

            # ==========================================================================
            # split for training/test
            # ==========================================================================
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                df_targets[target],
                test_size=0.25
            )

            # create matrices
            dtrain = xgb.DMatrix(x_train, label=y_train)
            dtest = xgb.DMatrix(x_test, label=y_test)

            model = xgb.XGBRegressor(objective="reg:squarederror", **best_hypers)
            model.fit(x_train, y_train)

            # kfold
            kfolds = 10
            r2s = cross_val_score(model, X=x_test, y=y_test, cv=kfolds, scoring="r2")

            y_preds = model.predict(x_test)

            model_results[target] = {
                'folds': kfolds,
                'avgR2': np.mean(r2s),
                'test_mae': metrics.mean_absolute_error(y_test, y_preds),
                'testRmse': np.sqrt(metrics.mean_squared_error(y_test, y_preds)),
                'testEv': metrics.explained_variance_score(y_test, y_preds)
            }

        return model_results
