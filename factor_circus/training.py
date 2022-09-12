import datetime
import logging
import os
import sys
from collections import defaultdict
from typing import Optional, List, Union, Callable, Dict

import numpy as np
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay, CustomBusinessMonthEnd
from scipy import stats
from sklearn.metrics import mean_squared_error, ndcg_score

from data_management.keeper.ZooKeeper import ZooKeeper
from data_management.pandas_utils.cache import panel_df_join
from model_zoo.BaseModel import BaseModel


def _is_classification_problem(model: BaseModel, label_name: str):
    if 'group' in label_name:
        regression = False
    else:
        regression = True
    if model.is_classifier() and not regression:
        return True
    elif not model.is_classifier() and regression:
        if label_name == 'label':
            print('Default label name. Treat it as Regression problem')
        return False
    else:
        raise ValueError('Label and model is not much. Model is {}, but label indicates {}'
                         .format(model.estimator_type, 'regression' if regression else 'classification'))


def linear_decay_weights(days):
    first_day = days[0]
    last_day = days[-1]
    half_life_days = (last_day - first_day).days // 2
    diff = (last_day - days).days.values
    weights = 2 ** (-diff / half_life_days)
    return weights


def get_sample_weights(sample_weights_policy: Optional[Union[str, Callable]], X: pd.DataFrame):
    if isinstance(sample_weights_policy, str):
        if sample_weights_policy == 'linear_decay':
            train_days = X.index.get_level_values(0)
            weights = linear_decay_weights(train_days)
            return weights
        else:
            raise NotImplementedError
    elif callable(sample_weights_policy):
        raise NotImplementedError
    else:
        raise NotImplementedError


def get_voting_weights(predict_ensemble_weights_policy: Optional[Union[str, Callable]],
                       models: Dict
                       ):
    if isinstance(predict_ensemble_weights_policy, str):
        if predict_ensemble_weights_policy == 'linear_decay':
            train_days = pd.Index(models.keys())
            weights = linear_decay_weights(train_days)
            return weights
        else:
            raise NotImplementedError
    elif callable(predict_ensemble_weights_policy):
        raise NotImplementedError
    else:
        raise NotImplementedError


def get_regression_voting_predict(X, models: Dict[pd.Timestamp, BaseModel], weights: np.ndarray):
    """

    Parameters
    ----------
    X
    models
    weights

    Returns
    -------

    """
    weights = weights / weights.sum()
    pred = pd.Series(0, index=X.index)
    for model, weight in zip(models.values(), weights):
        y_pred = model.predict(X) * weight
        pred += y_pred
    return pred


def get_classification_voting_predict(X, classes,
                                      models: Dict[pd.Timestamp, BaseModel],
                                      weights: np.ndarray,
                                      ):
    """

    Parameters
    ----------
    X
    models
    weights

    Returns
    -------

    """
    prob = get_voting_proba(X, classes, models, weights)
    idx = np.argmax(prob.values, axis=1)
    res = [prob.columns[i] for i in idx]
    return pd.Series(res, index=X.index)


def get_voting_proba(X, classes, models: Dict[pd.Timestamp, BaseModel],
                     weights: np.ndarray):
    """

    Parameters
    ----------
    X
    models
    weights

    Returns
    -------

    """
    weights = weights / weights.sum()
    pred_prob = pd.DataFrame(0, index=X.index, columns=classes)
    for model, weight in zip(models.values(), weights):
        y_prob = model.predict_proba(X) * weight
        pred_prob += y_prob

    return pred_prob


def cross_sectional_regression_trainer(factor_data: pd.DataFrame,
                                       label: pd.Series,
                                       model: BaseModel,
                                       embargo_offset,
                                       synthetic_factor_name,
                                       *,
                                       label_end: Optional[pd.Series] = None,
                                       rolling_window_size: Optional[CustomBusinessDay] = None,
                                       sample_weights_policy: Optional[Union[str, Callable]] = None,
                                       retrain_policy: Optional[
                                           Union[List, CustomBusinessDay, CustomBusinessMonthEnd, str, Callable]
                                       ] = None,
                                       predict_ensemble: [Optional[int]] = None,
                                       predict_ensemble_weights_policy: Optional[Union[str, Callable]] = None,
                                       factor_names: Optional[List[str]] = None,
                                       model_metrics=('IC', 'MSE'),
                                       keeper: Optional[ZooKeeper] = None,
                                       save_model_path: Optional[str] = None,
                                       **kwargs
                                       ):
    os.makedirs('../logs/training', exist_ok=True)
    log_filename = datetime.datetime.now().strftime(
        '../logs/training/' + synthetic_factor_name + "_%Y-%m-%d_%H_%M_%S.log")
    file_handler = logging.FileHandler(filename=log_filename)
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.INFO,
                        format='|%(levelname)s|%(asctime)s|%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=handlers)
    logger = logging.getLogger(synthetic_factor_name)

    factor_data.index.names = ['date', 'code']
    label_name = label.name
    if not label_name:
        label_name = 'label'
    # is_classification = _is_classification_problem(model, label_name)
    # if is_classification:
    #     classes = label.value_counts().sort_index().index
    # else:
    #     classes = None
    if factor_names:
        cols = factor_names + [label_name]
        factor_data = factor_data[factor_names]
    else:
        factor_names = factor_data.columns
        cols = factor_data.columns.tolist() + [label_name]
    label = label.to_frame()
    if label_end is not None:
        label = panel_df_join(label, label_end.to_frame('label_end'))
        # label = label.fillna()
    data = panel_df_join(factor_data, label)
    data = data.dropna()
    models = {}
    metrics = defaultdict(dict)
    features_importance = defaultdict(dict)
    synthetic_factor = []

    factor_dates = data.index.get_level_values(0).drop_duplicates().sort_values()
    last_training_date = factor_dates[0]
    first_date = factor_dates[0]

    for trade_date in factor_dates:
        logger.info('*' * 40)
        logger.info('rebalance: %s', trade_date)
        retrain = False
        # available factor data

        # available label data
        available_label_date = trade_date - embargo_offset

        if isinstance(retrain_policy, List):
            if trade_date in retrain_policy:
                retrain = True
                last_training_date = trade_date
        elif isinstance(retrain_policy, CustomBusinessDay):
            raise NotImplementedError
        elif isinstance(retrain_policy, CustomBusinessMonthEnd):
            raise NotImplementedError
        elif callable(retrain_policy):
            raise NotImplementedError

        if retrain:
            if keeper is not None and len(synthetic_factor) > 0:
                factor_values = pd.concat(synthetic_factor)
                factor_values.name = synthetic_factor_name
                # kwargs['features_importance'] = features_importance
                # kwargs['metrics'] = metrics
                keeper.save_factor_value('synthetic_factor', factor_values, **kwargs)

            logger.info('retrain: %s', trade_date)
            if rolling_window_size:
                sample_start = available_label_date - rolling_window_size
                if first_date > sample_start:
                    logger.info('no enough data for training, continue')
                    continue

            else:
                sample_start = factor_dates[0]

            available_data = data.loc[sample_start: available_label_date]
            if label_end is not None:
                available_data = available_data[available_data['label_end'] <= trade_date]
            available_data = available_data.dropna()
            if len(available_data) == 0:
                logger.info('No data pass')
                continue

            X = available_data[factor_names]
            y = available_data[label_name]
            dts = X.index.get_level_values(0).drop_duplicates().sort_values()
            logger.info("Train model on data from {} to {}. Total sample size: {}".format(dts[0], dts[-1], len(X)))
            if sample_weights_policy and rolling_window_size:
                sample_weight = get_sample_weights(sample_weights_policy, X)
                model.fit(X, y, sample_weight=sample_weight)
            else:
                model.fit(X, y)
            models[trade_date] = model
            features_importance[trade_date] = model.get_feature_importance()

            if save_model_path is not None:
                model_name = trade_date.strftime('%Y-%m-%d')
                model.save_model(save_model_path, model_name)

        if len(models) == 0:
            continue

        # predict trade_date data
        test_X = data.loc[trade_date:trade_date][factor_names]
        y_true = data.loc[trade_date:trade_date][label_name]
        if predict_ensemble:
            if len(models.keys()) < predict_ensemble:
                continue
            model_keys = list(models.keys())[-predict_ensemble:]
            models_used = {k: v for k, v in models.items() if k in model_keys}
            weights = get_voting_weights(predict_ensemble_weights_policy, models_used)
            y_predict = get_regression_voting_predict(test_X, models_used, weights)
        else:
            y_predict = model.predict(test_X)

        synthetic_factor.append(y_predict)
        if keeper is not None and len(synthetic_factor) > 0:
            factor_values = pd.concat(synthetic_factor)
            factor_values.name = synthetic_factor_name
            # kwargs['features_importance'] = features_importance
            # kwargs['metrics'] = metrics
            keeper.save_factor_value('synthetic_factor', factor_values, **kwargs)

        try:
            if 'IC' in model_metrics:
                ic = stats.spearmanr(y_true, y_predict, nan_policy='omit')[0]
                metrics[trade_date]["IC"] = ic

            if 'IC_ranking' in model_metrics:
                pred = np.digitize(y_predict, np.histogram(y_predict, int(y_true.max()) - 1)[1])
                ic = stats.spearmanr(y_true, pred, nan_policy='omit')[0]
                metrics[trade_date]["IC_ranking"] = ic

            if 'MSE' in model_metrics:
                mse = mean_squared_error(y_true.fillna(y_true.median()), y_predict)
                metrics[trade_date]["MSE"] = mse

            if "NDCG" in model_metrics:
                score = ndcg_score([y_true], [y_predict], k=150)
                metrics[trade_date]["NDCG@150"] = score
                score = ndcg_score([y_true], [y_predict], k=100)
                metrics[trade_date]["NDCG@100"] = score
                score = ndcg_score([y_true], [y_predict], k=50)
                metrics[trade_date]["NDCG@50"] = score
            logger.info(metrics[trade_date])
        except:
            pass

    metrics = pd.DataFrame(metrics).T
    features_importance = pd.DataFrame(features_importance).T
    synthetic_factor = pd.concat(synthetic_factor)
    synthetic_factor.name = synthetic_factor_name
    return synthetic_factor, metrics, features_importance, models


def cross_sectional_classification_trainer(factor_data: pd.DataFrame,
                                           label: pd.Series,
                                           model: BaseModel,
                                           embargo_offset, *,
                                           label_end: Optional[pd.Series] = None,
                                           rolling_window_size: Optional[CustomBusinessDay] = None,
                                           sample_weights_policy: Optional[Union[str, Callable]] = None,
                                           retrain_policy: Optional[
                                               Union[List, CustomBusinessDay, CustomBusinessMonthEnd, str, Callable]
                                           ] = None,
                                           predict_ensemble: [Optional[int]] = None,
                                           predict_ensemble_weights_policy: Optional[Union[str, Callable]] = None,
                                           factor_names: Optional[List[str]] = None,
                                           model_metrics=('IC',),
                                           ):
    factor_data.index.names = ['date', 'code']
    label_name = label.name
    if not label_name:
        label_name = 'label'
    is_classification = _is_classification_problem(model, label_name)
    if is_classification:
        classes = label.value_counts().sort_index().index
    else:
        raise ValueError('This is only for classification. Please use cross_sectional_regression_trainer instead')
    if factor_names:
        cols = factor_names + [label_name]
        factor_data = factor_data[factor_names]
    else:
        factor_names = factor_data.columns
        cols = factor_data.columns.tolist() + [label_name]

    if label_end is not None:
        label = label.to_frame().join(label_end.to_frame('label_end'))
    data = factor_data.join(label)

    models = {}
    metrics = defaultdict(dict)
    features_importance = defaultdict(dict)
    synthetic_factor = []

    factor_dates = data.index.get_level_values(0).drop_duplicates().sort_values()
    last_training_date = factor_dates[0]
    first_date = factor_dates[0]

    for trade_date in factor_dates:
        print('*' * 20)
        print('rebalance:', trade_date)
        retrain = False
        # available factor data

        # available label data
        available_label_date = trade_date - embargo_offset

        if isinstance(retrain_policy, List):
            if trade_date in retrain_policy:
                retrain = True
                last_training_date = trade_date
        elif isinstance(retrain_policy, CustomBusinessDay):
            raise NotImplementedError
        elif isinstance(retrain_policy, CustomBusinessMonthEnd):
            raise NotImplementedError
        elif callable(retrain_policy):
            raise NotImplementedError

        if retrain:
            print('retrain:', trade_date)
            if rolling_window_size:
                sample_start = available_label_date - rolling_window_size
                if first_date > sample_start:
                    print('no enough data for training, continue')
                    continue

            else:
                sample_start = available_label_date - pd.Timedelta(seconds=1)

            available_data = data.loc[sample_start: available_label_date]
            if label_end is not None:
                available_data = available_data[available_data['label_end'] <= trade_date]
            available_data = available_data.dropna()
            if len(available_data) == 0:
                print('No data pass')
                continue

            X = available_data[factor_names]
            y = available_data[label_name]
            dts = X.index.get_level_values(0).drop_duplicates().sort_values()
            print("Train model on data from {} to {}. Total sample size: {}".format(dts[0], dts[-1], len(X)))
            if sample_weights_policy and rolling_window_size:
                sample_weight = get_sample_weights(sample_weights_policy, X)
                model.fit(X, y, sample_weight=sample_weight)
            else:
                model.fit(X, y)
            models[trade_date] = model
            features_importance[trade_date] = model.get_feature_importance()

        if len(models) == 0:
            continue

        # predict trade_date data
        test_X = data.loc[trade_date:trade_date][factor_names]
        y_true = data.loc[trade_date:trade_date][label_name]
        if predict_ensemble:
            if len(models.keys()) < predict_ensemble:
                continue
            model_keys = list(models.keys())[-predict_ensemble:]
            models_used = {k: v for k, v in models.items() if k in model_keys}
            weights = get_voting_weights(predict_ensemble_weights_policy, models_used)
            y_predict = get_classification_voting_predict(test_X, classes, models_used, weights)
            y_prob = get_voting_proba(test_X, classes, models_used, weights)
        else:
            y_predict = model.predict(test_X)
            y_prob = model.predict_proba(test_X)

        # mid = classes.to_series().median()

        # pred_prob_df = y_predict.to_frame('predict').join(y_prob.to_frame('prob'))
        # y_factor = pred_prob_df.apply(lambda x: x['predict'] - 1 + x['prob'] if x['predict'] > mid else
        #                               x['predict'] + 1 - x['prob'] if x['predict'] < mid else
        #                               x['predict']
        #                               ,axis=1)
        y_factor = (y_prob * classes.values).sum(axis=1)

        synthetic_factor.append(y_factor)

        if 'IC' in model_metrics:
            ic = stats.spearmanr(y_true, y_factor, nan_policy='omit')[0]
            metrics[trade_date]["IC"] = ic

        # if 'MSE' in model_metrics and not is_classification:
        #     mse = mean_squared_error(y_true.fillna(y_true.median()), y_predict)
        #     metrics[trade_date]["MSE"] = mse
        # todo metric
        print(metrics[trade_date])

    metrics = pd.DataFrame(metrics).T
    features_importance = pd.DataFrame(features_importance, index=factor_names).T
    synthetic_factor = pd.concat(synthetic_factor)
    return synthetic_factor, metrics, features_importance, models


def time_series_regression_trainer(factor_data: pd.DataFrame,
                                   label: pd.Series,
                                   model: BaseModel,
                                   synthetic_factor_name,
                                   rolling_window_size: int,
                                   *,
                                   label_end: Optional[pd.Series] = None,
                                   sample_weights_policy: Optional[Union[str, Callable]] = None,
                                   retrain_policy: Optional[
                                       Union[List, CustomBusinessDay, CustomBusinessMonthEnd, str, Callable]
                                   ] = None,
                                   predict_ensemble: [Optional[int]] = None,
                                   predict_ensemble_weights_policy: Optional[Union[str, Callable]] = None,
                                   factor_names: Optional[List[str]] = None,
                                   model_metrics=('MSE', 'RET'),
                                   keeper: Optional[ZooKeeper] = None,
                                   save_model_path: Optional[str] = None,
                                   **kwargs
                                   ):
    os.makedirs('../logs/training', exist_ok=True)
    log_filename = datetime.datetime.now().strftime(
        '../logs/training/' + synthetic_factor_name + "_%Y-%m-%d_%H_%M_%S.log")
    file_handler = logging.FileHandler(filename=log_filename)
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.INFO,
                        format='|%(levelname)s|%(asctime)s|%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=handlers)
    logger = logging.getLogger(synthetic_factor_name)

    factor_data.index.names = ['date', 'code']
    label_name = label.name
    if not label_name:
        label_name = 'label'
    # is_classification = _is_classification_problem(model, label_name)
    # if is_classification:
    #     classes = label.value_counts().sort_index().index
    # else:
    #     classes = None
    if factor_names:
        cols = factor_names + [label_name]
        factor_data = factor_data[factor_names]
    else:
        factor_names = factor_data.columns
        cols = factor_data.columns.tolist() + [label_name]
    label = label.to_frame()
    if label_end is not None:
        label = panel_df_join(label, label_end.to_frame('label_end'))
        # label = label.fillna()
    data = panel_df_join(factor_data, label)
    data = data.dropna()
    models = {}
    metrics = defaultdict(dict)
    features_importance = defaultdict(dict)
    synthetic_factor = []

    factor_dates = data.index.get_level_values(0).drop_duplicates().sort_values()
    last_training_date = factor_dates[0]
    first_date = factor_dates[0]

    for trade_date in factor_dates:
        logger.info('*' * 40)
        logger.info('rebalance: %s', trade_date)
        retrain = False
        # available factor data

        # available label data
        available_label_date = trade_date - pd.Timedelta(seconds=1)

        if isinstance(retrain_policy, List):
            if trade_date in retrain_policy:
                retrain = True
                last_training_date = trade_date
        elif isinstance(retrain_policy, CustomBusinessDay):
            raise NotImplementedError
        elif isinstance(retrain_policy, CustomBusinessMonthEnd):
            raise NotImplementedError
        elif callable(retrain_policy):
            raise NotImplementedError

        if retrain:
            if keeper is not None and len(synthetic_factor) > 0:
                factor_values = pd.concat(synthetic_factor)
                factor_values.name = synthetic_factor_name
                # kwargs['features_importance'] = features_importance
                # kwargs['metrics'] = metrics
                keeper.save_factor_value('synthetic_factor', factor_values, **kwargs)

            logger.info('retrain: %s', trade_date)

            available_data = data.loc[: available_label_date]
            if label_end is not None:
                available_data = available_data[available_data['label_end'] <= trade_date]

            available_data = available_data.groupby(level=1).tail(rolling_window_size)
            available_data = available_data.dropna()

            count = available_data.groupby(level=1).count().min(axis=1)
            idx = count[count == rolling_window_size].index
            available_data = available_data.loc[:, idx, :]

            if available_data.empty:
                logger.info('No data pass')
                continue

            X = available_data[factor_names]
            y = available_data[label_name]
            dts = X.index.get_level_values(0).drop_duplicates().sort_values()
            logger.info("Train model on data from {} to {}. Total sample size: {}".format(dts[0], dts[-1], len(X)))
            if sample_weights_policy and rolling_window_size:
                sample_weight = get_sample_weights(sample_weights_policy, X)
                model.fit(X, y, sample_weight=sample_weight)
            else:
                model.fit(X, y)
            models[trade_date] = model
            features_importance[trade_date] = model.get_feature_importance()

            if save_model_path is not None:
                model_name = trade_date.strftime('%Y-%m-%d')
                model.save_model(save_model_path, model_name)

        if len(models) == 0:
            continue

        # predict trade_date data
        test_X = data.loc[:trade_date][factor_names]
        y_true = data.loc[trade_date:trade_date][label_name]
        if predict_ensemble:
            if len(models.keys()) < predict_ensemble:
                continue
            model_keys = list(models.keys())[-predict_ensemble:]
            models_used = {k: v for k, v in models.items() if k in model_keys}
            weights = get_voting_weights(predict_ensemble_weights_policy, models_used)
            y_predict = get_regression_voting_predict(test_X, models_used, weights)
        else:
            y_predict = model.predict(test_X)

        synthetic_factor.append(y_predict)
        if keeper is not None and len(synthetic_factor) > 0:
            factor_values = pd.concat(synthetic_factor)
            factor_values.name = synthetic_factor_name
            keeper.save_factor_value('synthetic_factor', factor_values, **kwargs)

        try:
            if 'MSE' in model_metrics:
                mse = mean_squared_error(y_true.fillna(y_true.median()), y_predict)
                metrics[trade_date]["MSE"] = mse

            if "NDCG" in model_metrics:
                score = ndcg_score([y_true], [y_predict], k=150)
                metrics[trade_date]["NDCG@150"] = score
                score = ndcg_score([y_true], [y_predict], k=100)
                metrics[trade_date]["NDCG@100"] = score
                score = ndcg_score([y_true], [y_predict], k=50)
                metrics[trade_date]["NDCG@50"] = score

            if "RET" in model_metrics:
                ret = (y_predict * y_true).mean()
                metrics[trade_date]['RET'] = ret

            logger.info(metrics[trade_date])
        except:
            pass

    metrics = pd.DataFrame(metrics).T
    features_importance = pd.DataFrame(features_importance).T
    synthetic_factor = pd.concat(synthetic_factor)
    synthetic_factor.name = synthetic_factor_name
    return synthetic_factor, metrics, features_importance, models
