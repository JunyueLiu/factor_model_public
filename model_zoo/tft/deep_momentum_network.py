import collections
import copy
import enum
from abc import abstractmethod
from collections import namedtuple
from typing import Tuple, Optional, List, Dict

import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from empyrical import sharpe_ratio
from tensorflow import keras

from data_management.pandas_utils.cache import panel_df_join
from model_zoo.BaseModel import BaseModel
from model_zoo.tft.settings.hp_grid import (
    HP_HIDDEN_LAYER_SIZE,
    HP_DROPOUT_RATE,
    HP_MAX_GRADIENT_NORM,
    HP_LEARNING_RATE,
)

TRAIN_VALID_RATIO = 0.90
# TIME_FEATURES = False
FORCE_OUTPUT_SHARPE_LENGTH = None
EVALUATE_DIVERSIFIED_VAL_SHARPE = True
early_stopping_patience = 25

MODLE_PARAMS = {
    "architecture": "TFT",
    "total_time_steps": 252,
    "early_stopping_patience": 25,
    "multiprocessing_workers": 32,
    "num_epochs": 300,
    "fill_blank_dates": False,
    "split_tickers_individually": True,
    "random_search_iterations": 50,
    "evaluate_diversified_val_sharpe": True,
    "train_valid_ratio": 0.90,
    "time_features": False,
    "force_output_sharpe_length": 0,
}


class DataTypes(enum.IntEnum):
    """Defines numerical types of each column."""

    REAL_VALUED = 0
    CATEGORICAL = 1
    DATE = 2


class InputTypes(enum.IntEnum):
    """Defines input types of each column."""

    TARGET = 0
    OBSERVED_INPUT = 1
    KNOWN_INPUT = 2
    STATIC_INPUT = 3
    ID = 4  # Single column used as an entity identifier
    TIME = 5  # Single column used as a time index


FeaturesLabel = namedtuple('FeaturesLabel', ["inputs", "outputs", "active_entries", "identifier", "date"])


def get_single_col_by_input_type(input_type, column_definition):
    """Returns name of single column.
    Args:
      input_type: Input type of column to extract
      column_definition: Column definition list for experiment
    """

    l = [tup[0] for tup in column_definition if tup[2] == input_type]

    if len(l) != 1:
        raise ValueError("Invalid number of columns for {}".format(input_type))

    return l[0]


def extract_cols_from_data_type(data_type, column_definition, excluded_input_types):
    """Extracts the names of columns that correspond to a define data_type.
    Args:
      data_type: DataType of columns to extract.
      column_definition: Column definition to use.
      excluded_input_types: Set of input types to exclude
    Returns:
      List of names for columns with data type specified.
    """
    return [
        tup[0]
        for tup in column_definition
        if tup[1] == data_type and tup[2] not in excluded_input_types
    ]


def extract_tuples_from_data_type(data_type, defn):
    return [
        tup
        for tup in defn
        if tup[1] == data_type
           and tup[2] not in {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET}
    ]

def get_locations(input_types, defn):
    return [i for i, tup in enumerate(defn) if tup[2] in input_types]


class SharpeLoss(tf.keras.losses.Loss):
    def __init__(self, output_size: int = 1):
        self.output_size = output_size  # in case we have multiple targets => output dim[-1] = output_size * n_quantiles
        super().__init__()

    def call(self, y_true, weights):
        captured_returns = weights * y_true
        mean_returns = tf.reduce_mean(captured_returns)
        return -(
                mean_returns
                / tf.sqrt(
            tf.reduce_mean(tf.square(captured_returns))
            - tf.square(mean_returns)
            + 1e-9
        )
                * tf.sqrt(252.0)
        )


class SharpeValidationLoss(keras.callbacks.Callback):
    # TODO check if weights already exist and pass in best sharpe
    def __init__(
            self,
            inputs,
            returns,
            time_indices,
            num_time,  # including a count for nulls which will be indexed as 0
            early_stopping_patience,
            n_multiprocessing_workers,
            weights_save_location="tmp/checkpoint",
            # verbose=0,
            min_delta=1e-4,
    ):
        super(keras.callbacks.Callback, self).__init__()
        self.inputs = inputs
        self.returns = returns
        self.time_indices = time_indices
        self.n_multiprocessing_workers = n_multiprocessing_workers
        self.early_stopping_patience = early_stopping_patience
        self.num_time = num_time
        self.min_delta = min_delta

        self.best_sharpe = np.NINF  # since calculating positive Sharpe...
        # self.best_weights = None
        self.weights_save_location = weights_save_location
        # self.verbose = verbose

    def set_weights_save_loc(self, weights_save_location):
        self.weights_save_location = weights_save_location

    def on_train_begin(self, logs=None):
        self.patience_counter = 0
        self.stopped_epoch = 0
        self.best_sharpe = np.NINF

    def on_epoch_end(self, epoch, logs=None):
        positions = self.model.predict(
            self.inputs,
            workers=self.n_multiprocessing_workers,
            use_multiprocessing=True,  # , batch_size=1
        )

        captured_returns = tf.math.unsorted_segment_mean(
            positions * self.returns, self.time_indices, self.num_time
        )[1:]
        # ignoring null times

        # TODO sharpe
        sharpe = (
                tf.reduce_mean(captured_returns)
                / tf.sqrt(
            tf.math.reduce_variance(captured_returns)
            + tf.constant(1e-9, dtype=tf.float64)
        )
                * tf.sqrt(tf.constant(252.0, dtype=tf.float64))
        ).numpy()
        if sharpe > self.best_sharpe + self.min_delta:
            self.best_sharpe = sharpe
            self.patience_counter = 0  # reset the count
            # self.best_weights = self.model.get_weights()
            self.model.save_weights(self.weights_save_location)
        else:
            # if self.verbose: #TODO
            self.patience_counter += 1
            if self.patience_counter >= self.early_stopping_patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.load_weights(self.weights_save_location)
        logs["sharpe"] = sharpe  # for keras tuner
        print(f"\nval_sharpe {logs['sharpe']}")


# Tuner = RandomSearch
class TunerValidationLoss(kt.tuners.RandomSearch):
    def __init__(
            self,
            hypermodel,
            objective,
            max_trials,
            hp_minibatch_size,
            seed=None,
            hyperparameters=None,
            tune_new_entries=True,
            allow_new_entries=True,
            **kwargs,
    ):
        self.hp_minibatch_size = hp_minibatch_size
        super().__init__(
            hypermodel,
            objective,
            max_trials,
            seed,
            hyperparameters,
            tune_new_entries,
            allow_new_entries,
            **kwargs,
        )

    def run_trial(self, trial, *args, **kwargs):
        kwargs["batch_size"] = trial.hyperparameters.Choice(
            "batch_size", values=self.hp_minibatch_size
        )
        super(TunerValidationLoss, self).run_trial(trial, *args, **kwargs)


class TunerDiversifiedSharpe(kt.tuners.RandomSearch):
    def __init__(
            self,
            hypermodel,
            objective,
            max_trials,
            hp_minibatch_size,
            seed=None,
            hyperparameters=None,
            tune_new_entries=True,
            allow_new_entries=True,
            **kwargs,
    ):
        self.hp_minibatch_size = hp_minibatch_size
        super().__init__(
            hypermodel,
            objective,
            max_trials,
            seed,
            hyperparameters,
            tune_new_entries,
            allow_new_entries,
            **kwargs,
        )

    def run_trial(self, trial, *args, **kwargs):
        kwargs["batch_size"] = trial.hyperparameters.Choice(
            "batch_size", values=self.hp_minibatch_size
        )

        original_callbacks = kwargs.pop("callbacks", [])

        for callback in original_callbacks:
            if isinstance(callback, SharpeValidationLoss):
                callback.set_weights_save_loc(
                    self._get_checkpoint_fname(trial.trial_id, self._reported_step)
                )

        # Run the training process multiple times.
        metrics = collections.defaultdict(list)
        for execution in range(self.executions_per_trial):
            copied_fit_kwargs = copy.copy(kwargs)
            callbacks = self._deepcopy_callbacks(original_callbacks)
            self._configure_tensorboard_dir(callbacks, trial, execution)
            callbacks.append(kt.engine.tuner_utils.TunerCallback(self, trial))
            # Only checkpoint the best epoch across all executions.
            # callbacks.append(model_checkpoint)
            copied_fit_kwargs["callbacks"] = callbacks

            history = self._build_and_fit_model(trial, args, copied_fit_kwargs)
            for metric, epoch_values in history.history.items():
                if self.oracle.objective.direction == "min":
                    best_value = np.min(epoch_values)
                else:
                    best_value = np.max(epoch_values)
                metrics[metric].append(best_value)

        # Average the results across executions and send to the Oracle.
        averaged_metrics = {}
        for metric, execution_values in metrics.items():
            averaged_metrics[metric] = np.mean(execution_values)
        self.oracle.update_trial(
            trial.trial_id, metrics=averaged_metrics, step=self._reported_step
        )


class DeepMomentumNetworkModel(BaseModel):
    def __init__(self,
                 total_time_steps: int,
                 input_size: int,
                 output_size: int,
                 hyperparameters: Optional[Dict[str, float]],
                 hp_minibatch_size: List[int],
                 *,
                 category_features: Optional[List] = None,
                 multiprocessing_workers: int = MODLE_PARAMS['multiprocessing_workers'],
                 num_epochs: int = MODLE_PARAMS['num_epochs'],
                 early_stopping_patience: int = MODLE_PARAMS['early_stopping_patience'],
                 random_search_iterations: int = MODLE_PARAMS['random_search_iterations'],
                 evaluate_diversified_val_sharpe: int = MODLE_PARAMS['evaluate_diversified_val_sharpe'],
                 force_output_sharpe_length: int = MODLE_PARAMS['force_output_sharpe_length'],
                 split_tickers_individually: int = MODLE_PARAMS['split_tickers_individually'],
                 train_valid_ratio: int = MODLE_PARAMS['train_valid_ratio'],
                 train_valid_sliding: bool = False,
                 project_name='DeepMomentumNetworkModel_project',
                 hp_directory='DeepMomentumNetworkModel_dir'):
        super().__init__()
        self.time_steps = total_time_steps
        self.input_size = input_size
        self.output_size = output_size
        self.n_multiprocessing_workers = multiprocessing_workers
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        # self.sliding_window = sliding_window
        self.random_search_iterations = random_search_iterations
        self.evaluate_diversified_val_sharpe = evaluate_diversified_val_sharpe
        self.force_output_sharpe_length = force_output_sharpe_length
        self.hyperparameters = hyperparameters
        self.split_tickers_individually = split_tickers_individually
        self.train_valid_ratio = train_valid_ratio
        self.train_valid_sliding = train_valid_sliding
        # self.validation_windows = validation_windows
        self.category_features = category_features

        self.model = None
        self.lags = force_output_sharpe_length

        # print("Deep Momentum Network params:")
        # for k in params:
        #     print(f"{k} = {params[k]}")

        # To build model
        self.tuner = None
        self.hp_minibatch_size = hp_minibatch_size
        self.project_name = project_name
        self.hp_directory = hp_directory


    @abstractmethod
    def model_builder(self, hp):
        return

    @staticmethod
    def _index_times(val_time):
        val_time_unique = np.sort(np.unique(val_time))
        if val_time_unique[0]:  # check if ""
            val_time_unique = np.insert(val_time_unique, 0, "")
        mapping = dict(zip(val_time_unique, range(len(val_time_unique))))

        @np.vectorize
        def get_indices(t):
            return mapping[t]

        return get_indices(val_time), len(mapping)

    def load_model(
            self, hyperparameters: Dict
    ) -> tf.keras.Model:
        hyp = kt.engine.hyperparameters.HyperParameters()
        hyp.values = hyperparameters
        return self.tuner.hypermodel.build(hyp)

    def set_tuner(self):
        def model_builder(hp):
            return self.model_builder(hp)

        if self.evaluate_diversified_val_sharpe:
            self.tuner = TunerDiversifiedSharpe(
                model_builder,
                # objective="val_loss",
                objective=kt.Objective("sharpe", "max"),
                hp_minibatch_size=self.hp_minibatch_size,
                max_trials=self.random_search_iterations,
                directory=self.hp_directory,
                project_name=self.project_name,
            )
        else:
            self.tuner = TunerValidationLoss(
                model_builder,
                objective="val_loss",
                hp_minibatch_size=self.hp_minibatch_size,
                max_trials=self.random_search_iterations,
                directory=self.hp_directory,
                project_name=self.project_name,
            )

    def reorder_input_df(self, raw_features: pd.DataFrame, raw_label: pd.Series):
        data = panel_df_join(raw_features, raw_label.to_frame())
        date_col, ticker_col = data.index.names
        label_col = raw_label.name
        features_cols = raw_features.columns.to_list()
        data = data.reset_index()
        # data[date_col] = data.index
        for c in self.category_features:
            features_cols.remove(c)

        data = data[[ticker_col, date_col, label_col] + features_cols + self.category_features]


        _column_definition = self.infer_column_definition(data, date_col, ticker_col, label_col)
        return data, _column_definition



    def infer_column_definition(self, raw_features_label, date_col, ticker_col, label_col):
        column_definition = []
        for c in raw_features_label:
            if c == date_col:
                column_definition.append((date_col, DataTypes.DATE, InputTypes.TIME))
            elif c == ticker_col:
                column_definition.append((ticker_col, DataTypes.CATEGORICAL, InputTypes.ID))
            elif c == label_col:
                column_definition.append((label_col, DataTypes.REAL_VALUED, InputTypes.TARGET))
            elif self.category_features and c in self.category_features:
                column_definition.append((c, DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT))
            else:
                column_definition.append((c, DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT))

        # column_definition.append((date_col, DataTypes.DATE, InputTypes.TIME))
        # column_definition.append((ticker_col, DataTypes.CATEGORICAL, InputTypes.ID))
        # column_definition.append((label_col, DataTypes.REAL_VALUED, InputTypes.TARGET))
        return column_definition

    def preprocess_raw_data(self, raw_features: pd.DataFrame, raw_label: pd.Series, train_valid_ratio) -> Tuple:
        """
        To replace the original ModelFeatures class
        Returns
        -------

        """

        # data = panel_df_join(raw_features, raw_label.to_frame())
        # date_col, ticker_col = data.index.names
        # label_col = raw_label.name
        #
        #
        # data = data.reset_index(level=1)
        # data[date_col] = data.index
        # _column_definition = self.infer_column_definition(data, date_col, ticker_col, label_col)
        #
        #
        # trainvalid = data
        trainvalid, _column_definition = self.reorder_input_df(raw_features, raw_label)
        if self.split_tickers_individually:
            if self.lags:
                tickers = (
                                  trainvalid.groupby(level=1).count()
                                  * (1.0 - train_valid_ratio)
                          ) >= self.time_steps
                tickers = tickers[tickers].index.tolist()
            else:
                tickers = list(trainvalid['code'].unique())

            train, valid = [], []
            for ticker in tickers:
                calib_data = trainvalid[trainvalid['code'] == ticker]
                T = len(calib_data)
                train_valid_split = int(train_valid_ratio * T)
                train.append(calib_data.iloc[:train_valid_split, :].copy())
                valid.append(calib_data.iloc[train_valid_split:, :].copy())

            train = pd.concat(train)
            valid = pd.concat(valid)

        else:
            # todo not test
            dates = trainvalid.index.get_level_values(0).drop_duplicates().sort_values()
            split_index = int(train_valid_ratio * len(dates))
            # train_dates = pd.DataFrame({"date": dates[:split_index]})
            # valid_dates = pd.DataFrame({"date": dates[split_index:]})

            train = trainvalid.loc[:dates[split_index]]
            valid = trainvalid.loc[dates[split_index]:]

            if self.lags:
                tickers = (
                        valid.groupby(level=1).count() > self.time_steps
                )
                tickers = tickers[tickers].index.tolist()
                train = train[train.ticker.isin(tickers)]

            else:
                # at least one full training sequence
                # tickers = (
                #     train.groupby("ticker")["ticker"].count() > self.total_time_steps
                # )
                # tickers = tickers[tickers].index.tolist()
                tickers = list(train.ticker.unique())
            valid = valid[valid.ticker.isin(tickers)]

            # self.tickers = tickers
            # self.num_tickers = len(tickers)
            # self.set_scalers(train)

        if self.lags:
            # todo not test
            train = self._batch_data_smaller_output(
                train, self.train_valid_sliding, self.lags
            )
            valid = self._batch_data_smaller_output(
                valid, self.train_valid_sliding, self.lags
            )
            # self.test_fixed = self._batch_data_smaller_output(test, False, self.lags)
            # self.test_sliding = self._batch_data_smaller_output(
            #     test_with_buffer, True, self.lags
            # )
        else:
            train = self._batch_data(train, self.train_valid_sliding, _column_definition)
            if not valid.empty:
                valid = self._batch_data(valid, self.train_valid_sliding, _column_definition)

        # train_inputs, train_outputs, train_active_entries, train_identifier, train_date = self._unpack(train)
        # valid_inputs, valid_outputs, valid_active_entries, valid_identifier, valid_date = self._unpack(valid)

        return train, valid

    def _unpack(self, data):
        return (
            data["inputs"],
            data["outputs"],
            data["active_entries"],
            data["identifier"],
            data["date"],
        )

    def _batch_data(self, data, sliding_window, _column_definition):
        """Batches data for training.

        Converts raw dataframe from a 2-D tabular format to a batched 3-D array
        to feed into Keras model.

        Args:
          data: DataFrame to batch

        Returns:
          Batched Numpy array with shape=(?, self.time_steps, self.input_size)
        """
        # TODO this works but is a bit of a mess
        data = data.copy()
        data["date"] = data["date"].dt.strftime("%Y-%m-%d %H:%M:%S")

        id_col = get_single_col_by_input_type(InputTypes.ID, _column_definition)
        time_col = get_single_col_by_input_type(
            InputTypes.TIME, _column_definition
        )
        target_col = get_single_col_by_input_type(
            InputTypes.TARGET, _column_definition
        )

        input_cols = [
            tup[0]
            for tup in _column_definition
            if tup[2] not in {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET}
        ]

        data_map = {}

        if sliding_window:
            # Functions.
            def _batch_single_entity(input_data):
                time_steps = len(input_data)
                lags = self.time_steps  # + int(self.extra_lookahead_steps)
                x = input_data.values
                if time_steps >= lags:
                    return np.stack(
                        [x[i: time_steps - (lags - 1) + i, :] for i in range(lags)],
                        axis=1,
                    )
                else:
                    return None

            for _, sliced in data.groupby(id_col):

                col_mappings = {
                    "identifier": [id_col],
                    "date": [time_col],
                    "outputs": [target_col],
                    "inputs": input_cols,
                }

                for k in col_mappings:
                    cols = col_mappings[k]
                    arr = _batch_single_entity(sliced[cols].copy())

                    if k not in data_map:
                        data_map[k] = [arr]
                    else:
                        data_map[k].append(arr)

            # Combine all data
            for k in data_map:
                data_map[k] = np.concatenate(data_map[k], axis=0)

            active_entries = np.ones_like(data_map["outputs"])
            if "active_entries" not in data_map:
                data_map["active_entries"] = active_entries
            else:
                data_map["active_entries"].append(active_entries)

        else:
            for _, sliced in data.groupby(id_col):

                col_mappings = {
                    "identifier": [id_col],
                    "date": [time_col],
                    "inputs": input_cols,
                    "outputs": [target_col],
                }

                time_steps = len(sliced)
                lags = self.time_steps
                additional_time_steps_required = lags - (time_steps % lags)

                def _batch_single_entity(input_data):
                    x = input_data.values
                    if additional_time_steps_required > 0:
                        x = np.concatenate(
                            [x, np.zeros((additional_time_steps_required, x.shape[1]))]
                        )
                    return x.reshape(-1, lags, x.shape[1])

                # for k in col_mappings:
                k = "outputs"
                cols = col_mappings[k]
                arr = _batch_single_entity(sliced[cols].copy())

                batch_size = arr.shape[0]
                sequence_lengths = [
                    (
                        lags
                        if i != batch_size - 1
                        else lags - additional_time_steps_required
                    )
                    for i in range(batch_size)
                ]
                active_entries = np.ones((arr.shape[0], arr.shape[1], arr.shape[2]))
                for i in range(batch_size):
                    active_entries[i, sequence_lengths[i]:, :] = 0
                sequence_lengths = np.array(sequence_lengths, dtype=np.int)

                if "active_entries" not in data_map:
                    data_map["active_entries"] = [
                        active_entries[sequence_lengths > 0, :, :]
                    ]
                else:
                    data_map["active_entries"].append(
                        active_entries[sequence_lengths > 0, :, :]
                    )

                if k not in data_map:
                    data_map[k] = [arr[sequence_lengths > 0, :, :]]
                else:
                    data_map[k].append(arr[sequence_lengths > 0, :, :])

                for k in set(col_mappings) - {"outputs"}:
                    cols = col_mappings[k]
                    arr = _batch_single_entity(sliced[cols].copy())

                    if k not in data_map:
                        data_map[k] = [arr[sequence_lengths > 0, :, :]]
                    else:
                        data_map[k].append(arr[sequence_lengths > 0, :, :])

            # Combine all data
            for k in data_map:
                data_map[k] = np.concatenate(data_map[k], axis=0)

        active_flags = (np.sum(data_map["active_entries"], axis=-1) > 0.0) * 1.0
        data_map["inputs"] = data_map["inputs"][: len(active_flags)]
        data_map["outputs"] = data_map["outputs"][: len(active_flags)]
        data_map["active_entries"] = active_flags
        data_map["identifier"] = data_map["identifier"][: len(active_flags)]
        data_map["identifier"][data_map["identifier"] == 0] = ""
        data_map["date"] = data_map["date"][: len(active_flags)]
        data_map["date"][data_map["date"] == 0] = ""
        return data_map

    def hyperparameter_search(self, train_data, valid_data):
        data, labels, active_flags, _, _ = self._unpack(train_data)
        val_data, val_labels, val_flags, _, val_time = self._unpack(valid_data)

        if self.evaluate_diversified_val_sharpe:
            val_time_indices, num_val_time = self._index_times(val_time)
            callbacks = [
                SharpeValidationLoss(
                    val_data,
                    val_labels,
                    val_time_indices,
                    num_val_time,
                    self.early_stopping_patience,
                    self.n_multiprocessing_workers,
                ),
                tf.keras.callbacks.TerminateOnNaN(),
            ]
            # self.model.run_eagerly = True
            self.tuner.search(
                x=data,
                y=labels,
                sample_weight=active_flags,
                epochs=self.num_epochs,
                # batch_size=minibatch_size,
                # covered by Tuner class
                callbacks=callbacks,
                shuffle=True,
                use_multiprocessing=True,
                workers=self.n_multiprocessing_workers,
            )
        else:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.early_stopping_patience,
                    min_delta=1e-4,
                ),
                # tf.keras.callbacks.TerminateOnNaN(),
            ]
            # self.model.run_eagerly = True
            self.tuner.search(
                x=data,
                y=labels,
                sample_weight=active_flags,
                epochs=self.num_epochs,
                # batch_size=minibatch_size,
                # covered by Tuner class
                validation_data=(
                    val_data,
                    val_labels,
                    val_flags,
                ),
                callbacks=callbacks,
                shuffle=True,
                use_multiprocessing=True,
                workers=self.n_multiprocessing_workers,
                # validation_batch_size=1,
            )

        best_hp = self.tuner.get_best_hyperparameters(num_trials=1)[0].values
        best_model = self.tuner.get_best_models(num_models=1)[0]
        return best_hp, best_model

    def fit(
            self,
            train_data,
            labels,
            sample_weight=None
    ):

        train_data, valid_data = self.preprocess_raw_data(train_data, labels, self.train_valid_ratio)
        # data (batch, time_steps, features)
        # labels (batch, time_steps, 1)
        #  active_flags (batch, time_stemps)
        data, labels, active_flags, _, _ = self._unpack(train_data)
        val_data, val_labels, val_flags, _, val_time = self._unpack(valid_data)

        self.set_tuner()
        if self.hyperparameters:
            self.model = self.load_model(self.hyperparameters)
        else:
            best_hp, best_model = self.hyperparameter_search(train_data, valid_data)
            self.hyperparameters = best_hp
            self.model = self.load_model(best_hp)
            print(best_hp)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.early_stopping_patience,
                min_delta=1e-4,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.TerminateOnNaN(),
        ]
        # self.model.run_eagerly = True
        self.model.fit(
            x=data,
            y=labels,
            sample_weight=active_flags,
            epochs=self.num_epochs,
            batch_size=self.hyperparameters['batch_size'],
            validation_data=(
                val_data,
                val_labels,
                val_flags,
            ),
            callbacks=callbacks,
            shuffle=True,
            use_multiprocessing=True,
            workers=self.n_multiprocessing_workers,
        )
        return self.model

    def is_classifier(self):
        return False

    def predict(self, X):
        test_df = X.groupby(level=1).tail(self.time_steps)

        label_placeholder = pd.Series(1, index=test_df.index, name='label_placeholder')
        test_data, _ = self.preprocess_raw_data(test_df, label_placeholder, 1)
        res, _ = self.get_positions(test_data)
        series = res.set_index(['date', 'code'])['position']
        return series

    def predict_proba(self, X):
        pass

    def get_feature_importance(self):
        return []

    def evaluate(self, data, model):
        """Applies evaluation metric to the training data.

        Args:
          data: Dataframe for evaluation
          eval_metric: Evaluation metic to return, based on model definition.

        Returns:
          Computed evaluation loss.
        """

        inputs, outputs, active_entries, _, _ = self._unpack(data)

        if self.evaluate_diversified_val_sharpe:
            _, performance = self.get_positions(data, model, False)
            return performance

        else:
            metric_values = model.evaluate(
                x=inputs,
                y=outputs,
                sample_weight=active_entries,
                workers=32,
                use_multiprocessing=True,
            )

            metrics = pd.Series(metric_values, model.metrics_names)
            return metrics["loss"]

    def get_positions(
            self,
            data,
            sliding_window=True,
            years_geq=np.iinfo(np.int32).min,
            years_lt=np.iinfo(np.int32).max,
    ):
        inputs, outputs, _, identifier, time = self._unpack(data)
        if sliding_window:
            time = pd.to_datetime(
                time[:, -1, 0].flatten()
            )  # TODO to_datetime maybe not needed
            years = time.map(lambda t: t.year)
            identifier = identifier[:, -1, 0].flatten()
            returns = outputs[:, -1, 0].flatten()
        else:
            time = pd.to_datetime(time.flatten())
            years = time.map(lambda t: t.year)
            identifier = identifier.flatten()
            # returns = outputs.flatten()
        mask = (years >= years_geq) & (years < years_lt)

        positions = self.model.predict(
            inputs,
            workers=self.n_multiprocessing_workers,
            use_multiprocessing=True,  # , batch_size=1
        )
        if sliding_window:
            positions = positions[:, -1, 0].flatten()
        else:
            positions = positions.flatten()

        captured_returns = returns * positions
        results = pd.DataFrame(
            {
                "code": identifier[mask],
                "date": time[mask],
                "returns": returns[mask],
                "position": positions[mask],
                "captured_returns": captured_returns[mask],
            }
        )

        # don't need to divide sum by n because not storing here
        # mean does not work as well (related to days where no information)
        performance = sharpe_ratio(results.groupby("date")["captured_returns"].sum())

        return results, performance


class LstmDeepMomentumNetworkModel(DeepMomentumNetworkModel):
    def __init__(
            self, total_time_steps,
            input_size,
            output_size,
            hyperparameters,
            hp_minibatch_size,
            *,
            category_features: Optional[List] = None,
            multiprocessing_workers: int = MODLE_PARAMS['multiprocessing_workers'],
            num_epochs: int = MODLE_PARAMS['num_epochs'],
            early_stopping_patience: int = MODLE_PARAMS['early_stopping_patience'],
            random_search_iterations: int = MODLE_PARAMS['random_search_iterations'],
            evaluate_diversified_val_sharpe: int = MODLE_PARAMS['evaluate_diversified_val_sharpe'],
            force_output_sharpe_length: int = MODLE_PARAMS['force_output_sharpe_length'],
            split_tickers_individually: int = MODLE_PARAMS['split_tickers_individually'],
            train_valid_ratio: int = MODLE_PARAMS['train_valid_ratio'],
            train_valid_sliding: bool = False,
            project_name='project_lstm',
            hp_directory='lstm_dir'
    ):
        super().__init__(total_time_steps, input_size, output_size, hyperparameters, hp_minibatch_size,
                         category_features=category_features,
                         multiprocessing_workers=multiprocessing_workers,
                         num_epochs=num_epochs,
                         early_stopping_patience=early_stopping_patience,
                         random_search_iterations=random_search_iterations,
                         evaluate_diversified_val_sharpe=evaluate_diversified_val_sharpe,
                         force_output_sharpe_length=force_output_sharpe_length,
                         split_tickers_individually=split_tickers_individually,
                         train_valid_ratio=train_valid_ratio,
                         train_valid_sliding=train_valid_sliding,
                         project_name=project_name,
                         hp_directory=hp_directory,
                         )

    def model_builder(self, hp):
        hidden_layer_size = hp.Choice("hidden_layer_size", values=HP_HIDDEN_LAYER_SIZE)
        dropout_rate = hp.Choice("dropout_rate", values=HP_DROPOUT_RATE)
        max_gradient_norm = hp.Choice("max_gradient_norm", values=HP_MAX_GRADIENT_NORM)
        learning_rate = hp.Choice("learning_rate", values=HP_LEARNING_RATE)
        # minibatch_size = hp.Choice("hidden_layer_size", HP_MINIBATCH_SIZE)

        input = keras.Input((self.time_steps, self.input_size))
        lstm = tf.keras.layers.LSTM(
            hidden_layer_size,
            return_sequences=True,
            dropout=dropout_rate,
            stateful=False,
            activation="tanh",
            recurrent_activation="sigmoid",
            recurrent_dropout=0,
            unroll=False,
            use_bias=True,
        )(input)
        dropout = keras.layers.Dropout(dropout_rate)(lstm)

        output = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(
                self.output_size,
                activation=tf.nn.tanh,
                kernel_constraint=keras.constraints.max_norm(3),
            )
        )(dropout[..., :, :])

        model = keras.Model(inputs=input, outputs=output)

        adam = keras.optimizers.Adam(lr=learning_rate, clipnorm=max_gradient_norm)

        sharpe_loss = SharpeLoss(self.output_size).call

        model.compile(
            loss=sharpe_loss,
            optimizer=adam,
            sample_weight_mode="temporal",
        )
        return model
