import os
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm as tqdm
import matplotlib.cm as cm
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.layers import (BatchNormalization, Dense, Dropout, Flatten, Input, Lambda,
                                     TimeDistributed, LSTM, Conv1D, concatenate, add, Reshape,
                                     AveragePooling1D, MaxPooling1D, GlobalMaxPooling1D, Embedding,
                                     Masking, RepeatVector)

from tensorflow.keras.layers.experimental import preprocessing

from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn import tree
import pickle

AUTOTUNE = tf.data.experimental.AUTOTUNE


class WindowGenerator():
    def __init__(self, input_width, label_width, offset,
                train_dataset=None, test_dataset=None, valid_dataset=None,
                label_columns=None, batch_size=128, pad_sequences=True, shuffle=False):
        
        self.train_ds = train_dataset
        self.test_ds = test_dataset
        self.valid_ds = valid_dataset
        
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.pad_sequences = pad_sequences
        
        # preprocessing inputs
#         self.normalization = preprocessing.Normalization()
#         self.normalization.adapt(self.train_ds.values)
        
        # define label column indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in 
                                  enumerate(label_columns)}
        self.column_indices = {name: i for i, name in 
                               enumerate(train_dataset.columns)}
        
        # define window parameters
        self.input_width = input_width
        self.label_width = label_width
        self.offset = offset
        
        self.total_window_size = input_width + offset
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        
    def split_window(self, features):
        """ function to split the examples into single/multi output examples."""
#         inputs = self.normalization(features[:,self.input_slice,:])
        inputs = features[:,self.input_slice,:]
        labels = features[:,self.labels_slice,:]
        if self.label_columns is not None:
            labels = tf.stack([labels[:,:, self.column_indices[name]] for name in self.label_columns], 
                              axis=-1)
        
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        
        return inputs, labels

    def make_dataset(self, data, sequence_stride=1):
        """ Function to create tensorflow dataset pipeline."""
        data = np.array(data, dtype=np.float32)


        ds = tf.keras.preprocessing.timeseries_dataset_from_array(data=data,
                                                                 targets=None,
                                                                 sequence_length=self.total_window_size,
                                                                 sequence_stride=sequence_stride,
                                                                 shuffle=self.shuffle,
                                                                 batch_size=self.batch_size)
        if self.pad_sequences:
            # pad the sequences to have the same shape
            ds = tf.keras.preprocessing.sequence.pad_sequences(ds, 
                                                               maxlen=None, 
                                                               dtype="float32", 
                                                               padding="post", 
                                                               truncating="pre", 
                                                               value=-2)

            ds = tf.data.Dataset.from_tensor_slices(ds)
        
        ds = ds.map(self.split_window)   
        return ds
    
    @property
    def train(self):
        return self.make_dataset(self.train_ds)
    
    @property
    def test(self):
        return self.make_dataset(self.test_ds)
    
    @property
    def valid(self):
        return self.make_dataset(self.valid_ds)
    
    @property
    def example(self):
        """ Get and cache example batch of 'inputs, labels' for plotting purposes."""
        
        result = getattr(self, '_example', None)
        if result is None:
            # no example batch was found, so get one from train
            result = next(iter(self.train))
            # cache it for the next time
            self._example = result
        
        return result
    
    def plot(self, model, plot_col, max_subplots=3, path=False, data=None, scatter=True, title=None, figsize=None, scaler=None):
        """Data window visualization"""
        
        inputs, labels = data if data else self.example
        if scaler:
            inputs = np.expand_dims(scaler.inverse_transform(np.squeeze(inputs, axis=0)), axis=0)
            labels = np.expand_dims(scaler.inverse_transform(np.squeeze(labels, axis=0)), axis=0)
            
        plt.figure(figsize=figsize if figsize else (10, 3*max_subplots))
        plot_col_index = self.column_indices[plot_col]
        
        max_n = min(max_subplots, len(inputs))
        
        for n in range(max_n):
            plt.subplot(max_subplots, 1, n+1)
            plt.ylabel(f'{plot_col} [MWh]')
            if title:
                plt.title(title)
                
            plt.plot(self.input_indices, inputs[n,:,plot_col_index],
                    label='History', marker='.', zorder=-10)
            
            if self.label_columns is not None:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index
                
            if plot_col_index is None:
                continue
            if scatter:
                plt.scatter(self.label_indices, labels[n,:,label_col_index],
                            edgecolors='k', label='Ground truth', c='g', s=64)
            else:
                plt.plot(self.label_indices, labels[n,:,label_col_index],
                         marker='.', label='Ground truth', c='g')
            
            if model is not None:
                predictions = model(inputs)
                if scatter:
                    plt.scatter(self.label_indices, predictions[n,:,label_col_index] if len(predictions.shape) == 3 else predictions[n, label_col_index],
                                edgecolors='k', marker='X',label='Predictions', c='r', s=64)
                else:
                    plt.plot(self.label_indices, predictions[n,:,label_col_index] if len(predictions.shape) == 3 else predictions[n, label_col_index],
                           marker='.', label='Predictions', c='r')
                
            if n==0:
                plt.legend()
                
        plt.xlabel('Time [h]')
        plt.tight_layout()
        
        if path:
            plt.savefig(path)
        plt.show()
        
    def prediction_scatter_plot(self, predictions, targets, labels=None, path=None, name=None):
        num_plots = len(predictions)
        
        fig, ax = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
        
        if num_plots>1:
            for i in range(num_plots):
                ax[i].scatter(targets, predictions)
                ax[i].set_title("Model - {} predictions".format(labels[i]))
                ax[i].set_xlabel('targets')
                ax[i].set_ylabel('predictions')
                
        else:
            ax.scatter(targets, predictions)
            ax.set_title("{} - {} predictions".format(name if name else 'Model',labels))
            ax.set_xlabel('targets')
            ax.set_ylabel('predictions')
            
        plt.tight_layout()
        if path:
            plt.savefig(path)
        plt.show()
        
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input Indices: {self.input_indices}',
            f'Label Indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'
        ])


class NNModels():
    def __init__(self,input_steps, output_steps, input_features_size, train_ds, output_features_size=None, batch_size=128, single_step=False):
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.input_features_size = input_features_size
        self.output_features_size = output_features_size if output_features_size else self.input_features_size
        self.batch_size = batch_size
        self.single_step = single_step
        
        self.train_ds = train_ds
        # preprocessing inputs
        self.normalization = preprocessing.Normalization()
        self.normalization.adapt(self.train_ds.values)
        self.normed_ds = self.normalization(self.train_ds.values).numpy()
        
        self.training_logs = {}
        
    def get_dense(self, units, 
                  optimizer=tf.optimizers.Adam(), 
                  loss=tf.losses.MeanSquaredError(), 
                  metrics=tf.metrics.RootMeanSquaredError()):
        
        _input = Input(shape=(self.input_steps, self.input_features_size))
        
        x = self.normalization(_input)
    
        x = Flatten()(x)

        x = Dense(units, activation='relu')(x)

        x = Dense(units, activation='relu')(x)
        
        if self.single_step:
            output = Dense(self.output_features_size)(x)
            
        else:
            # shape --> [batch, 1, OUTSTEPS * n_features]
            y = Dense(self.output_steps*self.output_features_size,
                     kernel_initializer=tf.initializers.zeros)(x)

            # shape --> [batch, OUT_STEPS, n_features]
            output = Reshape([self.output_steps, self.output_features_size])(y)

        self.dense_model = Model(_input, output)

        self.dense_model.compile(optimizer=optimizer,
           loss=loss,
           metrics=[metrics])
    
        self.dense_model.summary()
        
        return self.dense_model
    
    def get_cnn(self, units, kernel_size, pool_size=3,
                optimizer=tf.optimizers.Adam(), 
                loss=tf.losses.MeanSquaredError(), 
                metrics=tf.metrics.RootMeanSquaredError()):
        
        _input = Input(shape=(self.input_steps, self.input_features_size))
    
        x = self.normalization(_input)
        
        # CNN can take fixed-width timestep;
        # shape --> [batch, CONV_WIDTH, n_features]
        # x = Lambda(lambda x : x[:, -kernel_size:, :])(x)

        # shape --> [batch, 1, conv_units]
        x = Conv1D(units, kernel_size=(kernel_size,), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size)(x)

        x = Conv1D(units, kernel_size=(kernel_size,), padding='same', activation='relu')(x)
        x = GlobalMaxPooling1D()(x)
        
        if self.single_step:
            output = Dense(self.output_features_size)(x)
            
        else:
            # shape --> [batch, 1, OUTSTEPS * n_features]
            y = Dense(self.output_steps * self.output_features_size, 
                      kernel_initializer=tf.initializers.zeros)(x)

            # shape --> [batch, OUT_STEPS, n_features]
            output = Reshape([self.output_steps, self.output_features_size])(y)

        self.cnn_model = Model(_input, output)

        self.cnn_model.compile(optimizer=optimizer,
                   loss=loss,
                   metrics=[metrics])
    
        self.cnn_model.summary()

        return self.cnn_model

    def get_cnnlstm(self, cnn_units, lstm_units, kernel_size, dropout=0.3,
                    optimizer=tf.optimizers.Adam(),
                    loss=tf.losses.MeanSquaredError(),
                    metrics=tf.metrics.RootMeanSquaredError()):
        
        _input = Input(shape=(self.input_steps, self.input_features_size))
    
        x = self.normalization(_input)

        x = Conv1D(cnn_units, kernel_size=(kernel_size,), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)

        x = Conv1D(cnn_units, kernel_size=(kernel_size,), padding='same', activation='relu')(x)

        x = BatchNormalization()(x)
    
        x = TimeDistributed(Dense(lstm_units,
                                  kernel_initializer=tf.initializers.zeros))(x)
        
        x = LSTM(lstm_units, return_sequences=True)(x)

        x = Dropout(dropout)(x)
        
        if self.single_step:
            x = LSTM(lstm_units, return_sequences=True)(x)
            
            y = Dropout(dropout)(x)
            
            output = TimeDistributed(Dense(self.output_features_size))(y)
            
        else:
            x = LSTM(lstm_units, return_sequences=True)(x)

            x = Dropout(dropout)(x)

            output = TimeDistributed(Dense(self.output_features_size))(x)
            

        self.cnnlstm_model = Model(_input, output)

        self.cnnlstm_model.compile(optimizer=optimizer,
                   loss=loss,
                   metrics=[metrics])
    
        self.cnnlstm_model.summary()

        return self.cnnlstm_model
    
    
    def get_lstm(self, units,
                 optimizer=tf.optimizers.Adam(), 
                 loss=tf.losses.MeanSquaredError(), 
                 metrics=tf.metrics.RootMeanSquaredError(),
                 dropout=0.3):
        
        _input = Input(shape=(self.input_steps, self.input_features_size))
    
        x = self.normalization(_input)
            
        x = LSTM(units, return_sequences=True)(x)

        x = Dropout(dropout)(x)

        if self.single_step:
            
            x = LSTM(units, return_sequences=True)(x)

            x = Dropout(dropout)(x)
            
            output = TimeDistributed(Dense(self.output_features_size))(x)
            
        else:
            
            x = LSTM(units, return_sequences=True)(x)

            x = Dropout(dropout)(x)

            x = TimeDistributed(Dense(self.output_steps * self.output_features_size,
                                      kernel_initializer=tf.initializers.zeros))(x)

            output = TimeDistributed(Dense(self.output_features_size))(x)

        self.lstm_model = Model(_input, output)

        self.lstm_model.compile(optimizer=optimizer,
                             loss=loss,
                             metrics=[metrics])

        self.lstm_model.summary()

        return self.lstm_model
    
    def get_seq2seq_lstm(self, units,
                         optimizer=tf.optimizers.Adam(),
                         loss=tf.losses.MeanSquaredError(),
                         metrics=tf.metrics.RootMeanSquaredError(),
                         dropout=0.3):
        
        _input = Input(shape=(self.input_steps, self.input_features_size))
    
        x = self.normalization(_input)
         
        x = LSTM(units, return_sequences=False)(x)

        x = Dropout(dropout)(x)
        
        x = RepeatVector(self.input_steps)(x)
                    
        x = LSTM(units, return_sequences=True)(x)

        x = Dropout(dropout)(x)

        if self.single_step:
            output = TimeDistributed(Dense(self.output_features_size))(x)
            
        else:
            x = TimeDistributed(Dense(self.output_steps * self.output_features_size,
                                      kernel_initializer=tf.initializers.zeros))(x)

            output = TimeDistributed(Dense(self.output_features_size))(x)

        self.seq2seq_lstm_model = Model(_input, output)

        self.seq2seq_lstm_model.compile(optimizer=optimizer,
                             loss=loss,
                             metrics=[metrics])

        self.seq2seq_lstm_model.summary()

        return self.seq2seq_lstm_model
    
    def get_seq2seq_cnnlstm(self, cnn_units, lstm_units, kernel_size, pool_size=2,
                         optimizer=tf.optimizers.Adam(),
                         loss=tf.losses.MeanSquaredError(),
                         metrics=tf.metrics.RootMeanSquaredError(),
                         dropout=0.3):
        
        _input = Input(shape=(self.input_steps, self.input_features_size))
    
        x = self.normalization(_input)
        
        x = Conv1D(cnn_units, kernel_size=(kernel_size,), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)

        x = Conv1D(cnn_units, kernel_size=(kernel_size,), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        
        x = MaxPooling1D(pool_size)(x)
        
        x = Flatten()(x)
        
        x = RepeatVector(self.input_steps)(x)
                    
        x = LSTM(lstm_units, return_sequences=True)(x)

        x = Dropout(dropout)(x)

        if self.single_step:
            output = TimeDistributed(Dense(self.output_features_size))(x)
            
        else:
            x = TimeDistributed(Dense(self.output_steps * self.output_features_size,
                                      kernel_initializer=tf.initializers.zeros))(x)

            output = TimeDistributed(Dense(self.output_features_size))(x)

        self.seq2seq_cnnlstm_model = Model(_input, output)

        self.seq2seq_cnnlstm_model.compile(optimizer=optimizer,
                             loss=loss,
                             metrics=[metrics])

        self.seq2seq_cnnlstm_model.summary()

        return self.seq2seq_cnnlstm_model
    
    def get_stateful_lstm(self, units, dropout=0.3, 
                          optimizer=tf.optimizers.Adam(), 
                          loss=tf.losses.MeanSquaredError(), 
                          metrics=tf.metrics.RootMeanSquaredError()):
        _input = Input(batch_shape=[self.batch_size, self.input_steps, self.input_features_size])

        x = self.normalization(_input)
        
        # shape [batch, time, n_features] --> [batch, lstm_units]
        # since LSTM, here, needs to predict the last time step,
        # we turn the return_sequence=False (True in case of TimeDistributed Layer)
    #     y = Embedding(input_dim=n_features, output_dim=1024, name='embed1')(_input)
    
        x = LSTM(units, return_sequences=True, stateful=True)(x)

        x = Dropout(dropout)(x)

        x = LSTM(units, return_sequences=True, stateful=True)(x)

        x = Dropout(dropout)(x)
        
        if self.single_step:
            output = Dense(1)(x)
        else:
            # shape --> [batch, OUT_STEPS * n_features]
            output = TimeDistributed(Dense(self.output_features_size,
                                 kernel_initializer=tf.initializers.zeros))(x)

            # shape --> [batch, OUT_STEPS, n_features]
        #     output = Reshape([output_steps, n_features])(y)

        self.stateful_lstm_model = Model(_input, output)
        
        self.stateful_lstm_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=[metrics])
    
        self.stateful_lstm_model.summary()

        return self.stateful_lstm_model
    
    
    def get_callbacks(self, model_name, patience):
        if not os.path.exists(os.path.join('logs', model_name)):
            os.mkdir(os.path.join('logs', model_name))
        if not os.path.exists(os.path.join('weights', model_name)):
            os.mkdir(os.path.join('weights', model_name))

        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=os.path.join('logs', model_name),
                                           histogram_freq=1,
                                           embeddings_freq=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.1,
                                                 patience=5),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                             patience=patience,
                                             mode='min'),
            tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('weights', model_name),
                                               save_weights_only=True,
                                               monitor='val_root_mean_squared_error',
                                               mode='max',
                                               save_best_only=True)
                    ]
        return callbacks
    
    
    def training_violin_plot(self, path=None):
        df_normed = pd.DataFrame(self.normed_ds, columns=self.train_ds.columns, index=self.train_ds.index)

        df = df_normed.melt(var_name='Column', value_name='Normalized')
        plt.figure(figsize=(12, 6))
        ax = sns.violinplot(x='Column', y='Normalized', data=df)
        _ = ax.set_xticklabels(self.train_ds.keys(), rotation=90)
        plt.title("Violin plot of normalized data")
        if path:
            plt.savefig(path)
        plt.tight_layout()
        plt.show()
        
        
    def train(self, model, train_data, valid_data, model_name, epochs, callbacks=True, callbacks_patience=4, verbose=1):
        if callbacks:
            self.callbacks = self.get_callbacks(model_name, callbacks_patience)
        
        history = model.fit(train_data, validation_data=valid_data, 
                            epochs=epochs, 
                            batch_size=self.batch_size, 
                            callbacks=self.callbacks if callbacks else None,
                            verbose=verbose)
        
        history_df = pd.DataFrame(history.history)
        self.training_logs[model_name] = history_df
        
        return history_df
    
class ResidualWrapper(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)

        # The prediction for each timestep is the input
        # from the previous time step plus the delta
        # calculated by the model.
        return inputs + delta


class AutoRegressiveLSTM(tf.keras.Model):
    def __init__(self, units, out_steps, n_features, train_ds, dropout=0.2):
        super().__init__()
        
        self.out_steps = out_steps
        self.units = units
        
        self.train_ds = train_ds
        # preprocessing inputs
        self.normalization = preprocessing.Normalization()
        self.normalization.adapt(self.train_ds.values)
        
        # LSTM cells
        self.lstm_cell1 = tf.keras.layers.LSTMCell(units, recurrent_dropout=dropout)
        self.lstm_cell2 = tf.keras.layers.LSTMCell(units, recurrent_dropout=dropout)
#         self.lstm_cell3 = tf.keras.experimental.PeepholeLSTMCell(units, recurrent_dropout=dropout)
        
        # wrap LSTMCell in RNN
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell1, return_state=True)
        self.dense = tf.keras.layers.Dense(n_features)
        
    def warmup(self, inputs):
        # normalizing the inputs
        
        x = self.normalization(inputs)
        
        # masking the -2.0 values used to pad the sequences
        x = Masking(mask_value=-2.0)(x)
        
        # inputs shape --> [batch, time, n_features]
        # x.shape --> [batch, lstm_units]
        x, *state = self.lstm_rnn(x)
        
        # predictions.shape --> [batch, n_features]
        predictions = self.dense(x)
        
        return predictions, state
    
    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # normalizing the inputs
        x = self.normalization(inputs)
        
        # masking the -2.0 values used to pad the sequences
        x = Masking(mask_value=-2.0)(x)
        
        # Initialize the lstm state
        prediction, state = self.warmup(x)

        # Insert the first prediction
        predictions.append(prediction)

        # Run the rest of the prediction steps
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell1(x, states=state,
                                      training=training)
            
            x, state = self.lstm_cell2(x, states=state,
                                      training=training)
            
#             x, state = self.lstm_cell3(x, states=state,
#                                       training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions
    
    
    
    
def get_training_examples(data, target_feature, test_size=.2, random_state=42):
    """ Create the training and test examples by one-hot-encoding the non-numerical values."""
    
    ds_encoded = pd.get_dummies(data)
    
    if ds_encoded.isnull().values.any():
        raise Exception('There exist Nan values.')
    
    # set the labels to target prediction; Oslo Prices
    labels = np.array(ds_encoded[target_feature])
    # remove the target from dataset
    ds_encoded = ds_encoded.drop([target_feature], axis=1)
    # save a list of columns for future use
    features_list = list(ds_encoded.columns)
    # convert dataset to numpy array to feed into model
    ds_encoded = np.array(ds_encoded)

    # split dataset into training and test
    train_dataset, test_dataset, train_labels, test_labels = train_test_split(ds_encoded, labels,
                                                                         test_size=test_size, random_state=random_state)
    
    return train_dataset, test_dataset, train_labels, test_labels, features_list

def model_performance(predictions, test_labels):
    """ return the performance of the model."""
    
    errors = abs(predictions - test_labels)

    # mean absolute percentage error (MAPE)
    mape = 100 * (errors / (test_labels + EPSILON))
    accuracy = 100 - np.mean(mape)

    print("Mean Absolute Error : ", round(np.mean(errors), 2))
    print("Model Accuracy : ", round(accuracy, 2), '%')
    
    
def plot_variable_importances(model, features_list, path=None, title='Feature Importances', fontsize=18, figsize=None):
    """ plot the importance of each variable for Random Forest Regression trees.
    Return: list of variable importances.
    """
    
    importances = list(model.feature_importances_)

    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features_list, importances)]

    feature_importances = sorted(feature_importances, key= lambda x: x[1], reverse=True)

    # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
    
    plt.figure(figsize=(30,8) if figsize is None else figsize)
    # list of locations to plot
    x_values = list(range(len(importances)))

    plt.bar(x_values, importances, orientation='vertical', color='black')

    plt.xticks(x_values, features_list, rotation='vertical')

    plt.ylabel('Importance', fontsize=fontsize)
    plt.xlabel('Variable', fontsize=fontsize)
    plt.title(title)
    plt.tight_layout()
    if path:
        plt.savefig(path)
    
    plt.show()
    return feature_importances
    
def plot_decision_tree(model, tree_index, features_list, file_name):
    """ visualize the decision trees of the Random Forest Regressor."""
    
    if not os.path.exists('visualization'):
        os.mkdir('visualization')
        
    # an instance tree from model
    tree = model.estimators_[5]

    export_graphviz(tree,
                    out_file=os.path.join('visualization', file_name + '.dot'), 
                    feature_names = features_list,
                    rounded = True, 
                    proportion = False, 
                    precision = 2, 
                    filled = True)

    # using the system sub-process to utilize graphviz
    call(['dot', '-Tpng', os.path.join('visualization', file_name + '.dot'), 
                                       '-o', os.path.join('visualization', file_name + '.png'), '-Gdpi=600'])
    
    
def save_model(model, path):
    """ save scikit-learn model."""
    pickle.dump(model, open(path + '.sav', 'wb'))
    print('Model saved!')
    
def load_model(path):
    """ load scikit-learn model."""
    return pickle.load(open(path, 'rb'))

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """ transform a time series dataset into a supervised learning dataset."""
    
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = pd.concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values

def get_train_test_split(data, n_test):
    """ Split dataset into train and test examples.
    n_test: test example size
    """
    return data[:-n_test, :], data[-n_test:, :]

def model_forecast(model, train, testX):
    """ train the model"""
    
    # transform list into array
    train = np.asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict(np.asarray([testX]))
    return yhat[0]

def walk_forward_validation(model, data, n_test, verbose=0):
    """ Timeseries forecast training process. 
    model: xgboost, random forest, etc.
    data: input timeseries.
    n_test: sliding window size for walk forward validation.
    """
    
    predictions = list()
    # split dataset
    train, test = get_train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in tqdm(range(len(test))):
        # split test row into input and output columns
        testX, testy = test[i, :-1], test[i, -1]
        # fit model on history and make a prediction
        yhat = model_forecast(model, history, testX)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
        if not verbose == 0:
            # summarize progress
            print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
    # estimate prediction error
    error = mean_absolute_error(test[:, -1], predictions)
    return error, test, predictions

def make_predictions(model, data, timesteps=1, plot=False, model_name=None):
    """ Timeseries prediction function for Random forest, XGBoost and possibly 
    other machine learning models."""
    predictions = []
    for i in tqdm(range(len(data))):
        # split test row into input and output columns
        testX, testy = data[i, :-1], data[i, -1]
        # fit model on history and make a prediction
        yhat = model.predict(np.asarray([testX]))
        # store forecast in list of predictions
        predictions.append(yhat[0])
    mae = mean_absolute_error(data[:, -1], predictions)
    print('MAE: %.3f' % mae)
    if plot:
        plt.figure(figsize=(15, 8))
        plt.scatter(data[:, -1], predictions)
        plt.title('{} Predictions'.format(model_name))
        if model_name:
            plt.savefig(os.path.join('visualization', model_name + '_scatter.png'))
        plt.show()
    
    return mae

def R_squared(y_true, y_pred):
    """ Custom keras R-Squared metric function.
    Src: 
    https://stackoverflow.com/questions/42351184/how-to-calculate-r2-in-tensorflow 
    https://keras.io/api/metrics/
    """
    residual = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
    return tf.subtract(1.0, tf.divide(residual, total))
## TO DO ##
# class RandomForest():