#!./env python
#-*-coding:utf-8 -*-


"""
Forecast Price based on filled features by MLP

"""

import numpy as np
import pandas as pd
import sys
from os import system
from keras import regularizers
from keras import initializers, optimizers
import matplotlib.pyplot as plt


# fix random state
# from tensorflow import set_random_seed
np.random.seed(7)


# --------------- Functions ---------------
def overview(dataframe):
    print "\n--- ", locals().keys()[0]
    print pd.DataFrame({'dtype': dataframe.dtypes, 'count': dataframe.count()})


def read(file_name):
    print "\n >>> Reading data..."
    original_data = pd.read_pickle(file_name)
    # original_data.index = pd.to_datetime(original_data.index)
    print '\n', original_data.shape
    return original_data


def train_pred_split(data, label_name=['Price'], pred_ind=[-1]):
    # Split Predset and trainSet
    # Predset: last several data with no labels, later combined with predicted y to predict next y label
    # Trainset: data with labels
    print "\n >>> Splitting train and pred set"
    # Since nas are all same, only need one label to locate null
    predset_ind = np.argwhere(data[label_name[0]].isnull().values).T[0]
    assert (predset_ind == pred_ind).all()
    print '    Last %d Data have no y labels, set as predSet' % len(pred_ind)
    print '    PredSet Date: from ', data.index[pred_ind[0]].date(), ' to ', data.index[pred_ind[-1]].date(), '\n'

    trainset_ind = np.argwhere(data[label_name[0]].notnull().values).T[0]
    train_ind = range(pred_ind[0])
    assert (trainset_ind == train_ind).all()
    print '    %d Data have y labels, feed into training' % len(train_ind)
    print '    TrainSet Date: from ', data.index[train_ind[0]].date(), ' to ', data.index[train_ind[-1]].date()
    return pd.concat([data.ix[train_ind], data.ix[pred_ind]], keys=['train', 'pred'], verify_integrity=True)


def scale(data_split, label_name=['Price'], scale_range=(-1, 1)):
    # scale, based on the train set only (data with non-null labels)
    # meanwhile transform the prediction set using this scaler
    print "\n >>> Scaling data..."
    from sklearn.preprocessing import MinMaxScaler

    train_set = data_split.loc['train']
    pred_set = data_split.loc['pred']
    min_max_scaler = MinMaxScaler(feature_range=scale_range)
    min_max_scaler.fit(train_set)
    scaled_train = pd.DataFrame(min_max_scaler.transform(train_set),
                                columns=train_set.columns,
                                index=train_set.index)
    assert pred_set[label_name].isnull().all().all()
    for label in label_name:
        pred_set[label].fillna(train_set[label].mean(), inplace=True)
    scaled_pred = pd.DataFrame(min_max_scaler.transform(pred_set),
                               columns=pred_set.columns,
                               index=pred_set.index)
    pred_set[label_name] = np.nan
    scaled_pred[label_name] = np.nan  # Modify the label null
    return min_max_scaler, pd.concat([scaled_train, scaled_pred], keys=['train', 'pred'], verify_integrity=True)


def reframe(data_scaled, n_lag_month, n_pred_month=1):
    print "\n >>> Transform data to supervise learning..."
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_lag_month, 0, -1):
        all_null_check = data_scaled.shift(i).isnull().all()
        if all_null_check.any():
            raise ValueError('All null labels due to shift: ', all_null_check[all_null_check].index.values)
        cols.append(data_scaled.shift(i).bfill())
        names += [('%s(t-%d)' % (feature_name, i)) for feature_name in data_scaled.columns]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_pred_month):
        cols.append(data_scaled.shift(-i))
        if i == 0:
            names += [('%s(t)' % feature_name) for feature_name in data_scaled.columns]
        else:
            names += [('%s(t+%d)' % (feature_name, i)) for feature_name in data_scaled.columns]
    # put it all together
    reframed_data = pd.concat(cols, axis=1)
    reframed_data.columns = names
    reframed_data.index = data_scaled.index

    # Sanity check
    assert reframed_data.shape == (len(data_scaled), (n_lag_month + n_pred_month) * len(data_scaled.columns))  # check shape
    assert (reframed_data.loc['train'].index == data_scaled.loc['train'].index).all()  # check if the same train set
    assert (reframed_data.loc['pred'].index == data_scaled.loc['pred'].index).all()  # check if the same pred set
    return reframed_data


def prepare(data_train, n_lag_month, n_features, label_name_reframe=['Price(t)'], test_on=True, test_date=None,
            silent=False):
    # prepare feature and label
    if not silent:
        print "\n >>> Prepare data ..."
    label_len = len(label_name_reframe)
    if test_on:
        # copy to prevent changing the original dataframe
        train_set = data_train[data_train.index < test_date].copy()
        test_set = data_train[data_train.index >= test_date].copy()
        y_test = test_set[label_name_reframe].values
        # x_test = test_set.drop(label_name_reframe, axis=1).values
        # Add an all -1 feature to enable reshaping --> Wrong!! Notice the column position of label!
        # x_test = np.concatenate((x_test, np.array([-1] * len(x_test) * label_len).reshape(len(x_test), label_len)),
        #                         axis=1)
        test_set.loc[:, label_name_reframe] = -1  # Simply set the label column as -1
        x_test = test_set.values
        x_test = x_test.reshape((x_test.shape[0], n_lag_month + 1, n_features))
    else:
        train_set = data_train.copy()

    y_train = train_set[label_name_reframe].values
    # x_train = train_set.drop(label_name_reframe, axis=1).values
    # x_train = np.concatenate((x_train, np.array([-1] * len(x_train) * label_len).reshape(len(x_train), label_len)),
    #                          axis=1)
    train_set.loc[:, label_name_reframe] = -1
    x_train = train_set.values
    x_train = x_train.reshape((x_train.shape[0], n_lag_month + 1, n_features))
    if test_on:
        return x_train, y_train, x_test, y_test
    else:
        return x_train, y_train


def model_build(model, x_train, y_train, epochs=100, batch_size=10,
                # label_name=['Price'],
                test_on=True, x_test=None, y_test=None):
    # design network
    print "\n >>> Building model..."

    model.compile(loss='mae', optimizer='adam')
    if batch_size == 'full':
        batch_size = len(x_train)
    if test_on:
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                            verbose=1, shuffle=False, validation_data=(x_test, y_test))
    else:
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                            verbose=1, shuffle=False)
    return model, history


def get_reframe_label_name(label_name, time_step='t'):
    assert type(label_name) == list
    return map(lambda s: ('%s('+time_step+')') % s, label_name)


def train(data_orig, label_name=['Price'], pred_ind=[-1], n_lag_month=4 * 12,
          scale_range=(-1, 1), model=None,
          test_on=True, test_date=None,
          epochs=100, batch_size=10):
    # >>> scale, reframe, train
    data_split = train_pred_split(data_orig, label_name=label_name, pred_ind=pred_ind)
    scaler, data_scaled = scale(data_split, label_name=label_name, scale_range=scale_range)
    data_reframed = reframe(data_scaled, n_lag_month)

    # --------------- train ---------------
    n_features = len(data_scaled.columns)
    if test_on:
        x_train, y_train, x_test, y_test = prepare(data_reframed.loc['train'], n_lag_month, n_features,
                                                   label_name_reframe=get_reframe_label_name(label_name),
                                                   test_on=test_on, test_date=test_date)
        model_fitted, history = model_build(model, x_train, y_train,
                                            epochs=epochs, batch_size=batch_size,
                                            # label_name=label_name,
                                            test_on=test_on, x_test=x_test, y_test=y_test)
        return model_fitted, history, scaler, (data_reframed, data_scaled, data_split), (n_lag_month, n_features), \
               (x_train, y_train, x_test, y_test)

    elif ~test_on:
        x_train, y_train = prepare(data_reframed.loc['train'], n_lag_month, n_features,
                                   label_name_reframe=get_reframe_label_name(label_name),
                                   test_on=test_on)
        model_fitted, history = model_build(model, x_train, y_train,
                                            epochs=epochs, batch_size=batch_size,
                                            # label_name=label_name,
                                            test_on=test_on)
        return model_fitted, history, scaler, (data_reframed, data_scaled, data_split), (n_lag_month, n_features)


def inverse(scaler, x_scaled, y_scaled, y_fit,
            split_index, split_columns, reframe_columns,
            label_name=['Price'],
            is_pred_set=True,  # If predSet
            check_inverse=True, split_set=None,  # If check the inversion necessarily
            check_scale=True, scaled_set=None):  # If check the same x_scale data after reframing
    print '\n >>> Inverse transforming...'
    # predicted y inversion
    len_label = len(label_name)
    len_features = len(split_columns)
    df_inv = pd.DataFrame(index=split_index, columns=split_columns)
    # match x_data with reframe feature names
    # share index with split
    df_x_scaled = pd.DataFrame(x_scaled, index=split_index, columns=reframe_columns)
    # feature_name = df_inv.columns.difference(label_name)  # index.difference will sort automatically, ignore
    feature_name = [col for col in df_inv.columns if col not in label_name]

    # Inverse fitted data
    df_inv[label_name] = y_fit
    # Use timestep t in reframed data as previous scaled data
    reframe_feature_name = get_reframe_label_name(feature_name, time_step='t')
    x_scaled_last = df_x_scaled[reframe_feature_name].values
    # x_scaled_last = x_scaled[:, -len_features:][:, :-len_label]  # ??
    df_inv[feature_name] = x_scaled_last
    df_inv[df_inv.columns] = scaler.inverse_transform(df_inv)
    y_fit_inv = df_inv[label_name].values
    x_inv1 = df_inv[feature_name].values

    # Inverse original data to check
    df_inv[label_name] = y_scaled
    df_inv[feature_name] = x_scaled_last
    if is_pred_set:
        for label in label_name:
            df_inv[label].fillna(0, inplace=True)  # Fill null in the label of pred set
    df_inv[df_inv.columns] = scaler.inverse_transform(df_inv)
    if is_pred_set:
        df_inv[label_name] = np.nan
    y_expect_inv = df_inv[label_name].values
    x_inv2 = df_inv[feature_name].values

    # Check the consistency of reframe
    if check_scale:
        x_scaled_orig = scaled_set.drop(label_name, axis=1).values
        assert (x_scaled_last == x_scaled_orig).all()

    # Check the inversion
    if check_inverse:
        x_orig = split_set.drop(label_name, axis=1).values
        y_orig = split_set[label_name].values

        roundoff_err = 1.0e-6
        # Inverse x should equal to original x
        assert (abs(x_inv1 - x_inv2) < roundoff_err).all()
        assert (abs(x_inv1 - x_orig) < roundoff_err).all()
        if is_pred_set:
            # For pred set, y_test_orig should be nan
            assert (np.isnan(y_orig)).all()
        else:
            # Inverse y should equal to original y
            assert ((y_expect_inv - y_orig) < roundoff_err).all()

    df_predicted = pd.DataFrame(y_fit_inv, columns=label_name, index=split_index)
    df_expected = pd.DataFrame(y_orig, columns=label_name, index=split_index)
    df = pd.concat([df_predicted, df_expected], keys=['Predicted', 'Expected'], axis=1)
    # df = pd.DataFrame({'Predicted': y_fit_inv, 'Expected': y_orig}, index=dates)

    return df


# If we prepare predSet at the beginning of training together with train and test,
# This step can be removed
def predict_pred_set(model, reframed_pred, n_lag_month, n_features, label_name=['Price']):
    # PredSet - ladder
    len_pred = len(reframed_pred)
    len_label = len(label_name)
    x_pred_agg = np.zeros((len_pred, n_lag_month+1, n_features))
    y_pred_agg = np.zeros((len_pred, len_label))
    y_fit_pred_agg = np.zeros_like(y_pred_agg)
    print '    Fill Step: ',
    for i in range(len_pred):
        x_pred, y_pred = prepare(reframed_pred[i:i+1], n_lag_month, n_features,
                                 label_name_reframe=get_reframe_label_name(label_name),
                                 test_on=False, silent=True)
        sys.stdout.write(str(i+1)+' ')
        sys.stdout.flush()
        y_fit_pred = model.predict(x_pred)
        x_pred_agg[i, :] = x_pred
        y_pred_agg[i, :] = y_pred
        y_fit_pred_agg[i, :] = y_fit_pred
        for j, dt in zip(range(i+1, min(len_pred, i+1+n_lag_month)), range(1, min(len_pred-i, n_lag_month+1))):
            reframed_pred.ix[j, get_reframe_label_name(label_name, time_step=('t-%i' % dt))] = y_fit_pred
    print
    assert np.isnan(y_pred_agg).all()
    return x_pred_agg, y_pred_agg, y_fit_pred_agg


def predict(model, scaler, data_pack, const_pack, inplace=True,
            train_pack=None, test_date=None, label_name=['Price']):
    # unpack
    reframed_data, scaled_data, split_data = data_pack
    n_lag_month, n_features = const_pack
    # Scaled data is used for inversion
    # Original data is used for inversion verification
    # Reframed data is used for predset preparation

    # PredSet
    print '\n >>> Predicting Pred Set...'
    reframed_pred = reframed_data.loc['pred']
    # --> Prediction
    x_pred, y_pred, y_fit_pred = predict_pred_set(model, reframed_pred, n_lag_month, n_features, label_name=label_name)
    # --> Inversion
    x_pred = x_pred.reshape((x_pred.shape[0], (n_lag_month+1)*n_features))
    split_pred = split_data.loc['pred']
    scaled_pred = scaled_data.loc['pred']
    df_pred = inverse(scaler, x_pred, y_pred, y_fit_pred,
                      split_pred.index, split_pred.columns, reframed_pred.columns,
                      label_name=label_name,
                      is_pred_set=True,
                      check_inverse=True, split_set=split_pred,
                      check_scale=True, scaled_set=scaled_pred)
    if inplace:
        split_data.loc['pred'][label_name] = df_pred['Predicted']
        return split_data

    # TrainSet
    print '\n >>> Predicting Train Set...'
    split_train_set = split_data.loc['train']
    scaled_train_set = scaled_data.loc['train']
    x_train, y_train, x_test, y_test = train_pack

    # --> Test prediction
    y_fit_test = model.predict(x_test)
    # --> Inversion
    x_test = x_test.reshape((x_test.shape[0], (n_lag_month+1)*n_features))
    test_ind = split_train_set.index >= test_date
    split_test = split_train_set[test_ind]
    scaled_test = scaled_train_set[test_ind]
    df_test = inverse(scaler, x_test, y_test, y_fit_test,
                      split_test.index, split_test.columns, reframed_pred.columns,
                      label_name=label_name,
                      is_pred_set=False,
                      check_inverse=True, split_set=split_test,
                      check_scale=True, scaled_set=scaled_test)

    # --> Train prediction
    y_fit_train = model.predict(x_train)
    # --> Inversion
    x_train = x_train.reshape((x_train.shape[0], (n_lag_month+1)*n_features))
    train_ind = split_train_set.index < test_date
    split_train = split_train_set[train_ind]
    scaled_train = scaled_train_set[train_ind]
    df_train = inverse(scaler, x_train, y_train, y_fit_train,
                       split_train.index, split_train.columns, reframed_pred.columns,
                       label_name=label_name,
                       is_pred_set=False,
                       check_inverse=True, split_set=split_train,
                       check_scale=True, scaled_set=scaled_train)

    # concatenate
    data_fit_expect = pd.concat([df_train, df_test, df_pred], keys=['train', 'test', 'pred'], verify_integrity=True)
    return data_fit_expect


def find_null_feature_one_off(df):
    # Null list in length descending order
    nan_list_list = []
    nan_len_list = []
    null_feature_check = df.isnull().any()
    null_features = null_feature_check[null_feature_check].index.values
    for feature_name in null_features:
        nan_list = list(np.argwhere(df[feature_name].isnull().values).T[0])
        nan_list_list.append(nan_list)
        nan_len_list.append(len(nan_list))
    zip_list_sorted = sorted(zip(nan_len_list, nan_list_list, null_features), key=lambda x: x[0], reverse=True)
    nan_len_list, nan_list_list, nan_name_list = zip(*zip_list_sorted)
    target_len_list = [nan_len_list[0]]
    target_list_list = [nan_list_list[0]]
    target_name_list = [[nan_name_list[0]]]
    for i in range(1, len(nan_len_list)):
        if nan_len_list[i] == nan_len_list[i-1]:
            target_name_list[-1].append(nan_name_list[i])
        else:
            target_len_list.append(nan_len_list[i])
            target_list_list.append(nan_list_list[i])
            target_name_list.append([nan_name_list[i]])

    # Sanity check
    assert len(target_len_list) == len(target_list_list)
    assert len(target_len_list) == len(target_name_list)
    assert all(map(lambda ll: ll[-1] == len(df)-1, target_list_list))
    assert all(map(lambda ll: ll[-1]-ll[0]+1 == len(ll), target_list_list))
    assert all(map(lambda tpl: tpl[0] == len(tpl[1]), zip(target_len_list, target_list_list)))

    # Diff and rejoin
    diff_len_list = map(lambda x, y: x-y, target_len_list[:-1], target_len_list[1:])
    diff_len_list.append(target_len_list[-1])
    diff_list_list = map(lambda ll, sl: [ind for ind in ll if ind not in sl],
                         target_list_list[:-1], target_list_list[1:])
    diff_list_list.append(target_list_list[-1])
    assert all(map(lambda ln, l: ln == len(l), diff_len_list, diff_list_list))
    for i in range(1, len(target_name_list))[::-1]:
        for j in range(i)[::-1]:
            target_name_list[i].extend(target_name_list[j])
    diff_name_list = target_name_list

    return diff_len_list, diff_list_list, diff_name_list


# def fill():
#     # --------------- Read data ---------------
#     data_orig = read()
#     overview(data_orig)
#
#     # ----------------- fill na in features -----------------
#     data_feature = data_orig.drop('Price', axis=1)
#     n_lag_month = 4*12
#     if data_feature.isnull().any().any():
#         print '\n >>> Filling feature nulls...'
#         null_len_lists, null_list_lists, null_name_lists = find_null_feature_one_off(data_feature)
#         # n_rounds = len(null_len_lists)
#         # epochs_list = map(lambda x: (1000-400)/n_rounds*x + 400, range(n_rounds))
#         for add_id, (null_list_list, null_name_list) in enumerate(zip(null_list_lists, null_name_lists)):
#             print '\n ------------------- round %i -------------------' % (add_id+1)
#             print '\n >>> label to be filled: '
#             print '    ', null_name_list
#             data_to_null = data_feature[:null_list_list[-1]+1]
#             data_rest = data_feature[null_list_list[-1]+1:]
#             print '\n >>> --------------- Training..'
#             model, scaler, data_pack, const_pack = train(data_to_null,
#                                                          label_name=null_name_list,
#                                                          pred_ind=null_list_list,
#                                                          n_lag_month=n_lag_month,
#                                                          test_on=False,
#                                                          epochs=800,  #epochs_list[add_id],
#                                                          batch_size=10)
#             print '\n >>> --------------- Filling..'
#             data_fill = predict(model, scaler, data_pack, const_pack, label_name=null_name_list)
#             # Get rid of multi-index
#             data_fill = pd.concat([data_fill.loc['train'], data_fill.loc['pred']], axis=0)
#             data_feature = pd.concat([data_fill, data_rest], axis=0)
#     data_all = pd.concat([data_feature, data_orig['Price']], axis=1)
#     # -- Store filled feature data
#     data_all.to_pickle('Macro-Data-Fill.pkl')
#     return data_all


def forecast(data_all, label_list, fit_param):
    # ----------------- train and predict -----------------
    print '\n ---------------- train and predict -----------------'
    null_len, null_list, null_name = find_null_feature_one_off(data_all)
    assert null_name[0] == label_list
    print '\n >>> label to be predicted: ', null_name[0]
    print '\n >>> Prediction set:', data_all.index[null_list[0][0]].date(), 'to', \
                                    data_all.index[null_list[0][-1]].date()
    train_ind = np.argwhere(data_all.index < fit_param['test_date']).squeeze()
    assert train_ind[-1] < null_list[0][0] - 1
    print "\n >>> test Date: ", fit_param['test_date'].date()
    print '    test Set Dates: ', data_all.index[train_ind[-1]+1].date(), 'to', \
                                  data_all.index[null_list[0][0]-1].date()

    print '\n >>> --------------- Training...'
    model, history, scaler, data_pack, const_pack, train_pack = train(data_all,
                                                                      pred_ind=null_list[0],
                                                                      n_lag_month=fit_param['n_lag_month'],
                                                                      epochs=fit_param['epochs'],
                                                                      batch_size=fit_param['batch_size'],
                                                                      test_on=fit_param['test'],
                                                                      test_date=fit_param['test_date'],
                                                                      scale_range=fit_param['scale_range'],
                                                                      model=fit_param['model'])
    print '\n >>> --------------- Predicting...'
    # Predict and compare
    data_fit_expect = predict(model, scaler, data_pack, const_pack,
                              inplace=False,
                              train_pack=train_pack,
                              test_date=fit_param['test_date'])
    return data_fit_expect, history, train_pack, model
    # data_fit_expect.to_pickle('Macro-LSTM-DF.pkl')


def explain(model, x_train, x_test,
            feature_names, label_name=['Price'], categorical_feature_names=['PurchaseRestriction'],
            explain_ind=-1, explain_feature_num=5):
    print '\n >>> Explain prediction -----> '
    import lime.lime_tabular
    categorical_feature = [feature_names.index(x) for x in categorical_feature_names]
    explainer = lime.lime_tabular.RecurrentTabularExplainer(x_train,
                                                            feature_names=feature_names,
                                                            class_names=label_name,
                                                            categorical_features=categorical_feature,
                                                            verbose=True, mode='regression')
    lime.lime_tabular.LimeTabularExplainer(mode='regression')
    exp = explainer.explain_instance(x_test[explain_ind], model.predict, num_features=explain_feature_num)
    exp.as_pyplot_figure()
    # exp.show_in_notebook()


def plot_predict_expect(data, test_on=True, set0=False):
    data_train = data.loc['train']
    if test_on:
        data_test = data.loc['test']
        mean_date = max(data_train.index) + (min(data_test.index) - max(data_train.index)) / 2
    data_pred = data.loc['pred']

    print "\n >>> Plot prediction and expectation -----> "
    import matplotlib.lines as mlines

    fig = plt.figure(figsize=(10, 3))
    label_lists = list(data_train['Predicted'])
    len_sub = len(label_lists)
    for i, label in enumerate(label_lists):
        ax = fig.add_subplot(len_sub, 1, i+1)
        # ax = plt.gca()
        expected, = ax.plot(data_train.index, data_train['Expected'][label], 'k.-')
        predicted, = ax.plot(data_train.index, data_train['Predicted'][label], 'b.-')
        ax.plot(data_pred.index, data_pred['Predicted'][label], 'r.-')
        if set0:
            ax.plot(ax.get_xlim(), [0, 0], 'g--')
        if test_on:
            ax.plot(data_test.index, data_test['Expected'][label], 'k.-')
            ax.plot(data_test.index, data_test['Predicted'][label], 'b.-')
            ax.add_line(mlines.Line2D([mean_date, mean_date], list(ax.get_ylim()), color='r', linestyle='--'))
        # ax.set_ylabel('Median Price')
        ax.set_title(label)
        if i < len_sub-1:
            ax.set_xticks([])
        ax.legend((expected, predicted), ('Expected', 'Predicted'))
    plt.savefig('Macro-Train-DF.png', dpi=300)
    # plt.show()


def plot_moving_corr(data_orig, window=10, test_on=True):
    assert test_on
    data_train = data_orig.loc['train']
    data_test = data_orig.loc['test']
    len_test = len(data_test)
    mean_date = max(data_train.index) + (min(data_test.index) - max(data_train.index)) / 2
    data = data_orig.drop(index='pred')
    data_orig.index = data_orig.index.droplevel()
    data.index = data.index.droplevel()
    # data = pd.concat([data_train, data_test], axis=1)  # join train and test
    label_lists = list(data['Predicted'])
    window = window
    try:
        assert window < len_test
    except AssertionError:
        raise ValueError('Window size should be smaller than test size!')
    moving_corr = []
    moving_index = []
    for i in range(len(data)-window+1):
        index_list = data['Predicted'][label_lists][i:i+window].index
        moving_index.append(index_list[(len(index_list)+1)/2])
        corr_mat = np.corrcoef(data['Predicted'][label_lists][i:i+window].values.squeeze(),
                               data['Expected'][label_lists][i:i+window].values.squeeze())
        assert corr_mat.shape == (2, 2)
        moving_corr.append(corr_mat[0, 1])
    moving_corr = np.array(moving_corr)
    moving_index = np.array(moving_index)

    import matplotlib.lines as mlines
    fig, (ax0, ax) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    expected, = ax0.plot(data_orig.index, data_orig['Expected'][label_lists], 'k.-')
    predicted, = ax0.plot(data_orig.index, data_orig['Predicted'][label_lists], 'b.-')
    ax0.legend((expected, predicted), ('Expected', 'Predicted'))
    ax0.add_line(mlines.Line2D([mean_date, mean_date], list(ax0.get_ylim()), color='r', linestyle='--'))
    ax0.set_title(label_lists[0])

    ax.plot(moving_index, moving_corr, 'b.-')
    ax.set_ylim(-1, 1)
    ax.add_line(mlines.Line2D([mean_date, mean_date], list(ax.get_ylim()), color='r', linestyle='--'))
    ax.add_line(mlines.Line2D(list(ax.get_xlim()), [0, 0], color='k', linestyle='--'))
    ax.set_title('Corr')
    plt.savefig('Macro-Train-Mov Corr.png', dpi=300)
    # plt.show()


def plot_history(history):
    # plot history
    print '\n >>> Plot history ------> '
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig('Macro-Train-Loss.png', dpi=300)
    # plt.show()

if __name__ == '__main__':

    import time
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, Dropout

    fileName = 'Macro-Data-Fill.pkl'
    labelList = ['Price']
    nLag = 48  # Months # 4 years lookback

    # --------------- Read data ---------------
    dataOrig = read(fileName)
    # Explicitly drop some features
    feature_drop = ['HouseSoldPrice', 'DealHousePrice',  # high correlation
                    'HouseholdNumber', 'PrimarySchoolNumber']  # Sample Stats, Abnormal Value
    dataOrig.drop(feature_drop, axis=1, inplace=True)
    featureNames = list(dataOrig)

    # Model
    Model = Sequential()
    Model.add(LSTM(23, return_sequences=True, input_shape=(nLag+1, len(featureNames))))
    # model.add(LSTM(22, input_shape=(nLag+1, len(list(data)))))
    Model.add(LSTM(47))
    Model.add(Dropout(0.1))
    # model.add(LSTM(16))
    # model.add(Dense(64, activation='sigmoid'))
    # model.add(Dense(32, activation='sigmoid'))
    Model.add(Dense(len(labelList)))

    fitParam = {'n_lag_month': nLag,
                'epochs': 400,
                'batch_size': 'full',
                'scale_range': (0, 1),
                'test': True,
                'test_date': pd.to_datetime('20160601'),
                'model': Model}

    t0 = time.time()
    data_fit, fit_history, trainPack, Model = forecast(dataOrig, labelList, fitParam)
    print '\n  time: %.2f secs' % (time.time() - t0)

    plot_history(fit_history)
    plot_predict_expect(data_fit)
    plot_moving_corr(data_fit)

    # xTrain, yTrain, xTest, yTest = trainPack
    # explain(Model, xTrain, xTest, feature_names=featureNames, label_name=labelList,
    #         categorical_feature_names=['PurchaseRestriction'],
    #         explain_ind=-1, explain_feature_num=5)


