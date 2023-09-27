import tools.matlib
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tools.matlib
from sklearn.preprocessing import MinMaxScaler

def maria_time_sequence_predict():
    plt.rcParams['figure.figsize'] = (12, 6)
    pd.options.display.float_format = '{:,.2f}'.format
    np.set_printoptions(precision=2)
    warnings.filterwarnings("ignore")  # specify to ignore warning messages

    # 加载时间序列数据
    data = pd.read_csv('./res/data.csv', parse_dates=['date'], index_col='date')
    data.plot(y='value', subplots=True,
              figsize=(15, 8), fontsize=16)#指定y轴选定的数据
    plt.xlabel('timestamp', fontsize=16)
    plt.ylabel('value', fontsize=16)
    plt.show()

    # 现在区分开训练集和验证集
    train_start_dt = '1991-07-01'
    test_start_dt = '2005-12-01'
    data[(data.index < test_start_dt) & (data.index >= train_start_dt)][['value']].rename(
        columns={'value': 'train'}) \
        .join(data[test_start_dt:][['value']].rename(columns={'value': 'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=16)
    plt.xlabel('timestamp', fontsize=16)
    plt.ylabel('value', fontsize=16)
    plt.show()

    #训练数据集和验证集进行抽取
    train = data.copy()[(data.index >= train_start_dt) & (data.index < test_start_dt)][['value']]
    test = data.copy()[data.index >= test_start_dt][['value']]
    train_copy=train.copy()
    # 归一化处理
    scaler = MinMaxScaler()
    train['value'] = scaler.fit_transform(train)
    # print(train.head(10))

    data[(data.index >= train_start_dt) & (data.index < test_start_dt)][['value']].rename(
        columns={'value': 'original'}).plot.hist(bins=100, fontsize=12)
    train.rename(columns={'value': 'scaled value'}).plot.hist(bins=100, fontsize=12)
    plt.show()

    '''
    tools/matlib/adf_val函数
    模型平稳性检验：
    ‘adf’对比三个置信区间越小越好
    ‘pvalue’ MacKinnon基于MacKinnon的近似p值，pvalue约接近于0就代表序列越稳定
    '''

    # 0阶差分预测效果
    ts_data_0 = data['value'].astype('float32')
    print(type(ts_data_0))
    tools.matlib.adf_val(ts_data_0, 'raw time series')

    # 一阶差分预测效果
    #ts_data_1=np.diff(ts_data_0)
    ts_data_1=ts_data_0.diff().dropna()
    tools.matlib.adf_val(ts_data_1, 'raw time series')

    #二阶差分预测效果
    ts_data_2 = nts_data_1=ts_data_1.diff().dropna()
    tools.matlib.adf_val(ts_data_2, 'raw time series')

    #  纯随机检验
    tools.matlib.acorr_val(ts_data_2)
    # 从运算结果来看，不难发现二阶微分效果为最佳，

    fig = plt.figure(figsize=(20, 16))
    ax1 = fig.add_subplot(211)

    ts_data_1[100:].plot(ax=ax1)
    ax2 = fig.add_subplot(212)

    ts_data_2[100:].plot(ax=ax2)

    tools.matlib.autocorrelation(ts_data_2,15)


    '''
    现在通过平稳性分析和纯随机检验，我们以及得到了一个平稳的时间序列，接下来我们所需要的就是选择合适的ARIMA模型，即合适的p,d,q参数
    '''
    tools.matlib.decomposing(ts_data_2)
    #通过拆分时间序列可以更加明显的看出

    #fig = plt.figure(figsize=(20, 16))
    tools.matlib.MARIA_predict(train_copy,8,2,6,25).plot()
    plt.show()

    pass


if __name__=="__main__":
    maria_time_sequence_predict()












