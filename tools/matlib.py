
from statsmodels.tsa.arima.model import ARIMA
import statsmodels as sm

import matplotlib.pyplot as plt


from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller  # adf检验库
from statsmodels.stats.diagnostic import acorr_ljungbox  # 随机性检验库



def adf_val(ts, ts_title):
    '''
    ts: 时间序列数据，Series类型
    ts_title: 时间序列图的标题名称，字符串
    '''
    # 稳定性（ADF）检验
    adf, pvalue, usedlag, nobs, critical_values, icbest = adfuller(ts)

    name = ['adf', 'pvalue', 'usedlag',
            'nobs', 'critical_values', 'icbest']
    values = [adf, pvalue, usedlag, nobs,
              critical_values, icbest]
    print(list(zip(name, values)))

    return adf, pvalue, critical_values,
    # 返回adf值、adf的p值、三种状态的检验值


def acorr_val(ts):
    '''
    # 白噪声（随机性）检验
    ts: 时间序列数据，Series类型
    返回白噪声检验的P值，这是一个truple类型的数据，第一行是ljungbox统计量，第二行是统计得到的p值
    一般我们认为：
    如果p<0.05，拒绝原假设，说明原始序列存在相关性
    如果p>=0.05，接收原假设，说明原始序列独立，纯随机
    '''
    noise_check= acorr_ljungbox(ts, lags=1)  # 白噪声检验结果
    return noise_check




'''
对于时间序列自相关和偏自相关系数的分析：
自相关（Autocorrelation）： 对一个时间序列，现在值与其过去值的相关性。如果相关性为正，则说明现有趋势将继续保持。
偏自相关（Partial Autocorrelation）： 可以度量现在值与过去值更纯正的相关性。

拖尾和截尾
拖尾指序列以指数率单调递减或震荡衰减，而截尾指序列从某个时点变得非常小：

'''

def autocorrelation(timeseries, lags):
    '''
    用于对时间序列的自相关和偏自相关系数进行绘制
    :param timeseries: 传入的时间序列
    :param lags: 滞后的阶数
    :return: no return
    '''
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    sm.graphics.tsaplots.plot_acf(timeseries, lags=lags, ax=ax1)
    ax2 = fig.add_subplot(212)
    sm.graphics.tsaplots.plot_pacf(timeseries, lags=lags, ax=ax2)
    plt.show()



def decomposing(timeseries):
    decomposition = seasonal_decompose(timeseries)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.figure(figsize=(16, 12))
    plt.subplot(411)
    plt.plot(timeseries, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonarity')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residual')
    plt.legend(loc='best')
    plt.show()
    autocorrelation(seasonal,10)


# # 假设你已经有了时间序列数据，名为data
# data = pd.Series([10, 20, 30, 40, 50])
#
# # 定义 ARIMA 模型的 pdq 参数
# p = 1  # AR 阶数
# d = 1  # 差分阶数
# q = 1  # MA 阶数
#
# # 创建 ARIMA 模型对象
# model = ARIMA(data, order=(p, d, q))

def MARIA_predict(timeseq,p,d,q,forecast_step):
    # 创建 ARIMA 模型对象
    model = ARIMA(timeseq, order=(p, d, q))
    # 拟合模型
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_step)  # 预测未来 5 个时间步长的值
    # 打印预测结果
    print(forecast)
    return forecast
