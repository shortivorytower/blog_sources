import pandas as pd
import numpy as np
import statsmodels.api as sm
from math import sqrt

if __name__ == '__main__':
    # the two stocks
    hsbc_df = pd.read_csv('0005.HK.csv')
    zijin_mining_df = pd.read_csv('2899.HK.csv')

    # the three factors
    hsi_df = pd.read_csv('HSI.csv')
    sse_composite_df = pd.read_csv('000001.SS.csv')
    xau_df = pd.read_csv('XAU.csv')

    # get the daily return % change
    hsbc_df['return'] = hsbc_df['Close'].pct_change()
    zijin_mining_df['return'] = zijin_mining_df['Close'].pct_change()
    hsi_df['return'] = hsi_df['Close'].pct_change()
    sse_composite_df['return'] = sse_composite_df['Close'].pct_change()
    xau_df['return'] = xau_df['Close'].pct_change()

    # remove first row
    hsbc_df = hsbc_df[1:]
    zijin_mining_df = zijin_mining_df[1:]
    hsi_df = hsi_df[1:]
    sse_composite_df = sse_composite_df[1:]
    xau_df = xau_df[1:]

    daily_returns_df = pd.DataFrame({
        'Date': hsi_df['Date'],
        'HSBC_return': hsbc_df['return'],
        'Zijin_Mining_return': zijin_mining_df['return'],
        'HSI_return': hsi_df['return'],
        'SSE_Comp_return': sse_composite_df['return'],
        'XAU_return': xau_df['return'],
    }, columns=['Date', 'HSBC_return', 'Zijin_Mining_return', 'HSI_return', 'SSE_Comp_return', 'XAU_return'])

    # annualized return SD
    hsbc_return_sd = daily_returns_df['HSBC_return'].std() * sqrt(252)
    zijin_mining_return_sd = daily_returns_df['Zijin_Mining_return'].std() * sqrt(252)

    # factors SD
    hsi_return_sd = daily_returns_df['HSI_return'].std() * sqrt(252)
    sse_composite_return_sd = daily_returns_df['SSE_Comp_return'].std() * sqrt(252)
    xau_return_sd = daily_returns_df['XAU_return'].std() * sqrt(252)

    factors_correlations = daily_returns_df.drop(['Date', 'HSBC_return', 'Zijin_Mining_return'], axis=1).corr()
    # print(factors_correlations.values)

    hsbc_model = sm.OLS(daily_returns_df['HSBC_return'], daily_returns_df[['HSI_return', 'SSE_Comp_return', 'XAU_return']]).fit()
    hsbc_hsi_exposure = hsbc_model.params['HSI_return']
    hsbc_sse_exposure = hsbc_model.params['SSE_Comp_return']
    hsbc_xau_exposure = hsbc_model.params['XAU_return']
    hsbc_specific_risk = hsbc_model.resid.std() * sqrt(252)

    print('HSBC Exposure on HSI = {0}, SSE Comp = {1}, XAU = {2}'.format(hsbc_hsi_exposure, hsbc_sse_exposure, hsbc_xau_exposure))
    print('HSBC Specific Risk', hsbc_specific_risk)
    # print(hsbc_model.summary())

    zijin_model = sm.OLS(daily_returns_df['Zijin_Mining_return'], daily_returns_df[['HSI_return', 'SSE_Comp_return', 'XAU_return']]).fit()
    zijin_hsi_exposure = zijin_model.params['HSI_return']
    zijin_sse_exposure = zijin_model.params['SSE_Comp_return']
    zijin_xau_exposure = zijin_model.params['XAU_return']
    zijin_specific_risk = zijin_model.resid.std() * sqrt(252)

    print('Zijin Mining Exposure on HSI = {0}, SSE Comp = {1}, XAU = {2}'.format(zijin_hsi_exposure, zijin_sse_exposure, zijin_xau_exposure))
    print('Zijin Mining Specific Risk', zijin_specific_risk)
    # print(zijin_model.summary())

    # HSBC is stock 1, Zijin Mining is stock 2.
    hsbc_weight = 0.7
    zijin_mining_weight = 0.3

    # portfolio weight vector (h)
    portfolio_weight = np.array([[hsbc_weight],
                                 [zijin_mining_weight]
                                 ])
    # stock factor exposure matrix (X)
    stocks_exposure = np.array([
        [hsbc_hsi_exposure, hsbc_sse_exposure, hsbc_xau_exposure],
        [zijin_hsi_exposure, zijin_sse_exposure, zijin_xau_exposure]
    ])

    # portfolio exposure (x)
    portfolio_exposure = stocks_exposure.transpose() @ portfolio_weight

    # stock specific covariance matrix (D)
    stocks_specific_risk = np.array([
        [hsbc_specific_risk],
        [zijin_specific_risk]
    ])
    # assume no correlation so only diagonal variance values.
    stocks_specific_covariance = np.diag(stocks_specific_risk.flatten()) @ np.diag(stocks_specific_risk.flatten())

    factors_stdev = np.array([[hsi_return_sd], [sse_composite_return_sd], [xau_return_sd]])

    # factors covariance matrix (F)
    factors_covariance = np.diag(factors_stdev.flatten()) @ factors_correlations.values @ np.diag(factors_stdev.flatten())

    # portfolio total variance
    portfolio_total_variance = portfolio_weight.transpose() @ (stocks_exposure @ factors_covariance @ stocks_exposure.transpose() + stocks_specific_covariance) @ portfolio_weight

    # portfolio total risk (sigma)
    portfolio_total_risk = np.sqrt(portfolio_total_variance)

    print()
    print('Portfolio Total Risk', portfolio_total_risk[0, 0])

    print()

    common_factors_risk_contrib = portfolio_exposure.transpose() @ factors_covariance @ portfolio_exposure / portfolio_total_risk
    print('Risk Contributed by Common Factors', common_factors_risk_contrib[0, 0])

    # calculate the marginal for common factors (f_mc)
    common_factors_marginal_contrib = factors_covariance @ portfolio_exposure / portfolio_total_risk

    # correlation of factor marginal contribution (rho)
    correlation_factor_marginal = common_factors_marginal_contrib / factors_stdev

    common_factors_risk_decomposition = portfolio_exposure * factors_stdev * correlation_factor_marginal
    print('Contributed by HSI Factor', common_factors_risk_decomposition[0, 0])
    print('Contributed by SSE Composite Factor', common_factors_risk_decomposition[1, 0])
    print('Contributed by XAU Factor', common_factors_risk_decomposition[2, 0])

    print()

    specific_risk_contrib = portfolio_weight.transpose() @ stocks_specific_covariance @ portfolio_weight / portfolio_total_risk
    print('Risk Contributed by Specific Risk', specific_risk_contrib[0, 0])

    # calculate the marignal for specific risk (s_mc)
    specific_risk_marginal_contrib = stocks_specific_covariance @ portfolio_weight / portfolio_total_risk

    correlation_specific_risk_marginal = specific_risk_marginal_contrib / stocks_specific_risk

    specific_risk_decomposition = portfolio_weight * stocks_specific_risk * correlation_specific_risk_marginal
    print('Contributed by HSBC position', specific_risk_decomposition[0, 0])
    print('Contributed by Zijin Mining position', specific_risk_decomposition[1, 0])
