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

    # factors SD
    hsi_return_sd = daily_returns_df['HSI_return'].std() * sqrt(252)
    sse_composite_return_sd = daily_returns_df['SSE_Comp_return'].std() * sqrt(252)
    xau_return_sd = daily_returns_df['XAU_return'].std() * sqrt(252)

    factors_correlations = daily_returns_df.drop(['Date', 'HSBC_return', 'Zijin_Mining_return'], axis=1).corr()

    hsbc_model = sm.OLS(daily_returns_df['HSBC_return'], daily_returns_df[['HSI_return', 'SSE_Comp_return', 'XAU_return']]).fit()
    hsbc_hsi_exposure = hsbc_model.params['HSI_return']
    hsbc_sse_exposure = hsbc_model.params['SSE_Comp_return']
    hsbc_xau_exposure = hsbc_model.params['XAU_return']
    hsbc_specific_risk = hsbc_model.resid.std() * sqrt(252)

    print('Factor model for HSBC')
    print('HSBC Exposure on HSI = {0:.4f}, SSE Comp = {1:.4f}, XAU = {2:.4f}'.format(hsbc_hsi_exposure, hsbc_sse_exposure, hsbc_xau_exposure))
    print('HSBC Specific Risk {0:.4f}%'.format(hsbc_specific_risk*100))
    print()

    zijin_model = sm.OLS(daily_returns_df['Zijin_Mining_return'], daily_returns_df[['HSI_return', 'SSE_Comp_return', 'XAU_return']]).fit()
    zijin_hsi_exposure = zijin_model.params['HSI_return']
    zijin_sse_exposure = zijin_model.params['SSE_Comp_return']
    zijin_xau_exposure = zijin_model.params['XAU_return']
    zijin_specific_risk = zijin_model.resid.std() * sqrt(252)

    print('Factor model for Zijin Mining')
    print('Zijin Mining Exposure on HSI = {0:.4f}, SSE Comp = {1:.4f}, XAU = {2:.4f}'.format(zijin_hsi_exposure, zijin_sse_exposure, zijin_xau_exposure))
    print('Zijin Mining Specific Risk {0:.4f}%'.format(zijin_specific_risk*100))
    print()
    # print(zijin_model.summary())

    # HSBC is stock 1, Zijin Mining is stock 2.
    hsbc_weight = 0.7
    zijin_mining_weight = 0.3
    print('Portfolio weights: HSBC = {0:.4f}%, Zijin Mining = {1:.4f}%'.format(hsbc_weight, zijin_mining_weight))
    print()

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

    
    print('Portfolio Total Risk {0:.4f}%'.format(portfolio_total_risk[0, 0] * 100))

    print()

    common_factors_risk_contrib = portfolio_exposure.transpose() @ factors_covariance @ portfolio_exposure / portfolio_total_risk
    print('\tRisk Contributed by Common Factors {0:.4f}%'.format(common_factors_risk_contrib[0, 0] * 100))

    # calculate the marginal for common factors (f_mc)
    common_factors_marginal_contrib = factors_covariance @ portfolio_exposure / portfolio_total_risk

    # correlation of factor marginal contribution (rho)
    correlation_factor_marginal = common_factors_marginal_contrib / factors_stdev

    common_factors_risk_decomposition = portfolio_exposure * factors_stdev * correlation_factor_marginal
    print('\t\tContributed by HSI Factor {0:.4f}%'.format(common_factors_risk_decomposition[0, 0] * 100))
    print('\t\tContributed by SSE Composite Factor {0:.4f}%'.format(common_factors_risk_decomposition[1, 0] * 100))
    print('\t\tContributed by XAU Factor {0:.4f}%'.format(common_factors_risk_decomposition[2, 0] * 100))

    print()

    specific_risk_contrib = portfolio_weight.transpose() @ stocks_specific_covariance @ portfolio_weight / portfolio_total_risk
    print('\tRisk Contributed by Specific Risk {0:.4f}%'.format(specific_risk_contrib[0, 0] * 100))

    # calculate the marignal for specific risk (s_mc)
    specific_risk_marginal_contrib = stocks_specific_covariance @ portfolio_weight / portfolio_total_risk

    correlation_specific_risk_marginal = specific_risk_marginal_contrib / stocks_specific_risk

    specific_risk_decomposition = portfolio_weight * stocks_specific_risk * correlation_specific_risk_marginal
    print('\t\tContributed by HSBC position {0:.4f}%'.format(specific_risk_decomposition[0, 0] * 100))
    print('\t\tContributed by Zijin Mining position {0:.4f}%'.format(specific_risk_decomposition[1, 0] * 100))
