import pandas as pd
import numpy as np
import statsmodels.api as sm
from math import sqrt

class FactorModel:
    def __init__(self, hsbc_hsi_exposure, hsbc_sse_exposure, hsbc_xau_exposure, hsbc_specific_risk, 
            zijin_hsi_exposure, zijin_sse_exposure, zijin_xau_exposure, zijin_specific_risk, 
            hsi_return_sd, sse_composite_return_sd, xau_return_sd, factors_correlations):
        
        self.hsbc_hsi_exposure = hsbc_hsi_exposure
        self.hsbc_sse_exposure = hsbc_sse_exposure
        self.hsbc_xau_exposure = hsbc_xau_exposure
        self.hsbc_specific_risk = hsbc_specific_risk
        self.zijin_hsi_exposure = zijin_hsi_exposure
        self.zijin_sse_exposure = zijin_sse_exposure
        self.zijin_xau_exposure = zijin_xau_exposure
        self.zijin_specific_risk = zijin_specific_risk
        self.hsi_return_sd = hsi_return_sd
        self.sse_composite_return_sd = sse_composite_return_sd
        self.xau_return_sd = xau_return_sd
        self.factors_correlations = factors_correlations


def build_multi_factors_model():
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

    regression_data = daily_returns_df[['HSI_return', 'SSE_Comp_return', 'XAU_return']]
    regression_data = sm.add_constant(regression_data)

    hsbc_model = sm.OLS(daily_returns_df['HSBC_return'], regression_data).fit()
    hsbc_hsi_exposure = hsbc_model.params['HSI_return']
    hsbc_sse_exposure = hsbc_model.params['SSE_Comp_return']
    hsbc_xau_exposure = hsbc_model.params['XAU_return']
    hsbc_specific_risk = hsbc_model.resid.std() * sqrt(252)

    print('=================================')
    print('Factor model for HSBC')
    print('HSBC Alpha = {0:.4f}'.format(hsbc_model.params['const']))
    print('HSBC Exposure on HSI = {0:.4f}, SSE Comp = {1:.4f}, XAU = {2:.4f}'.format(hsbc_hsi_exposure, hsbc_sse_exposure, hsbc_xau_exposure))
    print('HSBC Specific Risk {0:.4f}%'.format(hsbc_specific_risk*100))
    print()

    zijin_model = sm.OLS(daily_returns_df['Zijin_Mining_return'], regression_data).fit()
    zijin_hsi_exposure = zijin_model.params['HSI_return']
    zijin_sse_exposure = zijin_model.params['SSE_Comp_return']
    zijin_xau_exposure = zijin_model.params['XAU_return']
    zijin_specific_risk = zijin_model.resid.std() * sqrt(252)

    print('Factor model for Zijin Mining')
    print('Zijin Mining Alpha = {0:.4f}'.format(zijin_model.params['const']))
    print('Zijin Mining Exposure on HSI = {0:.4f}, SSE Comp = {1:.4f}, XAU = {2:.4f}'.format(zijin_hsi_exposure, zijin_sse_exposure, zijin_xau_exposure))
    print('Zijin Mining Specific Risk {0:.4f}%'.format(zijin_specific_risk*100))
    print()

    print('Common Factors:')
    print('HSI return annualized stdev = {0:.4f}%'.format(hsi_return_sd * 100))
    print('SSE Composite return annalized stdev = {0:.4f}%'.format(sse_composite_return_sd * 100))
    print('XAU return annualized stdev = {0:.4f}%'.format(xau_return_sd * 100))
    print('Factors Correlation Matrix:')
    print(factors_correlations)
    print()
    print('=================================')

    return FactorModel(hsbc_hsi_exposure, hsbc_sse_exposure, hsbc_xau_exposure, hsbc_specific_risk, 
                    zijin_hsi_exposure, zijin_sse_exposure, zijin_xau_exposure, zijin_specific_risk, 
                    hsi_return_sd, sse_composite_return_sd, xau_return_sd, factors_correlations)

def x_sigma_rho_decomposition(hsbc_weight, zijin_mining_weight, model):

    print('********** X Sigma Rho Decomposition **********')
    print()
    print('Portfolio weights: HSBC = {0:.4f}%, Zijin Mining = {1:.4f}%'.format(hsbc_weight * 100, zijin_mining_weight * 100))
    print()

    # portfolio weight vector (w)
    portfolio_weight = np.array([[hsbc_weight],
                                 [zijin_mining_weight]
                                 ])
    # common factor exposure matrix (F)
    factor_exposure = np.array([
        [model.hsbc_hsi_exposure, model.hsbc_sse_exposure, model.hsbc_xau_exposure],
        [model.zijin_hsi_exposure, model.zijin_sse_exposure, model.zijin_xau_exposure]
    ])

    # portfolio exposure (x)
    portfolio_exposure = factor_exposure.transpose() @ portfolio_weight

    # stock specific covariance matrix (Q)
    stocks_specific_risk = np.array([
        [model.hsbc_specific_risk],
        [model.zijin_specific_risk]
    ])
    # assume no correlation so only diagonal variance values.
    stocks_specific_covariance = np.diag(stocks_specific_risk.flatten()) @ np.diag(stocks_specific_risk.flatten())

    factors_stdev = np.array([[model.hsi_return_sd], [model.sse_composite_return_sd], [model.xau_return_sd]])

    # factors covariance matrix (C)
    factors_covariance = np.diag(factors_stdev.flatten()) @ model.factors_correlations.values @ np.diag(factors_stdev.flatten())

    # portfolio total variance (Var(R_p))
    portfolio_total_variance = portfolio_weight.transpose() @ (factor_exposure @ factors_covariance @ factor_exposure.transpose() + stocks_specific_covariance) @ portfolio_weight

    # portfolio total risk (sigma_p)
    portfolio_total_risk = np.sqrt(portfolio_total_variance)
    
    print('Portfolio Total Risk {0:.4f}%'.format(portfolio_total_risk[0, 0] * 100))
    print()

    # common factor contribution (sigma_f)
    common_factors_risk_contrib = portfolio_exposure.transpose() @ factors_covariance @ portfolio_exposure / portfolio_total_risk
    print('\tRisk Contributed by Common Factors {0:.4f}%'.format(common_factors_risk_contrib[0, 0] * 100))

    # calculate the marginal for common factors (c_f)
    common_factors_marginal_contrib = factors_covariance @ portfolio_exposure / portfolio_total_risk

    # correlation of each factor return and total portfolio return (rho(R_i, R_p))
    correlation_factor_marginal = common_factors_marginal_contrib / factors_stdev

    # each individual summation term in sigma_f
    common_factors_risk_decomposition = portfolio_exposure * factors_stdev * correlation_factor_marginal
    print('\t\tContributed by HSI Factor {0:.4f}%'.format(common_factors_risk_decomposition[0, 0] * 100))
    print('\t\tContributed by SSE Composite Factor {0:.4f}%'.format(common_factors_risk_decomposition[1, 0] * 100))
    print('\t\tContributed by XAU Factor {0:.4f}%'.format(common_factors_risk_decomposition[2, 0] * 100))
    print()

    # specific risk contribution (sigma_q)
    specific_risk_contrib = portfolio_weight.transpose() @ stocks_specific_covariance @ portfolio_weight / portfolio_total_risk
    print('\tRisk Contributed by Specific Risk {0:.4f}%'.format(specific_risk_contrib[0, 0] * 100))

    # calculate the marignal for specific risk (c_q)
    specific_risk_marginal_contrib = stocks_specific_covariance @ portfolio_weight / portfolio_total_risk

    # correlation of asset specific return and total portfolio return (rho(epsilon_j, R_p))
    correlation_specific_risk_marginal = specific_risk_marginal_contrib / stocks_specific_risk

    # each individual summation term in sigma_q
    specific_risk_decomposition = portfolio_weight * stocks_specific_risk * correlation_specific_risk_marginal
    print('\t\tContributed by HSBC position {0:.4f}%'.format(specific_risk_decomposition[0, 0] * 100))
    print('\t\tContributed by Zijin Mining position {0:.4f}%'.format(specific_risk_decomposition[1, 0] * 100))

if __name__ == '__main__':

    # build the multi factors model.
    model = build_multi_factors_model()

    # Portfolio weighting on HSBC and Zijin Mining. Adding up to 1.
    hsbc_weight = 0.7
    zijin_mining_weight = 0.3

    x_sigma_rho_decomposition(hsbc_weight, zijin_mining_weight, model)

