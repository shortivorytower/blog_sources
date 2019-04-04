import pandas as pd
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
    hsi_df['return']=hsi_df['Close'].pct_change()
    sse_composite_df['return']=sse_composite_df['Close'].pct_change()
    xau_df['return']=xau_df['Close'].pct_change()

    # remove first row
    hsbc_df=hsbc_df[1:]
    zijin_mining_df=zijin_mining_df[1:]
    hsi_df=hsi_df[1:]
    sse_composite_df=sse_composite_df[1:]
    xau_df = xau_df[1:]


    daily_returns_df = pd.DataFrame({
        'Date' : hsi_df['Date'],
        'HSBC_return' : hsbc_df['return'],
        'Zijin_Mining_return' : zijin_mining_df['return'],
        'HSI_return' : hsi_df['return'],
        'SSE_Comp_return' : sse_composite_df['return'],
        'XAU_return' : xau_df['return'],
    }, columns = ['Date', 'HSBC_return', 'Zijin_Mining_return', 'HSI_return', 'SSE_Comp_return', 'XAU_return'])

    # annualized return SD
    hsbc_return_sd = daily_returns_df['HSBC_return'].std() * sqrt(252)
    zijin_mining_return_sd = daily_returns_df['Zijin_Mining_return'].std() * sqrt(252)

    # factors SD
    hsi_return_sd = daily_returns_df['HSI_return'].std() * sqrt(252)
    sse_composite_return_sd = daily_returns_df['SSE_Comp_return'].std() * sqrt(252)
    xau_return_sd = daily_returns_df['XAU_return'].std() * sqrt(252)

    factors_correlations = daily_returns_df.drop(['Date', 'HSBC_return', 'Zijin_Mining_return'], axis=1).corr()
    #print(factors_correlations)

    hsbc_model = sm.OLS(daily_returns_df['HSBC_return'], daily_returns_df[['HSI_return', 'SSE_Comp_return', 'XAU_return']]).fit()
    hsbc_hsi_exposure = hsbc_model.params['HSI_return']
    hsbc_sse_exposure = hsbc_model.params['SSE_Comp_return']
    hsbc_xau_exposure = hsbc_model.params['XAU_return']
    hsbc_specific_risk = hsbc_model.resid.std()*sqrt(252)

    print('HSBC Exposure on HSI = {0}, SSE Comp = {1}, XAU = {2}'.format(hsbc_hsi_exposure, hsbc_sse_exposure, hsbc_xau_exposure))
    print('HSBC Specific Risk', hsbc_specific_risk)
    #print(hsbc_model.summary())


    zijin_model = sm.OLS(daily_returns_df['Zijin_Mining_return'], daily_returns_df[['HSI_return', 'SSE_Comp_return', 'XAU_return']]).fit()
    zijin_hsi_exposure = zijin_model.params['HSI_return']
    zijin_sse_exposure = zijin_model.params['SSE_Comp_return']
    zijin_xau_exposure = zijin_model.params['XAU_return']
    zijin_specific_risk = zijin_model.resid.std()*sqrt(252)

    print('Zijin Mining Exposure on HSI = {0}, SSE Comp = {1}, XAU = {2}'.format(zijin_hsi_exposure, zijin_sse_exposure, zijin_xau_exposure))
    print('Zijin Mining Specific Risk', zijin_specific_risk)
    #print(zijin_model.summary())


    print('done')