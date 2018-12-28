import QuantLib as ql

def price_once(asof_date, spot_price, implied_vol, risk_free_rate):
    # Pricing TPX Index Call Expire on 2019-1-11, Strike at 1400
    ql.Settings.instance().evaluationDate = asof_date
    maturity_date = ql.Date(11, 1, 2019)
    strike_price = 1400
    day_count = ql.Actual365Fixed()
    calendar = ql.Japan()
    ql.Settings.instance().evaluationDate = asof_date
    european_option = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, strike_price),  ql.EuropeanExercise(maturity_date))
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
    rate_ts = ql.YieldTermStructureHandle(ql.FlatForward(asof_date, risk_free_rate, day_count))
    vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(asof_date, calendar, implied_vol, day_count))

    process = ql.BlackScholesProcess(spot_handle, rate_ts, vol_ts)

    pricing_engine = ql.AnalyticEuropeanEngine(process)
    european_option.setPricingEngine(pricing_engine)
    return european_option.NPV(), european_option.delta(), european_option.vega(), european_option.rho(), european_option.gamma(),  european_option.thetaPerDay()

# day zero at close

spot0 = 1517.16
rate0 = -0.00118
date0 = ql.Date(20, 12, 2018)
# we calibrated the implied vol to match the market value 121 as of 2018-12-20
vol0 = 0.2489938732267518 

fair_value0, delta0, vega0, rho0, gamma0, theta0 = price_once(date0, spot0, vol0, rate0)

# day one at close

spot1 = 1488.19
rate1 = -0.00118
date1 = ql.Date(21, 12, 2018)
# we calibrated the implied vol to match the market value 95.5 as of 2018-12-21
vol1 = 0.25521617690276865

fair_value1, delta1, vega1, rho1, gamma1, theta1 = price_once(date1, spot1, vol1, rate1)

print('Fair Value on Day Zero: {0:.5f}'.format(fair_value0))
print('Fair Value on Day One: {0:.5f}'.format(fair_value1))

pnl = fair_value1 - fair_value0
print('Total PnL: {0:.5f}'.format(pnl))

delta_pnl = delta0 * (spot1 - spot0)
vega_pnl = vega0 * (vol1- vol0)
rho_pnl = rho0 * (rate1 - rate0)
theta_pnl = theta0 * (date1 - date0)
gamma_pnl = 0.5 * gamma0 * (spot1 - spot0)**2
unexplained_pnl = pnl - delta_pnl - vega_pnl - rho_pnl - theta_pnl - gamma_pnl
print('PnL Attribution')
print('Delta: {0:.5f}, Vega: {1:.5f}, Rho: {2:.5f}, Theta: {3:.5f}, Gamma: {4:.5f}, Unexplained: {5:.5f}'.format(delta_pnl, vega_pnl, rho_pnl, theta_pnl, gamma_pnl, unexplained_pnl))



