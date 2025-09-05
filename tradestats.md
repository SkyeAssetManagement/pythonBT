The key stats to evaluate walk forward returns are listed here with there calculation logic.

As a guiding principal the nature of the strategies are that there can be only one trade per calendar day, so the presence of a return on a daily observation indicates there was a trade on that day.

Returns calculations assume that the return values are in correct magnitude, so a 1% return is 0.01.  If this is not the case, lets say a 1% return is represented as 1, then correct by dividing by 100.

Formatting
Use integer or decimal as appropriate.
Dont show the % sign or $ after a maetric.

Count
# Observations:	Count of the number of # Observations (days) in the out of sample walk forward periods (OOSWF).
Years of Data:  is the number of observations (days) in the OOSWF period divided by 252.
# Trades: is the number of trades signaled in the OOSWF period.
Trade Frequency %:  # Trades / # Observations
Ave Trade P.A.: Trade Frequency % / Years of Data
Ave Trades P.M.:  As above but number of months

Trades
Win %: Number winning trades in OOSFW period / # Trades
Ave Loss %: Ave % Return of losing trades 
Ave Profit %: Ave % Return of winning trades 
Ave PnL %:  Ave % Return of trades (winners and losers)
Expectency:  (Win % * Ave Profit %) - ((1 - Win %) * abs(Ave Loss %))
Best Day %:  Largest positive daily return.
Worst Day %:  Smallest negative daily return.

Model
Ave Annual %^:  (Ending Value / Beginning Value)^(1 / Number of Years) - 1
Max Draw %^: (Peak Value - Current Value) / Peak Value × 100
Sharpe: (Average daily return * 252) / (standard deviation daily returns * sqrt(252)) 	
Profit DD Ratio:  Ave Annual %^ / Max Draw %^
UPI: Ave Annual %^ / √(Mean(((Peak - Value) / Peak)² × 100²)).  Where for each data point: Peak is the highest value up to that point, and Value is the current equity value.

Important Note
^ these calcs are performed using observation sampled from a compound equity curve, Equity[t] = Equity[t-1] × (1 + Return[t]).  Use initial equity of $1000.

Charts (all for OOS period only)
Compound Equity Curve starting at notional value of 1,000.  Daily resolution with x axis being time, not trade count.
Drawdown chart.  Line or area chart showing the cumulative amount over time  equity retreats from it's prior peak.  
A table with monthly returns, one row per year.  Monthly returns calculated as the exponent of the MonthySum(ln(1+simple returns)).  Include annual returns on the chart, using the same calulation logic.
Frequency distribution chart of daily returns with the mean and median marked.
Rolling performance - rolling 50 trade (not 50 days) sharpe ratio and samUPI^^ with RHS and LHS scale for each.
  
^^ samUPI is a version of UPI that is based on trade count, not time as a lookback and normalisez to give similiar range outputs ad various trade count lookbacks.  Here is the calculation, written here in Amibroker AFL code.  arraytocalcon is simply a time series array of returns where any non zero return represents the occurance of a trade.

function UPIcalcLnTrans(UPIlookback,arraytocalcon,UPI1_Sumret0)
{
	sumRtn = 0; 
	maxSumRtn = -1000;
	sumDDsqd = 0;	
	
	trade = arraytocalcon != 0;

	for(i=UPIlookback;i>=1;i--)
	{
		sumRtn = sumRtn + ValueWhen(trade,arraytocalcon,i);
		maxSumRtn = Max(sumRtn,maxSumRtn);
		DD = sumRtn - maxSumRtn;
		sumDDsqd = sumDDsqd + DD^2;
	}

	//meanRtn = sumRtn / UPIlookback;
	ulcer = sqrt(sumDDsqd)+0.00001;
	UPI = sumRtn/Ulcer;
	UPI = UPI * UPIlookback;
	//UPI = UPI * 100; 
	
	if(UPI1_Sumret0)
	{
	outs = UPI;
	}
	else
	{
	outs = sumRtn;
	}
	
	return Nz(outs);
}



