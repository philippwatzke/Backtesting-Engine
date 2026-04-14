#region Using declarations
using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.Strategies;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class VwapLunchFadeMCL_Custom : Strategy
    {
        private const string LongSignalName = "LunchFadeLong";
        private const string StopExitSignalName = "AtrStop";
        private const string TargetExitSignalName = "VwapTarget";
        private const string HardCloseSignalName = "HardClose";

        private ATR atr;
        private MACD macd;
        private Series<double> manualVwapSeries;
        private Series<double> manualStdDevSeries;
        private Series<double> lowerBandSeries;
        private TimeZoneInfo easternTimeZone;
        private TimeZoneInfo platformTimeZone;

        private double sumVolume;
        private double sumPriceVolume;
        private double sumPriceSquaredVolume;
        private double pendingStopDistance;
        private double activeStopPrice;
        private bool hasPendingEntry;
        private bool hardCloseSubmitted;
        private DateTime currentVwapSessionDate;

        [NinjaScriptProperty]
        [Range(1, int.MaxValue)]
        [Display(Name = "ATR Period", GroupName = "Indicators", Order = 0)]
        public int AtrPeriod { get; set; }

        [NinjaScriptProperty]
        [Range(0.1, double.MaxValue)]
        [Display(Name = "ATR Stop Multiplier", GroupName = "Indicators", Order = 1)]
        public double AtrStopMultiplier { get; set; }

        [NinjaScriptProperty]
        [Range(1, int.MaxValue)]
        [Display(Name = "MACD Fast", GroupName = "Indicators", Order = 2)]
        public int MacdFast { get; set; }

        [NinjaScriptProperty]
        [Range(2, int.MaxValue)]
        [Display(Name = "MACD Slow", GroupName = "Indicators", Order = 3)]
        public int MacdSlow { get; set; }

        [NinjaScriptProperty]
        [Range(1, int.MaxValue)]
        [Display(Name = "MACD Signal", GroupName = "Indicators", Order = 4)]
        public int MacdSignal { get; set; }

        [NinjaScriptProperty]
        [Range(0.1, double.MaxValue)]
        [Display(Name = "VWAP StdDev Multiplier", GroupName = "Indicators", Order = 5)]
        public double VwapStdDevMultiplier { get; set; }

        [NinjaScriptProperty]
        [Range(0, 235959)]
        [Display(Name = "VWAP Reset Time ET", GroupName = "Schedule", Order = 0)]
        public int VwapResetTimeEt { get; set; }

        [NinjaScriptProperty]
        [Range(0, 235959)]
        [Display(Name = "Entry Start Time ET", GroupName = "Schedule", Order = 1)]
        public int EntryStartTimeEt { get; set; }

        [NinjaScriptProperty]
        [Range(0, 235959)]
        [Display(Name = "Entry End Time ET", GroupName = "Schedule", Order = 2)]
        public int EntryEndTimeEt { get; set; }

        [NinjaScriptProperty]
        [Range(0, 235959)]
        [Display(Name = "Hard Close Time ET", GroupName = "Schedule", Order = 3)]
        public int HardCloseTimeEt { get; set; }

        [NinjaScriptProperty]
        [Range(1, int.MaxValue)]
        [Display(Name = "Contracts", GroupName = "Execution", Order = 0)]
        public int Contracts { get; set; }

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Manual-VWAP long-only MCL lunch fade without Order Flow + dependency.";
                Name = "VwapLunchFadeMCL_Custom";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = false;
                ExitOnSessionCloseSeconds = 30;
                IsInstantiatedOnEachOptimizationIteration = true;
                SetOrderQuantity = SetOrderQuantity.DefaultQuantity;

                AtrPeriod = 14;
                AtrStopMultiplier = 1.5;
                MacdFast = 24;
                MacdSlow = 52;
                MacdSignal = 18;
                VwapStdDevMultiplier = 2.5;

                VwapResetTimeEt = 180000;
                EntryStartTimeEt = 120000;
                EntryEndTimeEt = 125959;
                HardCloseTimeEt = 142900;
                Contracts = 1;
                BarsRequiredToTrade = 60;
            }
            else if (State == State.Configure)
            {
                DefaultQuantity = Contracts;
            }
            else if (State == State.DataLoaded)
            {
                atr = ATR(AtrPeriod);
                macd = MACD(MacdFast, MacdSlow, MacdSignal);
                manualVwapSeries = new Series<double>(this);
                manualStdDevSeries = new Series<double>(this);
                lowerBandSeries = new Series<double>(this);

                AddChartIndicator(atr);
                AddChartIndicator(macd);

                easternTimeZone = TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time");
                platformTimeZone = Core.Globals.GeneralOptions.TimeZoneInfo ?? TimeZoneInfo.Local;
                ResetTradeState();
                ResetVwapAccumulator();
            }
        }

        protected override void OnBarUpdate()
        {
            if (BarsInProgress != 0)
                return;

            if (CurrentBar < BarsRequiredToTrade)
                return;

            DateTime barTimeEt = ToEasternTime(Time[0]);
            int currentTimeEt = ToTime(barTimeEt);

            UpdateManualVwap(barTimeEt);

            if (Position.MarketPosition == MarketPosition.Flat)
            {
                hardCloseSubmitted = false;
                if (hasPendingEntry && currentTimeEt > EntryEndTimeEt)
                    hasPendingEntry = false;
            }

            if (currentTimeEt >= HardCloseTimeEt)
            {
                SubmitHardClose();
                return;
            }

            if (!IsTradingDayAllowed(barTimeEt.DayOfWeek))
                return;

            if (!IndicatorsReady())
                return;

            if (Position.MarketPosition == MarketPosition.Long)
            {
                UpdateLongExits();
                return;
            }

            if (hasPendingEntry)
                return;

            if (currentTimeEt < EntryStartTimeEt || currentTimeEt > EntryEndTimeEt)
                return;

            double macdHistogramNow = macd.Default[0] - macd.Avg[0];
            double macdHistogramPrev = macd.Default[1] - macd.Avg[1];

            bool longSetup =
                Close[1] < lowerBandSeries[1] &&
                CrossAbove(Close, lowerBandSeries, 1) &&
                macdHistogramNow > macdHistogramPrev;

            if (!longSetup)
                return;

            pendingStopDistance = atr[0] * AtrStopMultiplier;
            activeStopPrice = 0.0;
            hasPendingEntry = true;
            EnterLong(LongSignalName);
        }

        protected override void OnExecutionUpdate(
            Execution execution,
            string executionId,
            double price,
            int quantity,
            MarketPosition marketPosition,
            string orderId,
            DateTime time)
        {
            if (execution == null || execution.Order == null)
                return;

            if (hasPendingEntry && marketPosition == MarketPosition.Long)
            {
                hasPendingEntry = false;
                activeStopPrice = execution.Order.AverageFillPrice - pendingStopDistance;
            }

            if (Position.MarketPosition == MarketPosition.Flat)
                ResetTradeState();
        }

        private void UpdateManualVwap(DateTime barTimeEt)
        {
            DateTime sessionDate = GetVwapSessionDate(barTimeEt);
            if (currentVwapSessionDate != sessionDate)
            {
                ResetVwapAccumulator();
                currentVwapSessionDate = sessionDate;
            }

            double volume = Math.Max(Volume[0], 0.0);
            double closePrice = Close[0];

            sumVolume += volume;
            sumPriceVolume += closePrice * volume;
            sumPriceSquaredVolume += closePrice * closePrice * volume;

            double vwapValue = sumVolume > 0.0 ? sumPriceVolume / sumVolume : closePrice;
            double variance = sumVolume > 0.0 ? (sumPriceSquaredVolume / sumVolume) - (vwapValue * vwapValue) : 0.0;
            if (variance < 0.0)
                variance = 0.0;

            double stdDev = Math.Sqrt(variance);
            manualVwapSeries[0] = vwapValue;
            manualStdDevSeries[0] = stdDev;
            lowerBandSeries[0] = vwapValue - (VwapStdDevMultiplier * stdDev);
        }

        private void UpdateLongExits()
        {
            if (activeStopPrice <= 0.0)
                activeStopPrice = Position.AveragePrice - (atr[0] * AtrStopMultiplier);

            double vwapPrice = manualVwapSeries[0];
            if (double.IsNaN(vwapPrice) || vwapPrice <= 0.0)
                return;

            ExitLongStopMarket(0, true, Position.Quantity, activeStopPrice, StopExitSignalName, LongSignalName);
            ExitLongLimit(0, true, Position.Quantity, vwapPrice, TargetExitSignalName, LongSignalName);
        }

        private void SubmitHardClose()
        {
            if (Position.MarketPosition != MarketPosition.Long || hardCloseSubmitted)
                return;

            hardCloseSubmitted = true;
            ExitLong(HardCloseSignalName, LongSignalName);
        }

        private bool IndicatorsReady()
        {
            return !double.IsNaN(atr[0])
                && !double.IsNaN(macd.Default[0])
                && !double.IsNaN(macd.Avg[0])
                && !double.IsNaN(manualVwapSeries[0])
                && !double.IsNaN(lowerBandSeries[0])
                && !double.IsNaN(lowerBandSeries[1]);
        }

        private bool IsTradingDayAllowed(DayOfWeek dayOfWeek)
        {
            return dayOfWeek == DayOfWeek.Monday
                || dayOfWeek == DayOfWeek.Tuesday
                || dayOfWeek == DayOfWeek.Thursday;
        }

        private DateTime GetVwapSessionDate(DateTime barTimeEt)
        {
            return ToTime(barTimeEt) >= VwapResetTimeEt
                ? barTimeEt.Date.AddDays(1)
                : barTimeEt.Date;
        }

        private DateTime ToEasternTime(DateTime barTime)
        {
            DateTime unspecified = DateTime.SpecifyKind(barTime, DateTimeKind.Unspecified);
            return TimeZoneInfo.ConvertTime(unspecified, platformTimeZone, easternTimeZone);
        }

        private void ResetTradeState()
        {
            pendingStopDistance = 0.0;
            activeStopPrice = 0.0;
            hasPendingEntry = false;
            hardCloseSubmitted = false;
        }

        private void ResetVwapAccumulator()
        {
            sumVolume = 0.0;
            sumPriceVolume = 0.0;
            sumPriceSquaredVolume = 0.0;
        }
    }
}
