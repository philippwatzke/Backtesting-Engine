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
    public class VwapLunchFadeMCL : Strategy
    {
        private const string LongSignalName = "LunchFadeLong";
        private const string StopExitSignalName = "AtrStop";
        private const string TargetExitSignalName = "VwapTarget";
        private const string HardCloseSignalName = "HardClose";

        private ATR atr;
        private MACD macd;
        private OrderFlowVWAP vwap;
        private TimeZoneInfo easternTimeZone;

        private double pendingStopDistance;
        private double activeStopPrice;
        private bool hasPendingEntry;
        private bool hardCloseSubmitted;

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
        [Display(Name = "Entry Start Time ET", GroupName = "Schedule", Order = 0)]
        public int EntryStartTimeEt { get; set; }

        [NinjaScriptProperty]
        [Range(0, 235959)]
        [Display(Name = "Entry End Time ET", GroupName = "Schedule", Order = 1)]
        public int EntryEndTimeEt { get; set; }

        [NinjaScriptProperty]
        [Range(0, 235959)]
        [Display(Name = "Hard Close Time ET", GroupName = "Schedule", Order = 2)]
        public int HardCloseTimeEt { get; set; }

        [NinjaScriptProperty]
        [Range(1, int.MaxValue)]
        [Display(Name = "Contracts", GroupName = "Execution", Order = 0)]
        public int Contracts { get; set; }

        [NinjaScriptProperty]
        [Range(0.0, double.MaxValue)]
        [Display(Name = "Simulated Slippage (Ticks)", GroupName = "Execution", Order = 1)]
        public double SimulatedSlippageTicks { get; set; }

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Long-only MCL VWAP lunch fade: Monday/Tuesday/Thursday, 12:00-12:59 ET entries, VWAP mean-reversion target, ATR stop, and 14:29 ET hard close.";
                Name = "VwapLunchFadeMCL";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = false;
                ExitOnSessionCloseSeconds = 30;
                IsInstantiatedOnEachOptimizationIteration = true;
                SetOrderQuantity = SetOrderQuantity.DefaultQuantity;
                BarsRequiredToTrade = 60;

                AtrPeriod = 14;
                AtrStopMultiplier = 1.5;
                MacdFast = 24;
                MacdSlow = 52;
                MacdSignal = 18;
                VwapStdDevMultiplier = 2.5;

                EntryStartTimeEt = 120000;
                EntryEndTimeEt = 125959;
                HardCloseTimeEt = 142900;
                Contracts = 1;
                SimulatedSlippageTicks = 1.22;
                Slippage = 1;
            }
            else if (State == State.Configure)
            {
                DefaultQuantity = Contracts;
                // NinjaTrader's strategy slippage setting is integer ticks only.
                Slippage = Math.Max(0, (int)Math.Round(SimulatedSlippageTicks, MidpointRounding.AwayFromZero));
            }
            else if (State == State.DataLoaded)
            {
                atr = ATR(AtrPeriod);
                macd = MACD(MacdFast, MacdSlow, MacdSignal);
                vwap = OrderFlowVWAP(VWAPResolution.Standard, Bars.TradingHours, VWAPStandardDeviations.Three, VwapStdDevMultiplier, 3.0, 4.0);

                AddChartIndicator(atr);
                AddChartIndicator(macd);
                AddChartIndicator(vwap);

                easternTimeZone = TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time");
                ResetTradeState();
            }
        }

        protected override void OnBarUpdate()
        {
            if (BarsInProgress != 0)
                return;

            if (CurrentBar < BarsRequiredToTrade)
                return;

            if (Bars.IsFirstBarOfSession)
                ResetTradeState();

            DateTime barTimeEt = ToEasternTime(Time[0]);
            int currentTimeEt = ToTime(barTimeEt);

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

            double lowerBandPrev = vwap.StdDev1Lower[1];
            double lowerBandNow = vwap.StdDev1Lower[0];
            double macdHistogramNow = macd.Default[0] - macd.Avg[0];
            double macdHistogramPrev = macd.Default[1] - macd.Avg[1];

            bool longSetup =
                Close[1] < lowerBandPrev &&
                Close[0] > lowerBandNow &&
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

            if (execution.Order.Name == LongSignalName && execution.Order.OrderState == OrderState.Filled)
            {
                hasPendingEntry = false;
                activeStopPrice = execution.Order.AverageFillPrice - pendingStopDistance;
            }

            if (Position.MarketPosition == MarketPosition.Flat)
                ResetTradeState();
        }

        private void UpdateLongExits()
        {
            if (activeStopPrice <= 0.0)
                activeStopPrice = Position.AveragePrice - (atr[0] * AtrStopMultiplier);

            double vwapPrice = vwap.VWAP[0];
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
                && !double.IsNaN(vwap.VWAP[0])
                && !double.IsNaN(vwap.StdDev1Lower[0])
                && !double.IsNaN(vwap.StdDev1Lower[1]);
        }

        private bool IsTradingDayAllowed(DayOfWeek dayOfWeek)
        {
            return dayOfWeek == DayOfWeek.Monday
                || dayOfWeek == DayOfWeek.Tuesday
                || dayOfWeek == DayOfWeek.Thursday;
        }

        private DateTime ToEasternTime(DateTime barTime)
        {
            DateTime unspecified = DateTime.SpecifyKind(barTime, DateTimeKind.Unspecified);
            return TimeZoneInfo.ConvertTime(unspecified, Bars.TradingHours.TimeZoneInfo, easternTimeZone);
        }

        private void ResetTradeState()
        {
            pendingStopDistance = 0.0;
            activeStopPrice = 0.0;
            hasPendingEntry = false;
            hardCloseSubmitted = false;
        }
    }
}
