#region Using declarations
using System;
using System.Collections.Generic;
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
    public class PropFirmBreakoutStrategy : Strategy
    {
        private const string LongSignalName = "LongBreakout";
        private const string ShortSignalName = "ShortBreakout";
        private const string HardCloseLongSignalName = "HardCloseLong";
        private const string HardCloseShortSignalName = "HardCloseShort";

        private ATR atr;
        private SMA intradaySma;
        private MAX donchianHigh;
        private MIN donchianLow;
        private TimeZoneInfo easternTimeZone;
        private TimeZoneInfo platformTimeZone;

        private readonly Queue<double> completedSessionCloses = new Queue<double>();
        private double completedSessionCloseSum;
        private double currentDailyRegimeBias;
        private double sessionStartCumProfit;
        private DateTime activeSessionDate;
        private DateTime lastResearchSessionDate;
        private double lastResearchSessionClose;
        private int dailyTradeCount;

        [NinjaScriptProperty]
        [Range(1, int.MaxValue)]
        [Display(Name = "DonchianLookback", GroupName = "Parameters", Order = 0)]
        public int DonchianLookback { get; set; }

        [NinjaScriptProperty]
        [Range(0.1, double.MaxValue)]
        [Display(Name = "StopLossATR", GroupName = "Parameters", Order = 1)]
        public double StopLossATR { get; set; }

        [NinjaScriptProperty]
        [Range(0.1, double.MaxValue)]
        [Display(Name = "TargetATR", GroupName = "Parameters", Order = 2)]
        public double TargetATR { get; set; }

        [NinjaScriptProperty]
        [Range(0, 235959)]
        [Display(Name = "StartTime", GroupName = "Parameters", Order = 3)]
        public int StartTime { get; set; }

        [NinjaScriptProperty]
        [Range(0, 235959)]
        [Display(Name = "EndTime", GroupName = "Parameters", Order = 4)]
        public int EndTime { get; set; }

        [NinjaScriptProperty]
        [Range(0.0, double.MaxValue)]
        [Display(Name = "DailyLossLimitUSD", GroupName = "Risk", Order = 0)]
        public double DailyLossLimitUSD { get; set; }

        [NinjaScriptProperty]
        [Range(0.0, double.MaxValue)]
        [Display(Name = "DailyTargetUSD", GroupName = "Risk", Order = 1)]
        public double DailyTargetUSD { get; set; }

        [NinjaScriptProperty]
        [Range(1, int.MaxValue)]
        [Display(Name = "MaxTradesPerDay", GroupName = "Risk", Order = 2)]
        public int MaxTradesPerDay { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "UseDailyRegimeFilter", GroupName = "Filters", Order = 0)]
        public bool UseDailyRegimeFilter { get; set; }

        [NinjaScriptProperty]
        [Range(-1, 6)]
        [Display(Name = "BlockedWeekday", GroupName = "Filters", Order = 1)]
        public int BlockedWeekday { get; set; }

        [NinjaScriptProperty]
        [Range(1, int.MaxValue)]
        [Display(Name = "Contracts", GroupName = "Execution", Order = 0)]
        public int Contracts { get; set; }

        [NinjaScriptProperty]
        [Range(1, int.MaxValue)]
        [Display(Name = "SmaPeriod", GroupName = "Indicators", Order = 0)]
        public int SmaPeriod { get; set; }

        [NinjaScriptProperty]
        [Range(1, int.MaxValue)]
        [Display(Name = "AtrPeriod", GroupName = "Indicators", Order = 1)]
        public int AtrPeriod { get; set; }

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Python-parallel Donchian breakout with intraday SMA gate, prior-session daily regime filter, one trade per session, and session-template-driven close.";
                Name = "PropFirmBreakoutStrategy";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = false;
                ExitOnSessionCloseSeconds = 0;
                IsInstantiatedOnEachOptimizationIteration = true;
                BarsRequiredToTrade = 60;
                DefaultQuantity = 1;
                SetOrderQuantity = SetOrderQuantity.DefaultQuantity;
                Slippage = 1;

                DonchianLookback = 5;
                StopLossATR = 1.5;
                TargetATR = 10.0;
                StartTime = 90000;
                EndTime = 110000;
                DailyLossLimitUSD = 750.0;
                DailyTargetUSD = 600.0;
                MaxTradesPerDay = 1;
                UseDailyRegimeFilter = true;
                BlockedWeekday = -1;
                Contracts = 1;
                SmaPeriod = 50;
                AtrPeriod = 14;
            }
            else if (State == State.Configure)
            {
                DefaultQuantity = Contracts;
            }
            else if (State == State.DataLoaded)
            {
                atr = ATR(AtrPeriod);
                intradaySma = SMA(Close, SmaPeriod);
                donchianHigh = MAX(High, DonchianLookback);
                donchianLow = MIN(Low, DonchianLookback);
                easternTimeZone = TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time");
                platformTimeZone = Core.Globals.GeneralOptions.TimeZoneInfo ?? TimeZoneInfo.Local;

                ResetSessionState();
            }
        }

        protected override void OnBarUpdate()
        {
            if (BarsInProgress != 0)
                return;

            DateTime barTimeEt = ToEasternTime(Time[0]);
            bool insideResearchSession = IsInsideResearchSession(barTimeEt);

            if (insideResearchSession && activeSessionDate != barTimeEt.Date)
                StartNewSession(barTimeEt.Date);

            if (CurrentBar < Math.Max(BarsRequiredToTrade, Math.Max(SmaPeriod, DonchianLookback) + 1))
            {
                if (insideResearchSession)
                    UpdateResearchSessionClose(barTimeEt.Date, Close[0]);
                return;
            }

            if (!insideResearchSession)
                return;

            if (Position.MarketPosition != MarketPosition.Flat && IsLastResearchBar(barTimeEt))
            {
                SubmitHardClose();
                UpdateResearchSessionClose(barTimeEt.Date, Close[0]);
                return;
            }

            if (!IndicatorsReady())
            {
                UpdateResearchSessionClose(barTimeEt.Date, Close[0]);
                return;
            }

            if (Position.MarketPosition != MarketPosition.Flat)
            {
                UpdateResearchSessionClose(barTimeEt.Date, Close[0]);
                return;
            }

            if (!IsTradingDayAllowed(barTimeEt.DayOfWeek))
            {
                UpdateResearchSessionClose(barTimeEt.Date, Close[0]);
                return;
            }

            if (dailyTradeCount >= MaxTradesPerDay)
            {
                UpdateResearchSessionClose(barTimeEt.Date, Close[0]);
                return;
            }

            double sessionPnl = GetSessionRealizedPnl();
            if (DailyLossLimitUSD > 0.0 && sessionPnl <= -DailyLossLimitUSD)
            {
                UpdateResearchSessionClose(barTimeEt.Date, Close[0]);
                return;
            }

            if (DailyTargetUSD > 0.0 && sessionPnl >= DailyTargetUSD)
            {
                UpdateResearchSessionClose(barTimeEt.Date, Close[0]);
                return;
            }

            int currentTime = ToTime(barTimeEt);
            if (currentTime < StartTime || currentTime > EndTime)
            {
                UpdateResearchSessionClose(barTimeEt.Date, Close[0]);
                return;
            }

            double priorChannelHigh = donchianHigh[1];
            double priorChannelLow = donchianLow[1];
            double smaValue = intradaySma[0];
            double currentAtr = atr[0];

            if (double.IsNaN(priorChannelHigh) || double.IsNaN(priorChannelLow) || double.IsNaN(smaValue) || double.IsNaN(currentAtr))
            {
                UpdateResearchSessionClose(barTimeEt.Date, Close[0]);
                return;
            }

            bool longSignal = Close[0] > priorChannelHigh && Close[0] > smaValue;
            bool shortSignal = Close[0] < priorChannelLow && Close[0] < smaValue;

            if (UseDailyRegimeFilter)
            {
                if (longSignal && currentDailyRegimeBias <= 0.0)
                    longSignal = false;

                if (shortSignal && currentDailyRegimeBias >= 0.0)
                    shortSignal = false;
            }

            if (!longSignal && !shortSignal)
            {
                UpdateResearchSessionClose(barTimeEt.Date, Close[0]);
                return;
            }

            double stopTicks = Math.Max(1.0, currentAtr * StopLossATR / TickSize);
            double targetTicks = Math.Max(1.0, currentAtr * TargetATR / TickSize);

            if (longSignal)
            {
                SetStopLoss(LongSignalName, CalculationMode.Ticks, stopTicks, false);
                SetProfitTarget(LongSignalName, CalculationMode.Ticks, targetTicks);
                EnterLong(Contracts, LongSignalName);
            }
            else if (shortSignal)
            {
                SetStopLoss(ShortSignalName, CalculationMode.Ticks, stopTicks, false);
                SetProfitTarget(ShortSignalName, CalculationMode.Ticks, targetTicks);
                EnterShort(Contracts, ShortSignalName);
            }

            UpdateResearchSessionClose(barTimeEt.Date, Close[0]);
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

            if (execution.Order.OrderState != OrderState.Filled)
                return;

            string orderName = execution.Order.Name;
            if (orderName == LongSignalName || orderName == ShortSignalName)
                dailyTradeCount += 1;
        }

        private void StartNewSession(DateTime sessionDate)
        {
            if (lastResearchSessionDate != DateTime.MinValue && lastResearchSessionDate != sessionDate)
                RegisterCompletedSessionClose(lastResearchSessionClose);

            activeSessionDate = sessionDate;
            sessionStartCumProfit = SystemPerformance.AllTrades.TradesPerformance.Currency.CumProfit;
            dailyTradeCount = 0;
            currentDailyRegimeBias = ComputeCurrentDailyRegimeBias();
        }

        private void SubmitHardClose()
        {
            if (Position.MarketPosition == MarketPosition.Long)
                ExitLong(HardCloseLongSignalName, LongSignalName);
            else if (Position.MarketPosition == MarketPosition.Short)
                ExitShort(HardCloseShortSignalName, ShortSignalName);
        }

        private void RegisterCompletedSessionClose(double sessionClose)
        {
            completedSessionCloses.Enqueue(sessionClose);
            completedSessionCloseSum += sessionClose;

            while (completedSessionCloses.Count > SmaPeriod)
                completedSessionCloseSum -= completedSessionCloses.Dequeue();
        }

        private double ComputeCurrentDailyRegimeBias()
        {
            if (completedSessionCloses.Count < SmaPeriod)
                return 0.0;

            double previousSessionClose = 0.0;
            foreach (double value in completedSessionCloses)
                previousSessionClose = value;

            double dailySmaValue = completedSessionCloseSum / completedSessionCloses.Count;
            if (previousSessionClose > dailySmaValue)
                return 1.0;

            if (previousSessionClose < dailySmaValue)
                return -1.0;

            return 0.0;
        }

        private double GetSessionRealizedPnl()
        {
            return SystemPerformance.AllTrades.TradesPerformance.Currency.CumProfit - sessionStartCumProfit;
        }

        private bool IndicatorsReady()
        {
            return !double.IsNaN(atr[0])
                && !double.IsNaN(intradaySma[0])
                && !double.IsNaN(donchianHigh[1])
                && !double.IsNaN(donchianLow[1]);
        }

        private bool IsTradingDayAllowed(DayOfWeek dayOfWeek)
        {
            if (BlockedWeekday >= 0 && (int) dayOfWeek == BlockedWeekday)
                return false;

            return dayOfWeek != DayOfWeek.Saturday && dayOfWeek != DayOfWeek.Sunday;
        }

        private DateTime ToEasternTime(DateTime barTime)
        {
            DateTime unspecified = DateTime.SpecifyKind(barTime, DateTimeKind.Unspecified);
            return TimeZoneInfo.ConvertTime(unspecified, platformTimeZone, easternTimeZone);
        }

        private bool IsInsideResearchSession(DateTime barTimeEt)
        {
            int timeEt = ToTime(barTimeEt);
            return timeEt >= 80000 && timeEt <= 155900;
        }

        private bool IsLastResearchBar(DateTime barTimeEt)
        {
            if (!IsInsideResearchSession(barTimeEt))
                return false;

            int timeframeMinutes = BarsPeriod.BarsPeriodType == BarsPeriodType.Minute ? BarsPeriod.Value : 0;
            if (timeframeMinutes <= 0)
                return false;

            TimeSpan barStart = barTimeEt.TimeOfDay;
            TimeSpan nextBarStart = barStart.Add(TimeSpan.FromMinutes(timeframeMinutes));
            return nextBarStart >= new TimeSpan(16, 0, 0);
        }

        private void UpdateResearchSessionClose(DateTime sessionDate, double sessionClose)
        {
            lastResearchSessionDate = sessionDate;
            lastResearchSessionClose = sessionClose;
        }

        private void ResetSessionState()
        {
            completedSessionCloses.Clear();
            completedSessionCloseSum = 0.0;
            currentDailyRegimeBias = 0.0;
            sessionStartCumProfit = 0.0;
            activeSessionDate = DateTime.MinValue;
            lastResearchSessionDate = DateTime.MinValue;
            lastResearchSessionClose = 0.0;
            dailyTradeCount = 0;
        }
    }
}
