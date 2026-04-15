#region Using declarations
using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Globalization;
using System.IO;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;
#endregion

namespace NinjaTrader.NinjaScript.Indicators
{
    public class DataDumpExporter : Indicator
    {
        private SMA sma50;
        private ATR atr14;
        private MAX donchianHigh5;
        private MIN donchianLow5;
        private StreamWriter writer;
        private readonly CultureInfo invariant = CultureInfo.InvariantCulture;

        [NinjaScriptProperty]
        [Display(Name = "OutputPath", GroupName = "Export", Order = 0)]
        public string OutputPath { get; set; }

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Dump chart bars plus core indicators to CSV on each OnBarClose update.";
                Name = "DataDumpExporter";
                Calculate = Calculate.OnBarClose;
                IsOverlay = false;
                DisplayInDataBox = false;
                DrawOnPricePanel = false;
                PaintPriceMarkers = false;
                IsSuspendedWhileInactive = true;
                OutputPath = @"C:\temp\NT8_Dump.csv";
            }
            else if (State == State.DataLoaded)
            {
                sma50 = SMA(Close, 50);
                atr14 = ATR(14);
                donchianHigh5 = MAX(High, 5);
                donchianLow5 = MIN(Low, 5);

                string directory = Path.GetDirectoryName(OutputPath);
                if (string.IsNullOrWhiteSpace(directory))
                    throw new InvalidOperationException("OutputPath must include a directory.");

                Directory.CreateDirectory(directory);
                writer = new StreamWriter(OutputPath, false);
                writer.AutoFlush = true;
                writer.WriteLine("Timestamp_UTC,Open,High,Low,Close,Volume,SMA_50,ATR_14_Wilder,DonchianHigh_5,DonchianLow_5");
            }
            else if (State == State.Terminated)
            {
                DisposeWriter();
            }
        }

        protected override void OnBarUpdate()
        {
            if (BarsInProgress != 0 || writer == null)
                return;

            string timestampUtc = Time[0].ToUniversalTime().ToString("yyyy-MM-dd HH:mm:ss", invariant);
            string openText = Open[0].ToString("G17", invariant);
            string highText = High[0].ToString("G17", invariant);
            string lowText = Low[0].ToString("G17", invariant);
            string closeText = Close[0].ToString("G17", invariant);
            string volumeText = Volume[0].ToString(invariant);

            string smaText = CurrentBar >= 49 ? sma50[0].ToString("G17", invariant) : string.Empty;
            string atrText = CurrentBar >= 13 ? atr14[0].ToString("G17", invariant) : string.Empty;
            string donchianHighText = CurrentBar >= 5 ? donchianHigh5[1].ToString("G17", invariant) : string.Empty;
            string donchianLowText = CurrentBar >= 5 ? donchianLow5[1].ToString("G17", invariant) : string.Empty;

            writer.WriteLine(string.Join(
                ",",
                timestampUtc,
                openText,
                highText,
                lowText,
                closeText,
                volumeText,
                smaText,
                atrText,
                donchianHighText,
                donchianLowText));
        }

        private void DisposeWriter()
        {
            if (writer == null)
                return;

            writer.Flush();
            writer.Dispose();
            writer = null;
        }
    }
}
