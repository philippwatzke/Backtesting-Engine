#region Using declarations
using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Globalization;
using System.IO;
using NinjaTrader.NinjaScript;
#endregion

namespace NinjaTrader.NinjaScript.Indicators
{
    public class RawBarDumpExporter : Indicator
    {
        private StreamWriter writer;
        private readonly CultureInfo invariant = CultureInfo.InvariantCulture;

        [NinjaScriptProperty]
        [Display(Name = "OutputPath", GroupName = "Export", Order = 0)]
        public string OutputPath { get; set; }

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Dump raw chart bars to CSV on each OnBarClose update.";
                Name = "RawBarDumpExporter";
                Calculate = Calculate.OnBarClose;
                IsOverlay = false;
                DisplayInDataBox = false;
                DrawOnPricePanel = false;
                PaintPriceMarkers = false;
                IsSuspendedWhileInactive = true;
                OutputPath = @"C:\temp\NT8_RawDump.csv";
            }
            else if (State == State.DataLoaded)
            {
                string directory = Path.GetDirectoryName(OutputPath);
                if (string.IsNullOrWhiteSpace(directory))
                    throw new InvalidOperationException("OutputPath must include a directory.");

                Directory.CreateDirectory(directory);
                writer = new StreamWriter(OutputPath, false);
                writer.AutoFlush = true;
                writer.WriteLine("Timestamp_UTC,Open,High,Low,Close,Volume");
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

            writer.WriteLine(string.Join(
                ",",
                Time[0].ToUniversalTime().ToString("yyyy-MM-dd HH:mm:ss", invariant),
                Open[0].ToString("G17", invariant),
                High[0].ToString("G17", invariant),
                Low[0].ToString("G17", invariant),
                Close[0].ToString("G17", invariant),
                Volume[0].ToString(invariant)));
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
