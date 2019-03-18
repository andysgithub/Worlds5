using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Worlds5
{
    public class SettingsData
    {
        public class MainWindow
        {
            public string MainState { get; set; }
            public int MainWidth { get; set; }
            public int MainHeight { get; set; }
            public int MainLeft { get; set; }
            public int MainTop { get; set; }
        }

        public class User
        {
            public string NavPath { get; set; }
            public string SeqPath { get; set; }
            public bool Toolbar { get; set; }
            public bool Labels { get; set; }
            public bool ToolTips { get; set; }
            public bool StatusBar { get; set; }
        }

        public class Animation
        {
            public int FramesPerSec { get; set; }
            public bool AutoRepeat { get; set; }
        }

        public class Viewing
        {
            public double ViewportResolution { get; set; }
            public double SphereRadius { get; set; }
            public double CentreLatitude { get; set; }
            public double CentreLongitude { get; set; }
            public double VerticalView { get; set; }
            public double HorizontalView { get; set; }
        }

        public class Raytracing
        {
            public double[] SamplingInterval { get; set; }
            public int[] RayPoints { get; set; }
            public int[] MaxSamples { get; set; }
            public int[] BinarySearchSteps { get; set; }
            public double SurfaceThickness { get; set; }
            public double BoundaryInterval { get; set; }
            public int ActiveIndex { get; set; }
        }

        public class Rendering
        {
            public float[] ExposureValue { get; set; }
            public float[] Saturation { get; set; }
            public double[] StartDistance { get; set; }
            public double[] EndDistance { get; set; }
            public float SurfaceContrast { get; set; }
            public float LightingAngle { get; set; }
            public int BitmapHeight { get; set; }
            public int BitmapWidth { get; set; }
        }

        public class RootObject
        {
            public MainWindow MainWindow { get; set; }
            public User User { get; set; }
            public Animation Animation { get; set; }
            public Viewing Viewing { get; set; }
            public Raytracing Raytracing { get; set; }
            public Rendering Rendering { get; set; }
        }
    }
}
