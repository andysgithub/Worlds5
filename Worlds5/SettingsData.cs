using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Worlds5
{
    public static class SettingsData
    {
        public struct MainWindow
        {
            public static string MainState;
            public static int MainWidth;
            public static int MainHeight;
            public static int MainLeft;
            public static int MainTop;
        }

        public struct User
        {
            public static string NavPath;
            public static string SeqPath;
            public static bool Toolbar;
            public static bool Labels;
            public static bool ToolTips;
            public static bool StatusBar;
        }

        public struct Animation
        {
            public static int FramesPerSec;
            public static bool AutoRepeat;
        }

        public struct Viewing
        {
            public static double ViewportResolution;
            public static double SphereRadius;
            public static double CentreLatitude;
            public static double CentreLongitude;
            public static double VerticalView;
            public static double HorizontalView;
        }

        public struct Raytracing
        {
            public static double[] SamplingInterval = new double[2];
            public static int[] RayPoints = new int[2];
            public static int[] MaxSamples = new int[2];
            public static int[] BinarySearchSteps = new int[2];
            public static double SurfaceThickness;
            public static double BoundaryInterval;
            public static int ActiveIndex;
        }

        public struct Rendering
        {
            public static float[] ExposureValue = new float[2];
            public static float[] Saturation = new float[2];
            public static double[] StartDistance = new double[2];
            public static double[] EndDistance = new double[2];
            public static float SurfaceContrast;
            public static float LightingAngle;
            public static int BitmapHeight;
            public static int BitmapWidth;
        }
    }
}
