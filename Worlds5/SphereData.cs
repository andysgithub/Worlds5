using System;
using System.Drawing;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Model;

namespace Worlds5
{
    public class SphereData
    {
        public class Type
        {
            public string FileType { get; set; }
            public int Dimensions { get; set; }
        }

        public class Navigation
        {
            public double[,] PositionMatrix { get; set; }
            public string RayMap { get; set; }
            public string ViewportImage { get; set; }
        }

        public class Viewing
        {
            public double AngularResolution { get; set; }
            public double Radius { get; set; }
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
            public double SurfaceSmoothing { get; set; }
            public double SurfaceThickness { get; set; }
            public double BoundaryInterval { get; set; }
            public float Bailout { get; set; }
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
        }

        public class Colour
        {
            public float ColourCompression { get; set; }
            public float ColourOffset { get; set; }
        }

        public class RootObject
        {
            public Type Type { get; set; }
            public Navigation Navigation { get; set; }
            public Viewing Viewing { get; set; }
            public Raytracing Raytracing { get; set; }
            public Rendering Rendering { get; set; }
            public Colour Colour { get; set; }
        }
    }
}