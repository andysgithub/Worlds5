using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Worlds5
{
    public class SphereData
    {
        public class Type
        {
            public int FileType { get; set; }
            public int Dimensions { get; set; }
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
            public double[] ExposureValue { get; set; }
            public int[] Saturation { get; set; }
            public double[] StartDistance { get; set; }
            public double[] EndDistance { get; set; }
            public double[] ColourDetail { get; set; }
            public double SurfaceContrast { get; set; }
            public double LightingAngle { get; set; }
        }

        public class RootObject
        {
            public Type Type { get; set; }
            public double[][] PositionMatrix { get; set; }
            public double ScaleValue { get; set; }
            public double Bailout { get; set; }
            public Viewing Viewing { get; set; }
            public Raytracing Raytracing { get; set; }
            public Rendering Rendering { get; set; }
        }
    }
}