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
            public float[,] PositionMatrix { get; set; }
            public string RayMap { get; set; }
            public string ViewportImage { get; set; }
        }

        public class Viewing
        {
            public float AngularResolution { get; set; }
            public float Radius { get; set; }
            public float CentreLatitude { get; set; }
            public float CentreLongitude { get; set; }
            public float VerticalView { get; set; }
            public float HorizontalView { get; set; }
        }

        public class Raytracing
        {
            public float[] SamplingInterval { get; set; }
            public int[] MaxSamples { get; set; }
            public int[] BinarySearchSteps { get; set; }
            public float SurfaceSmoothing { get; set; }
            public float SurfaceThickness { get; set; }
            public float BoundaryInterval { get; set; }
            public float Bailout { get; set; }
            public int ActiveIndex { get; set; }
            public bool CudaMode { get; set; }
        }

        public class Rendering
        {
            public float[] ExposureValue { get; set; }
            public float[] Saturation { get; set; }
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