using System;
using System.Runtime.InteropServices;
using Model;
using static Model.TracedRay;

namespace Worlds5
{
    sealed public class RayProcessing
    {
        [StructLayout(LayoutKind.Sequential)]
        public struct RayTracingParams
        {
            public int activeIndex;
            public float angularResolution;
            public float bailout;
            public int binarySearchSteps;
            public float boundaryInterval;
            public AxisPair clippingAxes;
            public float clippingOffset;
            public bool cudaMode;
            public float latitudeStart;
            public float longitudeStart;
            public int maxSamples;
            public float samplingInterval;
            public float sphereRadius;
            public float surfaceSmoothing;
            public float surfaceThickness;
            public bool useClipping;

            public RayTracingParams(clsSphere.Settings settings)
            {
                activeIndex = settings.ActiveIndex;
                angularResolution = settings.AngularResolution;
                bailout = settings.Bailout;
                binarySearchSteps = settings.BinarySearchSteps[settings.ActiveIndex];
                boundaryInterval = settings.BoundaryInterval;
                clippingAxes = settings.ClippingAxes;
                clippingOffset = settings.ClippingOffset;
                cudaMode = settings.CudaMode;
                latitudeStart = settings.LatitudeStart;
                longitudeStart = settings.LongitudeStart;
                maxSamples = settings.MaxSamples[settings.ActiveIndex];
                samplingInterval = settings.SamplingInterval[settings.ActiveIndex];
                sphereRadius = settings.SphereRadius;
                surfaceSmoothing = settings.SurfaceSmoothing;
                surfaceThickness = settings.SurfaceThickness;
                useClipping = settings.UseClipping;
            }
        }
    }
}
