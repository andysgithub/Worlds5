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


        [DllImport("Unmanaged.dll", CallingConvention = CallingConvention.Cdecl)]
        static extern int TraceRay(
            float startDistance, RayTracingParams rayParams,
            float xFactor, float yFactor, float zFactor,
            int[] externalsArray, float[] valuesArray, float[] anglesArray, float[] distancesArray);

        private int[] externalPoints;
        private float[] modulusValues;
        private float[] angleValues;
        private float[] distanceValues;

        public RayProcessing()
        {
            externalPoints = new int[100];
            modulusValues = new float[100];
            angleValues = new float[100];
            distanceValues = new float[100];
        }

        // Trace the ray on this latitude line
        public TracedRay.RayDataType ProcessRay(RayTracingParams rayParams, RenderingParams renderParams, int rayCountX, int rayCountY)
        {
            float latitude = rayParams.latitudeStart - rayCountY * rayParams.angularResolution;
            float longitude = rayParams.longitudeStart - rayCountX * rayParams.angularResolution;

            float latRadians = latitude * Globals.DEG_TO_RAD;
            float longRadians = longitude * Globals.DEG_TO_RAD;

            float xFactor = (float)Math.Cos(latRadians) * (float)Math.Sin(-longRadians);
            float yFactor = (float)Math.Sin(latRadians);
            float zFactor = (float)Math.Cos(latRadians) * (float)Math.Cos(-longRadians);

            // Set the start distance to the sphere sphereRadius
            float startDistance = rayParams.sphereRadius;

            // If clipping is enabled
            if (rayParams.useClipping)
            {
                // Get the 5D coordinates for the intersection between this vector and the clipping plane
                float distance = Clipping.CalculateDistance(latRadians, longRadians, rayParams.clippingAxes, rayParams.clippingOffset);

                // Set the start distance to this value if larger than the sphere radius
                if (distance > startDistance) startDistance = distance;
            }         

            // Trace the ray from the starting point outwards
            int points = TraceRay(
                startDistance, rayParams,
                xFactor, yFactor, zFactor,
                externalPoints, modulusValues, angleValues, distanceValues);

            // Resize arrays to the recordedPoints value
            Array.Resize(ref externalPoints, points);
            Array.Resize(ref modulusValues, points);
            Array.Resize(ref angleValues, points);
            Array.Resize(ref distanceValues, points);

            // Record the fractal value collection for this ray 
            TracedRay tracedRay = new TracedRay(externalPoints, modulusValues, angleValues, distanceValues, renderParams);

            // Add this ray to the ray map in the sphere
            return tracedRay.RayData;
        }
    }
}
