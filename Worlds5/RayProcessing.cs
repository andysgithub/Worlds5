using System;
using System.Runtime.InteropServices;
using Model;

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
            public bool cudaMode;
            public int maxSamples;
            public int rayPoints;
            public float samplingInterval;
            public float sphereRadius;
            public float surfaceSmoothing;
            public float surfaceThickness;

            public RayTracingParams(clsSphere.Settings settings)
            {
                activeIndex = settings.ActiveIndex;
                angularResolution = settings.AngularResolution;
                bailout = settings.Bailout;
                binarySearchSteps = settings.BinarySearchSteps[settings.ActiveIndex];
                boundaryInterval = settings.BoundaryInterval;
                cudaMode = settings.CudaMode;
                maxSamples = settings.MaxSamples[settings.ActiveIndex];
                rayPoints = (int)(settings.MaxSamples[settings.ActiveIndex] * settings.SamplingInterval[settings.ActiveIndex]);
                samplingInterval = settings.SamplingInterval[settings.ActiveIndex];
                sphereRadius = settings.SphereRadius;
                surfaceSmoothing = settings.SurfaceSmoothing;
                surfaceThickness = settings.SurfaceThickness;
            }
        }


        [DllImport("Unmanaged.dll", CallingConvention = CallingConvention.Cdecl)]
        static extern int TraceRay(float startDistance, float samplingInterval, float surfaceSmoothing, float surfaceThickness,
            float xFactor, float yFactor, float zFactor, float bailout,
            int[] externalsArray, float[] valuesArray, float[] anglesArray, float[] distancesArray,
            int rayPoints, int maxSamples, float boundaryInterval, int binarySearchSteps,
            int activeIndex, bool cudaMode);

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
        public void ProcessRay(clsSphere sphere, int rayCountX, int rayCountY)
        {
            clsSphere.Settings settings = sphere.settings;
            float latitude = settings.LatitudeStart - rayCountY * settings.AngularResolution;
            float longitude = settings.LongitudeStart - rayCountX * settings.AngularResolution;
            int i = settings.ActiveIndex;
            int rayPoints = (int)(settings.MaxSamples[i] * settings.SamplingInterval[i]);

            float latRadians = latitude * Globals.DEG_TO_RAD;
            float longRadians = longitude * Globals.DEG_TO_RAD;

            float xFactor = (float)Math.Cos(latRadians) * (float)Math.Sin(-longRadians);
            float yFactor = (float)Math.Sin(latRadians);
            float zFactor = (float)Math.Cos(latRadians) * (float)Math.Cos(-longRadians);

            // Set the start distance to the sphere sphereRadius
            float startDistance = settings.SphereRadius;

            // If clipping is enabled
            if (settings.UseClipping)
            {
                // Get the 5D coordinates for the intersection between this vector and the clipping plane
                float distance = Clipping.CalculateDistance(latRadians, longRadians, settings.ClippingAxes, settings.ClippingOffset);

                // Set the start distance to this value if larger than sphere sphereRadius
                if (distance > startDistance) startDistance = distance;
            }         

            // Trace the ray from the starting point outwards
            int points = TraceRay(startDistance, settings.SamplingInterval[i], settings.SurfaceSmoothing, settings.SurfaceThickness,
                        xFactor, yFactor, zFactor, settings.Bailout,
                        externalPoints, modulusValues, angleValues, distanceValues,
                        rayPoints, settings.MaxSamples[i], settings.BoundaryInterval, settings.BinarySearchSteps[i],
                        i, settings.CudaMode);

            // Resize arrays to the recordedPoints value
            Array.Resize(ref externalPoints, points);
            Array.Resize(ref modulusValues, points);
            Array.Resize(ref angleValues, points);
            Array.Resize(ref distanceValues, points);

            // Record the fractal value collection for this ray 
            TracedRay tracedRay = new TracedRay(externalPoints, modulusValues, angleValues, distanceValues);

            // Add this ray to the ray map in the sphere
            sphere.RayMap[rayCountX, rayCountY] = tracedRay.RayData;
        }
    }
}
