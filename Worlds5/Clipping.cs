using System;
using Model;

namespace Worlds5
{
    public class Clipping
    {
        public static AxisPair GetAxes(int axesIndex)
        {
            return (AxisPair)Enum.ToObject(typeof(AxisPair), axesIndex);
        }
        public static int GetIndex(AxisPair axisPair)
        {
            return (int)axisPair;
        }

        /// <summary>
        /// Find the distance between the viewpoint and clipping plane
        /// along a vector defined by the lat/long point on the sphere
        /// </summary>
        /// <param name="viewpoint5D"></param> The viewpoint coordinate in fractal space
        /// <param name="latRadians"></param> The latitude in radians of the point on the sphere to trace through
        /// <param name="longRadians"></param> The longitude in radians of the point on the sphere to trace through
        /// <param name="axes"></param> The selected axis pair for the clipping plane
        /// <param name="offset"></param> The offset for the clipping plane in the remaining axes
        /// <returns>The distance value as a float-precision float</returns>
        public static float CalculateDistance(float latRadians, float longRadians, AxisPair axisPair, float offset)
        {
            Vector3 direction3D = new Vector3(
                (float)Math.Cos(latRadians) * (float)Math.Sin(longRadians),
                (float)Math.Sin(latRadians),
                (float)Math.Cos(latRadians) * (float)Math.Cos(longRadians)
            );
            Vector5 viewpoint5D = Transformation.ImageToFractalSpace(0, new Vector3(0, 0, 0));
            Vector5 direction5D = Transformation.ImageToFractalSpace(5, direction3D);

            // Calculate the plane normal in 5D space
            Vector5 planeNormal = GetPlaneNormal(axisPair);

            // Calculate the dot product of the direction with the plane normal
            float dotProduct = Vector5.Dot(direction5D, planeNormal);

            // Calculate t
            float t = (dotProduct != 0) ? (offset - Vector5.Dot(viewpoint5D, planeNormal)) / dotProduct : 0;

            // Calculate distance
            float distance = Math.Abs(t * direction5D.Magnitude());
            return distance;
        }

        private static Vector5 GetPlaneNormal(AxisPair axisPair)
        {
            switch (axisPair)
            {
                case AxisPair.XY: return new Vector5(0, 0, 1, 0, 0);
                case AxisPair.XZ: return new Vector5(0, 1, 0, 0, 0);
                case AxisPair.XW: return new Vector5(0, 0, 0, 1, 0);
                case AxisPair.XV: return new Vector5(0, 0, 0, 0, 1);
                case AxisPair.YZ: return new Vector5(1, 0, 0, 0, 0);
                case AxisPair.YW: return new Vector5(0, 1, 0, 1, 0);
                case AxisPair.YV: return new Vector5(0, 1, 0, 0, 1);
                case AxisPair.ZW: return new Vector5(0, 0, 1, 1, 0);
                case AxisPair.ZV: return new Vector5(0, 0, 1, 0, 1);
                case AxisPair.WV: return new Vector5(0, 0, 0, 1, 1);
                default: throw new ArgumentException("Invalid AxisPair");
            }
        }
    }
}
