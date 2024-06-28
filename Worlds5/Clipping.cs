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
            // Convert latitude and longitude to a unit direction vector in 3D space
            Vector3 direction3D = new Vector3(
                (float)(float)Math.Cos(latRadians) * (float)Math.Sin(longRadians),
                (float)Math.Sin(latRadians),
                (float)(float)Math.Cos(latRadians) * (float)(float)Math.Cos(longRadians)
            );

            // Get the 5D coordinates in the fractal space for the sphere centre
            Vector5 viewpoint5D = Transformation.ImageToFractalSpace(0, new Vector3(0, 0, 0));

            // Get the 5D coordinates in the fractal space for the vector
            Vector5 direction5D = Transformation.ImageToFractalSpace(5, direction3D);

            // Determine the intersection parameter t based on the selected axis pair
            float t = getIntersection(axisPair, offset, viewpoint5D, direction5D);

            float distance = Math.Abs(t * direction5D.Magnitude());

            return distance;
        }

        private static float getIntersection(AxisPair axisPair, float offset, Vector5 viewpoint, Vector5 direction)
        {
            float t = 0;

            switch (axisPair)
            {
                case AxisPair.XY:
                    if (direction.Z != 0) t = (offset - viewpoint.Z) / direction.Z;
                    break;
                case AxisPair.XZ:
                    if (direction.Y != 0) t = (offset - viewpoint.Y) / direction.Y;
                    break;
                case AxisPair.XW:
                case AxisPair.YW:
                case AxisPair.ZW:
                    if (direction.V != 0) t = (offset - viewpoint.V) / direction.V;
                    break;
                case AxisPair.XV:
                case AxisPair.YV:
                case AxisPair.ZV:
                    if (direction.W != 0) t = (offset - viewpoint.W) / direction.W;
                    break;
                case AxisPair.YZ:
                case AxisPair.WV:
                    if (direction.X != 0) t = (offset - viewpoint.X) / direction.X;
                    break;
            }

            return t;
        }
    }
}
