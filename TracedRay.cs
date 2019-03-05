using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices;

namespace DataClasses
{
    public unsafe class TracedRay
    {
        [DllImport("Unmanaged.dll")]
        static extern void HSVtoRGB(float h, float s, float v, byte* r, byte* g, byte* b);

        public List<bool> externalPoints;
        public List<float> modulusValues;
        public List<float> angleValues;
        public List<double> distanceValues;

        public int sampleCount;
        // RGB colour
        public Globals.RGBQUAD bmiColors;

        public TracedRay(bool[] externalPoints, float[] modulusValues, float[] angleValues, double[] distanceValues)
        {
            this.externalPoints = new List<bool>(externalPoints);
            this.modulusValues = new List<float>(modulusValues);
            this.angleValues = new List<float>(angleValues);
            this.distanceValues = new List<double>(distanceValues);
            sampleCount = externalPoints.Length;
        }

        public void SetColour(float exposureValue, float saturation, double startDistance, double endDistance)
        {
            byte r, g, b;
            Globals.RGBTRIPLE totalRGB;

            // Initialise RGB to 0,0,0
            totalRGB.rgbRed = 0;
            totalRGB.rgbGreen = 0;
            totalRGB.rgbBlue = 0;

            // For each point on the ray
            // If another point follows this one
            for (int i = 0; i < modulusValues.Count - 1; i++)
            {
                if (Double.IsPositiveInfinity(distanceValues[i + 1])
                    || distanceValues[i + 1] > endDistance)
                    break;

                if (distanceValues[i] < startDistance)
                    continue;

                if (Math.Abs(modulusValues[i]) < 10)
                {
                    // Get distance between points
                    double distance = distanceValues[i + 1] - distanceValues[i];

                    // Get Hue from the orbit angle
                    float Hue = (float)(angleValues[i] * 57.2957795 * 2);
                    // S & V set by distance * exposure value
                    float Lightness = (float)distance * exposureValue;
                    float Saturation = Lightness * saturation;

                    // Limit S & V to 1 maximum
                    Saturation = Saturation > 1 ? 1 : Saturation;
                    Lightness = Lightness > 1 ? 1 : Lightness;

                    // Convert HSV to RGB
                    HSVtoRGB(Hue, Saturation, Lightness, &r, &g, &b);
                    // Add result to RGB total
                    totalRGB.rgbRed += r;
                    totalRGB.rgbGreen += g;
                    totalRGB.rgbBlue += b;
                    // Limit RGB values to 255
                    totalRGB.rgbRed = totalRGB.rgbRed > 255 ? 255 : totalRGB.rgbRed;
                    totalRGB.rgbGreen = totalRGB.rgbGreen > 255 ? 255 : totalRGB.rgbGreen;
                    totalRGB.rgbBlue = totalRGB.rgbBlue > 255 ? 255 : totalRGB.rgbBlue;
                }
            }

            // Call GetColour to set RGB values
            //GetColour(externalPoints[i], modulusValues[i], angleValues[i], &r, &g, &b);

            bmiColors.rgbRed = (byte)totalRGB.rgbRed;
            bmiColors.rgbGreen = (byte)totalRGB.rgbGreen;
            bmiColors.rgbBlue = (byte)totalRGB.rgbBlue;
        }

        /// <summary>
        /// The number of sampled points along the ray
        /// </summary>
        public int Length
        {
            get { return sampleCount; }
        }

        // Return distance values for boundary positions
        public List<double> Boundaries
        {
            get { return distanceValues; }
        }

        // Return angle values for boundary positions
        public List<float> AngleValues
        {
            get { return angleValues; }
        }

        // Return modulus values for boundary positions
        public List<float> ModulusValues
        {
            get { return modulusValues; }
        }

        // Return external points for boundary positions
        public List<bool> ExternalPoints
        {
            get { return externalPoints; }
        }

        // Retun the total number of boundary positions
        public int BoundaryTotal
        {
            get { return distanceValues.Count; }
        }

        // Retun distance values for boundary positions
        public double Boundary(int index)
        {
            return distanceValues[index]; 
        }

        // Retun distance values for boundary positions
        public float Angle(int index)
        {
            return angleValues[index]; 
        }
    }
}
