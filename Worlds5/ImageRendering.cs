using System;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using Model;

namespace Worlds5
{
    sealed public class ImageRendering
    {
        #region Member Variables

        private clsSphere sphere;
        private ImageDisplay imageDisplay;
        private int linesProcessed = 0;

        //  Sequence playback
        // private static int m_CurrentKey;		// Current key frame for sequence
        // private static int m_FrameCount;		// Frame to display for current key

        #endregion

        #region Delegates

        public delegate void UpdateStatusDelegate(int rowCount, int totalLines);
        public event UpdateStatusDelegate updateStatus;

        #endregion

        #region Class Properties

        // public static Size SeqSize
        // {
        //     get { return m_SeqSize; }
        //     set { m_SeqSize = value; }
        // }

        //public static double ScaleValue
        //{
        //    get { return m_ScaleValue; }
        //    set { m_ScaleValue = value; }
        //}

        // public static int CurrentKey
        // {
        //     get { return m_CurrentKey; }
        //     set { m_CurrentKey = value; }
        // }

        // public static int FrameCount
        // {
        //     get { return m_FrameCount; }
        //     set { m_FrameCount = value; }
        // }

        #endregion

        #region DLL Imports

        [DllImport("Unmanaged.dll")]
        static extern int TraceRay(double startDistance, double increment, double smoothness, double surfaceThickness,
            double xFactor, double yFactor, double zFactor, int[] externalsArray,
            float[] valuesArray, float[] anglesArray, double[] distancesArray,
            int rayPoints, int maxSamples, double boundaryInterval, int binarySearchSteps,
            int activeIndex);

        [DllImport("Unmanaged.dll")]
        static extern void InitSphere(float fBailout, double Resolution,
                                      double Latitude, double Longitude,
                                      double Radius, double verticalView, double horizontalView, double[,] PositionMatrix);

        #endregion

        public ImageRendering()
        {

        }

        public void InitialiseSphere()
        {
            // Set the ImageRendering sphere to the global sphere
            sphere = Model.Globals.Sphere;

            // Initialise the sphere in the dll from the ImageRendering sphere
            InitSphere(sphere.settings.Bailout, sphere.settings.AngularResolution,
                       sphere.settings.CentreLatitude, sphere.settings.CentreLongitude,
                       sphere.settings.Radius, sphere.settings.VerticalView, sphere.settings.HorizontalView, sphere.settings.PositionMatrix);
        }

        public async Task<bool> PerformRayTracing()
        {
            return await Task.Run(() =>
            {
                imageDisplay = new ImageDisplay();
                imageDisplay.LockBitmap();

                try
                {
                    // Initialise ray map
                    sphere.InitialiseRayMap();

                    int totalLines = (int)(sphere.settings.VerticalView / sphere.settings.AngularResolution);
                    int totalRays = (int)(sphere.settings.HorizontalView / sphere.settings.AngularResolution);

                    linesProcessed = 0;

                    PerformParallel(totalLines, totalRays);
                }
                catch (Exception e)
                {
                    MessageBox.Show("Error: " + e.Message, "Raytracing Error");
                }

                imageDisplay.UnlockBitmap();
                // Store the image in the sphere
                sphere.ViewportImage = imageDisplay.GetBitmap();
                return true;
            });
        }

        private void PerformParallel(int totalLines, int totalRays)
        {
            Parallel.For(0, totalLines, rayCountY =>
            {
                // For each longitude point on this line
                Parallel.For(0, totalRays, rayCountX =>
                {
                    try
                    {
                        // Perform raytracing
                        TracedRay tracedRay = ProcessRay(sphere, rayCountX, rayCountY);
                        // Add this ray to the ray map in the sphere
                        sphere.RecordRay(tracedRay, rayCountX, rayCountY);
                        ProgressChanged(rayCountX, rayCountY, tracedRay);
                    }
                    catch
                    { }
                });
                RowCompleted((int)rayCountY, DisplayOption.None);
            });

            Parallel.For(0, totalLines, rayCountY =>
            {
                // For each longitude point on this line
                Parallel.For(0, totalRays, rayCountX =>
                 {
                     try
                     {
                         // Set the colour and display the point
                         TracedRay tracedRay = SetRayColour(sphere, rayCountX, rayCountY);
                         imageDisplay.updateImage(rayCountX, rayCountY, tracedRay.bmiColors);
                     }
                     catch
                     { }
                 });
            });
        }

        public async Task<bool> Redisplay()
        {
            return await Task.Run(() =>
            {
                //picImage.Image = new Bitmap(picImage.Image.Width, picImage.Image.Height);
                if (Model.Globals.Sphere.RayMap != null)
                {
                    int totalLines = (int)(sphere.settings.VerticalView / sphere.settings.AngularResolution);
                    linesProcessed = 0;

                    try
                    {
                        // for (int lineIndex = 0; lineIndex < totalLines; lineIndex++)
                        Parallel.For(0, totalLines, lineIndex =>
                        {
                            // Set pixel colours for this line
                            Redisplay(lineIndex);
                            RowCompleted((int)lineIndex, DisplayOption.None);
                        });
                        sphere.ViewportImage = imageDisplay.GetBitmap();
                    }
                    catch (Exception e)
                    {
                        MessageBox.Show("Error: " + e.Message, "Redisplay Error");
                    }
                }
                return true;
            });
        }

        private void ProgressChanged(int rayCountX, int rayCountY, TracedRay ray)
        {
            if (ray != null)
            {
                int totalRays = (int)(sphere.settings.HorizontalView / sphere.settings.AngularResolution);

                // If current row is still being processed
                if (rayCountX < totalRays - 1)
                {
                    imageDisplay.updateImage(rayCountX, rayCountY, ray.bmiColors);
                }
            }
        }

        private void RowCompleted(int lineIndex, DisplayOption displayOption)
        {
            linesProcessed++;
            int totalLines = (int)(sphere.settings.VerticalView / sphere.settings.AngularResolution);

            // Call the UpdateStatus function in Main
            updateStatus(linesProcessed, totalLines);
        }

        public void Redisplay(int rayCountY)
        {
            int totalRays = (int)(sphere.settings.HorizontalView / sphere.settings.AngularResolution);

            // For each longitude point on this line
            for (int rayCountX = 0; rayCountX < totalRays; rayCountX++)
            {
                // Display the point on this line of latitude
                TracedRay tracedRay = SetRayColour(sphere, rayCountX, rayCountY);
                ProgressChanged(rayCountX, rayCountY, tracedRay);
            }
        }

        // Trace the ray on this latitude line
        private TracedRay ProcessRay(clsSphere sphere, int rayCountX, int rayCountY)
        {
            double latitude = sphere.settings.LatitudeStart - rayCountY * sphere.settings.AngularResolution;
            double longitude = sphere.settings.LongitudeStart - rayCountX * sphere.settings.AngularResolution;

            double xFactor = Math.Cos(latitude * Globals.DEG_TO_RAD) * Math.Sin(-longitude * Globals.DEG_TO_RAD);
            double yFactor = Math.Sin(latitude * Globals.DEG_TO_RAD);
            double zFactor = Math.Cos(latitude * Globals.DEG_TO_RAD) * Math.Cos(-longitude * Globals.DEG_TO_RAD);

            TracedRay tracedRay;
            int[] externalPoints = new int[100];
            float[] modulusValues = new float[100];
            float[] angleValues = new float[100];
            double[] distanceValues = new double[100];
            int i = sphere.settings.ActiveIndex;

            int rayPoints = (int)(sphere.settings.MaxSamples[i] * sphere.settings.SamplingInterval[i]);

            // Trace the ray from the sphere radius outwards
            int points = TraceRay(sphere.settings.Radius, sphere.settings.SamplingInterval[i], sphere.settings.SurfaceSmoothing, sphere.settings.SurfaceThickness,
                        xFactor, yFactor, zFactor,
                        externalPoints, modulusValues, angleValues, distanceValues,
                        rayPoints, sphere.settings.MaxSamples[i], sphere.settings.BoundaryInterval, sphere.settings.BinarySearchSteps[i],
                        i);

            // Resize arrays to the recordedPoints value
            Array.Resize(ref externalPoints, points);
            Array.Resize(ref modulusValues, points);
            Array.Resize(ref angleValues, points);
            Array.Resize(ref distanceValues, points);

            // Record the fractal value collection for this ray
            tracedRay = new TracedRay(externalPoints, modulusValues, angleValues, distanceValues);
            return tracedRay;
        }

        private TracedRay SetRayColour(clsSphere sphere, int rayCountX, int rayCountY)
        {
            if (rayCountX >= sphere.RayMap.GetUpperBound(0) || rayCountY >= sphere.RayMap.GetUpperBound(1))
            {
                return null;
            }
            // Get the ray from the ray map
            TracedRay.RayDataType rayData = sphere.RayMap[rayCountX++, rayCountY];
            TracedRay tracedRay = new TracedRay(rayData.ExternalPoints, rayData.ModulusValues, rayData.AngleValues, rayData.DistanceValues);

            // Calculate the tilt values from the previous rays
            if (rayCountX > 0)
            {
                tracedRay.xTiltValues = sphere.addTiltValues(tracedRay, rayCountX - 1, rayCountY);
            }
            if (rayCountY > 0)
            {
                tracedRay.yTiltValues = sphere.addTiltValues(tracedRay, rayCountX, rayCountY - 1);
            }

            if (tracedRay != null)
            {
                // Convert the fractal value collection into an rgb colour value
                tracedRay.SetColour();
            }
            return tracedRay;
        }

        public Bitmap GetBitmap()
        {
            return imageDisplay.GetBitmap();
        }

        //private void extendBoundaries(int row)
        //{
        //    // Start with the first ray
        //    int rayNumber = 0;

        //    // For each longitude point on this line working forwards
        //    for (double longitude = sphere.settings.LongitudeStart - sphere.settings.Resolution;
        //        longitude > sphere.settings.LongitudeEnd; longitude -= sphere.settings.Resolution)
        //    {
        //        // Get the ray from the ray map
        //        TracedRay thisRay = sphere.RayMap[rayNumber, row];
        //        // Get the next ray from the ray map
        //        TracedRay nextRay = sphere.RayMap[rayNumber + 1, row];
        //        // Add any additional boundaries on this ray to the next one
        //        addNewBoundaries(thisRay, ref nextRay);
        //        // Go on to the next ray in the row
        //        rayNumber++;
        //    }

        //    // Start with the last ray

        //    // For each longitude point on this line working backwards
        //    for (double longitude = sphere.settings.LongitudeEnd + sphere.settings.Resolution;
        //        longitude < sphere.settings.LongitudeStart; longitude += sphere.settings.Resolution)
        //    {
        //        // Get the ray from the ray map
        //        TracedRay thisRay = sphere.RayMap[rayNumber, row];
        //        // Get the previous ray from the ray map
        //        TracedRay previousRay = sphere.RayMap[rayNumber - 1, row];
        //        // Add any additional boundaries on this ray to the previous one
        //        addNewBoundaries(thisRay, ref previousRay);
        //        // Go on to the previous ray in the row
        //        rayNumber--;
        //    }
        //}

        ///// <summary>
        ///// Find extra boundaries from the source list which aren't in the target list
        ///// and return a new target list.
        ///// </summary>
        ///// <param name="sourceBoundaries">The list of boundary distances to test.</param>
        ///// <param name="targetBoundaries">The list of boundary distances to add to.</param>
        ///// <returns></returns>
        //private void addNewBoundaries(TracedRay sourceRay, ref TracedRay targetRay)
        //{
        //    List<double> newBoundaries = targetRay.Boundaries;
        //    int sourcePos = -1;
        //    int targetPos = 0;

        //    // For each source boundary position value
        //    foreach (double sourcePosition in sourceRay.Boundaries)
        //    {
        //        sourcePos++;
        //        bool boundaryFound = false;

        //        // Search for the nearest item in the target list
        //        while (targetRay.Boundary(targetPos) < sourcePosition)
        //        {
        //            if (targetRay.Boundary(targetPos + 1) > sourcePosition)
        //            {
        //                boundaryFound = true;
        //                break;
        //            }
        //            targetPos++;
        //        }

        //        if (Double.IsPositiveInfinity(targetRay.Boundary(targetPos)))
        //        {
        //            return;
        //        }

        //        // If no position found to insert this boundary
        //        if (!boundaryFound)
        //        {
        //            // Go on to the next source boundary position
        //            continue;
        //        }

        //        // If this angle is sufficiently close to the test angle
        //        if (Math.Abs(sourceRay.Angle(sourcePos) - targetRay.Angle(targetPos)) < sphere.settings.BoundaryInterval)
        //        {
        //            // Go on to the next source boundary position
        //            continue;
        //        }

        //        // Set increment to the distance between the target and source points
        //        double separation = sourcePosition - targetRay.Boundary(targetPos);

        //        float Modulus, Angle;
        //        bool externalPoint;

        //        // Perform binary search between the target and the source points, to determine new boundary position
        //        double newPosition = FindBoundary(separation, sphere.settings.BinarySearchSteps, sourcePosition, targetRay.Angle(targetPos),
        //                              sphere.settings.BoundaryInterval, &externalPoint, &Modulus, &Angle);

        //        // Insert this new boundary data into the current position in the target ray
        //        targetRay.Boundaries.Insert(targetPos+1, newPosition);
        //        targetRay.AngleValues.Insert(targetPos+1, Angle);
        //        targetRay.ModulusValues.Insert(targetPos+1, Modulus);
        //        targetRay.ExternalPoints.Insert(targetPos+1, externalPoint);
        //        targetPos++;
        //    }
        //}

    }
}
