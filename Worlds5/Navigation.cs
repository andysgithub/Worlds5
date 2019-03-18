using System;
using System.IO;
using System.Collections;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Diagnostics;
using System.Windows.Forms;
using Model;

namespace Worlds5
{
    sealed public class Navigation
    {
        public static string LocationName;

        public static bool Navigate(string Address)
        {
            // Check that address is available if attempting to load file
            if (Address != "" && !Address.StartsWith("seq:"))
            {
                // Load navigation file from Address if known
                if (File.Exists(Address + ".nav"))
                {
                    // Load navigation parameters
                    if (LoadData(Address + ".nav"))
                    {
                        return true;
                    }
                }
                else
                {
                    MessageBox.Show("The file is not available:\n" + Address, "Navigation Error 2",
                                    MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
                }
            }
            return false;
        }

        public static bool SaveData(ref string FileName)
        {
            double MatrixValue = 0;
            int i = 0, j = 0;

            float[] ColourDetail = new float[2];
            int Count = 0;
            int ErrorNo = 0;

            do
            {
                try
                {
                    for (Count = 0; Count <= 1; Count++)
                    {
                        ColourDetail[Count] = Model.Globals.Sphere.ColourDetail[Count];
                    }

                    //  Open file
                    TextWriter tw = new StreamWriter(FileName);

                    //  Save file-type reference
                    tw.WriteLine("{0}", 2);
                    tw.WriteLine("");

                    //  Save transformation matrix
                    for (i = 0; i <= Model.Globals.Dimensions; i++)
                    {
                        for (j = 0; j <= Model.Globals.Dimensions - 1; j++)
                        {
                            MatrixValue = Model.Globals.Sphere.PositionMatrix[i, j];
                            tw.Write("{0}", MatrixValue);
                        }
                        tw.WriteLine("");
                    }
                    tw.WriteLine("{0}", ImageRendering.ScaleValue);
                    tw.WriteLine("");

                    //  Save Options settings
                    tw.WriteLine(ImageRendering.Bailout);

                    for (Count = 0; Count <= 1; Count++)
                    {
                        tw.WriteLine(ColourDetail[Count], "");
                    }
                    //  Close file
                    tw.Close();
                    return true;
                }
                catch (Exception ex)
                {
                    ErrorNo = (int)(MessageBox.Show(
                                        "Couldn't write data to the file:\n" + FileName +
                                        "\n\nError reported: " + ex.Message + "\n",
                                        "Save Data Error",
                                        MessageBoxButtons.RetryCancel,
                                        MessageBoxIcon.Exclamation));

                    if (ErrorNo == (int)DialogResult.Cancel)
                    {
                        return false;
                    }
                }
            }
            while (true);
        }

        //  Read navigation data into RenderImage class
        private static bool LoadData(string FileName)
        {
            int i = 0, j = 0;
            int FileType = 0;
            string Linefeed = null;
            int Count = 0;
            string[] data;

            try
            {
                clsSphere sphere = Model.Globals.Sphere;
                //  Open file
                StreamReader s = File.OpenText(FileName);

                //  Load file-type reference and total dimensions
                data = s.ReadLine().Split();
                FileType = Convert.ToInt16(data[0]);

                // Redimension the transformation matrix
                int iDimensions = Convert.ToInt16(data[1]);
                Model.Globals.Dimensions = iDimensions;
                sphere.PositionMatrix = new double[iDimensions + 1, iDimensions]; 

                Linefeed = s.ReadLine();

                //  Load transformation matrix
                for (i = 0; i <= iDimensions; i++)
                {
                    data = s.ReadLine().Split();

                    for (j = 0; j <= iDimensions - 1; j++)
                        sphere.PositionMatrix[i, j] = double.Parse(data[j]);
                }

                ImageRendering.ScaleValue = double.Parse(s.ReadLine());
                Linefeed = s.ReadLine();

                //  Load Rendering settings
                if (FileType == 3)
                {
                    data = s.ReadLine().Split();
                    ImageRendering.Bailout = float.Parse(data[2]);
                    for (Count = 0; Count <= 1; Count++)
                    {
                        data = s.ReadLine().Split();
                        sphere.ColourDetail[Count] = float.Parse(data[0]);
                    }

                    // Sphere Viewing window
                    sphere.AngularResolution = viewing.ViewportResolution;
                    sphere.Radius = viewing.SphereRadius;
                    sphere.CentreLatitude = viewing.CentreLatitude;
                    sphere.CentreLongitude = viewing.CentreLongitude;
                    sphere.VerticalView = viewing.VerticalView;
                    sphere.HorizontalView = viewing.HorizontalView;

                    // Raytracing
                    sphere.SamplingInterval = raytracing.SamplingInterval;
                    sphere.SurfaceThickness = raytracing.SurfaceThickness;
                    sphere.RayPoints = raytracing.RayPoints;
                    sphere.MaxSamples = raytracing.MaxSamples;
                    sphere.BoundaryInterval = raytracing.BoundaryInterval;
                    sphere.BinarySearchSteps = raytracing.BinarySearchSteps;
                    sphere.ActiveIndex = raytracing.ActiveIndex;

                    // Rendering
                    sphere.ExposureValue = rendering.ExposureValue;
                    sphere.Saturation = rendering.Saturation;
                    sphere.StartDistance = rendering.StartDistance;
                    sphere.EndDistance = rendering.EndDistance;
                    sphere.SurfaceContrast = rendering.SurfaceContrast;
                    sphere.LightingAngle = rendering.LightingAngle;

                    //  Close file
                    s.Close();
                }
                else
                {
                    MessageBox.Show(
                        "File type not recognised:\n" + FileName + "\n",
                        "Navigation Error",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Exclamation);
                    return false;
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    "Error reading navigation file:\n" + FileName + "\n" + ex.Message,
                    "Navigation Error",
                    MessageBoxButtons.RetryCancel,
                    MessageBoxIcon.Exclamation);
                return false;
            }
            return true;
        }
    }
}
