using Model;
using System;
using System.Windows.Forms;
using System.Threading.Tasks;

namespace Worlds5
{
    public partial class SphereSettings : Form
    {
        #region Delegates

        public delegate void RefreshDelegate();
        public delegate Task<bool> RaytraceDelegate();
        public delegate void StatusDelegate(string statusMessage);
        public event RefreshDelegate RefreshImage;
        public event RaytraceDelegate RaytraceImage;
        public event StatusDelegate UpdateStatus;

        #endregion

        private clsSphere.Settings sphereSettings = Model.Globals.Sphere.settings;
        private clsSphere.Settings oldSphereSettings = Model.Globals.Sphere.settings.Clone();
        private int regionIndex = 0;
        private int planeIndex = 0;
        private int axisIndex = 0;
        private bool isLoaded = false;
        private double[] translationValues = new double[5];
        private double[] rotationValues = new double[10];

        public SphereSettings()
        {
            InitializeComponent();
            cmbRegion.SelectedIndex = regionIndex;
            cmbPlane.SelectedIndex = planeIndex;
            updAxis.Value = (decimal)axisIndex + 1;
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            UpdateStatus("Saving settings...");
            SaveSettings();
            SaveRendering();
            UpdateStatus("");
            this.Close();
        }

        private async void btnRaytrace_Click(object sender, EventArgs e)
        {
            UpdateStatus("Initialising...");
            SaveSettings();
            SaveRendering();
            UpdateStatus("");
            await RaytraceImage();
        }

        private double[,] Angles
        {
            get
            {
                // Factor to convert total degrees into radians
                double factor = Globals.DEG_TO_RAD;
                double[,] angles = new double[5, 6];

                angles[1, 2] = rotationValues[0] * factor;
                angles[1, 3] = rotationValues[1] * factor;
                angles[1, 4] = rotationValues[2] * factor;
                angles[1, 5] = rotationValues[3] * factor;
                angles[2, 3] = rotationValues[4] * factor;
                angles[2, 4] = rotationValues[5] * factor;
                angles[2, 5] = rotationValues[6] * factor;
                angles[3, 4] = rotationValues[7] * factor;
                angles[3, 5] = rotationValues[8] * factor;
                angles[4, 5] = rotationValues[9] * factor;

                return angles;
            }
        }

        private void SaveSettings()
        {
            // Transfer rotation values to position matrix
            double[,] angles = Angles;
            for (int axis1 = 1; axis1 < 5; axis1++)
            {
                for (int axis2 = 2; axis2 < 6; axis2++)
                {
                    // If rotation is set for this plane
                    if (angles[axis1, axis2] != 0)
                    {
                        // Rotate by the given angle
                        Transformation.SetRotation(axis1 - 1, axis2 - 1, angles[axis1, axis2]);
                        Transformation.PreMulR();
                    }
                }
            }

            sphereSettings.PositionMatrix = Transformation.GetPositionMatrix();
            // Clear the rotation input
            updRotate.Value = 0;

            // Sphere Viewing window
            sphereSettings.AngularResolution = (double)updResolution.Value;
            sphereSettings.Radius = (double)updSphereRadius.Value;
            sphereSettings.CentreLatitude = (double)updCentreLatitude.Value;
            sphereSettings.CentreLongitude = (double)updCentreLongitude.Value;
            sphereSettings.VerticalView = (double)updViewportHeight.Value;
            sphereSettings.HorizontalView = (double)updViewportWidth.Value;

            // Position
            for (int axis = 0; axis < 5; axis++)
            {
                sphereSettings.PositionMatrix[5, axis] += translationValues[axis];
                translationValues[axis] = 0;
            }
            // Clear the translation input
            updTranslate.Value = 0;

            // Raytracing
            sphereSettings.ActiveIndex = chkShowSurface.Checked ? 0 : 1;

            sphereSettings.SamplingInterval[0] = (double)updSamplingInterval_0.Value;
            sphereSettings.BinarySearchSteps[0] = (int)updBinarySearchSteps_0.Value;
            sphereSettings.MaxSamples[0] = (int)updMaxSamples_0.Value;

            sphereSettings.SamplingInterval[1] = (double)updSamplingInterval_1.Value;
            sphereSettings.BinarySearchSteps[1] = (int)updBinarySearchSteps_1.Value;
            sphereSettings.MaxSamples[1] = (int)updMaxSamples_1.Value;

            // Surface
            sphereSettings.Bailout = (float)updBailout.Value;
            sphereSettings.BoundaryInterval = (double)updBoundaryInterval.Value;
            sphereSettings.SurfaceSmoothing = (double)updSurfaceSmoothing.Value;
            sphereSettings.SurfaceThickness = (double)updSurfaceThickness.Value;
        }

        private void SaveRendering()
        {
            sphereSettings.ExposureValue[regionIndex] = (float)updExposureValue.Value;
            sphereSettings.Saturation[regionIndex] = (float)updSaturation.Value;

            sphereSettings.SurfaceContrast = (float)updSurfaceContrast.Value;
            sphereSettings.LightingAngle = (float)updLightingAngle.Value;

            sphereSettings.ColourCompression = (float)updCompression.Value;
            sphereSettings.ColourOffset = (float)updOffset.Value;

            Model.Globals.Sphere.settings = sphereSettings.Clone();
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            Model.Globals.Sphere.settings = oldSphereSettings.Clone();
            this.Close();
        }

        private void Settings_Load(object sender, EventArgs e)
        {
            isLoaded = true;

            // Sphere Viewing window
            updResolution.Value = (decimal)sphereSettings.AngularResolution;
            updSphereRadius.Value = (decimal)sphereSettings.Radius;
            updCentreLatitude.Value = (decimal)sphereSettings.CentreLatitude;
            updCentreLongitude.Value = (decimal)sphereSettings.CentreLongitude;
            updViewportHeight.Value = (decimal)sphereSettings.VerticalView;
            updViewportWidth.Value = (decimal)sphereSettings.HorizontalView;

            // Position
            /*updTranslate.Value = 0;*/

            // Raytracing
            chkShowSurface.Checked = sphereSettings.ActiveIndex == 0;
            chkShowVolume.Checked = sphereSettings.ActiveIndex == 1;

            updSamplingInterval_0.Value = (decimal)sphereSettings.SamplingInterval[0];
            updBinarySearchSteps_0.Value = sphereSettings.BinarySearchSteps[0];
            updMaxSamples_0.Value = sphereSettings.MaxSamples[0];

            updSamplingInterval_1.Value = (decimal)sphereSettings.SamplingInterval[1];
            updBinarySearchSteps_1.Value = sphereSettings.BinarySearchSteps[1];
            updMaxSamples_1.Value = sphereSettings.MaxSamples[1];
            
            // Surface
            updBoundaryInterval.Value = (decimal)sphereSettings.BoundaryInterval;
            updSurfaceSmoothing.Value = (decimal)sphereSettings.SurfaceSmoothing;
            updSurfaceThickness.Value = (decimal)sphereSettings.SurfaceThickness;
            updBailout.Value = (decimal)sphereSettings.Bailout;

            // Rendering
            updExposureValue.Value = (decimal)sphereSettings.ExposureValue[regionIndex];
            updSaturation.Value = (decimal)sphereSettings.Saturation[regionIndex];

            updSurfaceContrast.Value = (decimal)sphereSettings.SurfaceContrast;
            updLightingAngle.Value = (decimal)sphereSettings.LightingAngle;

            updCompression.Value = (decimal)sphereSettings.ColourCompression;
            updOffset.Value = (decimal)sphereSettings.ColourOffset;
        }

        #region Help functions

        private void updExposureValue_HelpRequested(object sender, HelpEventArgs hlpevent)
        {
            showDetails("Exposure Value", "");
        }

        private void updSaturation_HelpRequested(object sender, HelpEventArgs hlpevent)
        {
            showDetails("Saturation", "");
        }

        private void updBitmapHeight_HelpRequested(object sender, HelpEventArgs hlpevent)
        {
            showDetails("Bitmap Height", "");
        }

        private void updBitmapWidth_HelpRequested(object sender, HelpEventArgs hlpevent)
        {
            showDetails("Bitmap Width", "");
        }

        private void cmbImageFileFormat_HelpRequested(object sender, HelpEventArgs hlpevent)
        {
            showDetails("Image File Format", "");
        }

        private void updResolution_HelpRequested(object sender, HelpEventArgs hlpevent)
        {
            showDetails("Viewport Resolution", "Angular resolution of the viewport surface (degrees)");
        }

        private void updSphereRadius_HelpRequested(object sender, HelpEventArgs hlpevent)
        {
            showDetails("Sphere Radius", "The distance from the centre of the sphereSettings to first ray tracing point");
        }

        private void updCentreLatitude_HelpRequested(object sender, HelpEventArgs hlpevent)
        {
            showDetails("Centre Latitude", "Latitude of the viewing centre (degrees)");
        }

        private void updCentreLongitude_HelpRequested(object sender, HelpEventArgs hlpevent)
        {
            showDetails("Centre Longitude", "Longitude of the viewing centre (degrees)");
        }

        private void updViewportHeight_HelpRequested(object sender, HelpEventArgs hlpevent)
        {
            showDetails("Viewport Height", "Vertical field of view for the viewport (degrees)");
        }

        private void updViewportWidth_HelpRequested(object sender, HelpEventArgs hlpevent)
        {
            showDetails("Viewport Width", "Horizontal field of view for the viewport (degrees)");
        }

        private void updSamplingInterval_HelpRequested(object sender, HelpEventArgs hlpevent)
        {
            showDetails("Sampling Interval", "The distance between sampling points during ray tracing");
        }

        private void updMaxSamples_HelpRequested(object sender, HelpEventArgs hlpevent)
        {
            showDetails("Maximum Samples", "The maximum number of points examined during ray tracing");
        }

        private void updBoundaryInterval_HelpRequested(object sender, HelpEventArgs hlpevent)
        {
            showDetails("Boundary Interval", "The amount that the current orbit value is sufficiently different from the last recorded sample to start a binary search for the boundary");
        }

        private void updBinarySearchSteps_HelpRequested(object sender, HelpEventArgs hlpevent)
        {
            showDetails("Binary Search Steps", "The number of steps in the binary search for an orbit value boundary");
        }

        #endregion

        private void showDetails(string title, string description)
        {
            Details form = new Details(title, description);
            form.ShowDialog(this);
        }

        private void btnApply_Click(object sender, EventArgs e)
        {
            UpdateStatus("Initialising...");
            SaveRendering();
            UpdateStatus("");
            RefreshImage();
        }

        private void chkShowSurface_CheckedChanged(object sender, EventArgs e)
        {
            updSurfaceThickness.Enabled = chkShowSurface.Checked;
            lblSurfaceThickness.Enabled = chkShowSurface.Checked;

            if (!chkShowSurface.Checked)
            {
                chkShowVolume.Checked = true;
            }
        }

        private void chkShowVolume_CheckedChanged(object sender, EventArgs e)
        {
            if (!chkShowVolume.Checked)
            {
                chkShowSurface.Checked = true;
            }
        }

        private void cmbRegion_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (isLoaded)
            {
                sphereSettings.ExposureValue[regionIndex] = (float)updExposureValue.Value;
                sphereSettings.Saturation[regionIndex] = (float)updSaturation.Value;

                regionIndex = cmbRegion.SelectedIndex;

                updExposureValue.Value = (decimal)sphereSettings.ExposureValue[regionIndex];
                updSaturation.Value = (decimal)sphereSettings.Saturation[regionIndex];

                updSurfaceContrast.Enabled = (regionIndex == 0);
                updLightingAngle.Enabled = (regionIndex == 0);
            }
        }

        private void btnApplyColour_Click(object sender, EventArgs e)
        {
            SaveRendering();
            RefreshImage();
        }

        private void updAxis_ValueChanged(object sender, EventArgs e)
        {
            if (isLoaded)
            {
                translationValues[axisIndex] = (double)updTranslate.Value;
                axisIndex = (int)updAxis.Value - 1;
                updTranslate.Value = (decimal)translationValues[axisIndex];
            }
        }

        private void updTranslate_Leave(object sender, EventArgs e)
        {
            axisIndex = (int)updAxis.Value - 1;
            translationValues[axisIndex] = (double)updTranslate.Value;
        }

        private void cmbPlane_SelectedIndexChanged_1(object sender, EventArgs e)
        {
            if (isLoaded)
            {
                rotationValues[planeIndex] = (double)updRotate.Value;
                planeIndex = cmbPlane.SelectedIndex - 1;
                updRotate.Value = (decimal)rotationValues[planeIndex];
            }
        }

        private void updRotate_Leave(object sender, EventArgs e)
        {
            planeIndex = (int)cmbPlane.SelectedIndex;
            rotationValues[planeIndex] = (double)updRotate.Value;
        }
    }
}
