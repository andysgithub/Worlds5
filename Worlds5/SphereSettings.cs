using Model;
using System;
using System.Windows.Forms;
using System.Threading.Tasks;
using System.Reflection;

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
        private int navPlaneIndex = 0;
        private int clipPlaneIndex = 0;
        private int axisIndex = 0;
        private bool isLoaded = false;
        private float[] translationValues = new float[5];
        private float[] navRotationValues = new float[10];
        private float[] clipRotationValues = new float[10];

        public SphereSettings()
        {
            InitializeComponent();
            cmbRegion.SelectedIndex = regionIndex;
            cmbNavPlane.SelectedIndex = navPlaneIndex;
            cmbClipPlane.SelectedIndex = clipPlaneIndex;
            cmbNavCentre.SelectedIndex = 0;
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
            enableButtons(false);
            UpdateStatus("Initialising...");
            SaveSettings();
            SaveRendering();
            UpdateStatus("");
            await RaytraceImage();
            enableButtons(true);
        }

        private float[,] NavAngles
        {
            get
            {
                // Factor to convert total degrees into radians
                float factor = Globals.DEG_TO_RAD;
                float[,] angles = new float[5, 6];

                angles[1, 2] = navRotationValues[0] * factor;
                angles[1, 3] = navRotationValues[1] * factor;
                angles[1, 4] = navRotationValues[2] * factor;
                angles[1, 5] = navRotationValues[3] * factor;
                angles[2, 3] = navRotationValues[4] * factor;
                angles[2, 4] = navRotationValues[5] * factor;
                angles[2, 5] = navRotationValues[6] * factor;
                angles[3, 4] = navRotationValues[7] * factor;
                angles[3, 5] = navRotationValues[8] * factor;
                angles[4, 5] = navRotationValues[9] * factor;

                return angles;
            }
        }

        private void SaveSettings()
        {
            // Transfer rotation values to position matrix
            Transformation.RotateImage((RotationCentre)cmbNavCentre.SelectedIndex, NavAngles);
            sphereSettings.PositionMatrix = Transformation.GetPositionMatrix();
/*
            // Clear the rotation input
            updNavRotate.Value = 0;
            // Clear the rotation values
            for (int i = 0; i < 10; ++i)
            {
                navRotationValues[i] = 0;
            }
*/
            // Sphere viewing window
            sphereSettings.AngularResolution = (float)updResolution.Value;
            sphereSettings.Radius = (float)updSphereRadius.Value;
            sphereSettings.CentreLatitude = (float)updCentreLatitude.Value;
            sphereSettings.CentreLongitude = (float)updCentreLongitude.Value;
            sphereSettings.VerticalView = (float)updViewportHeight.Value;
            sphereSettings.HorizontalView = (float)updViewportWidth.Value;

            // Position
            for (int axis = 0; axis < 5; axis++)
            {
                sphereSettings.PositionMatrix[5, axis] += translationValues[axis];
                //translationValues[axis] = 0;
            }
            /*            // Clear the translation input
                        updTranslate.Value = 0;*/

            // Raytracing
            sphereSettings.ActiveIndex = chkShowSurface.Checked ? 0 : 1;
            sphereSettings.CudaMode = chkCudaMode.Checked;

            sphereSettings.SamplingInterval[0] = (float)updSamplingInterval_0.Value;
            sphereSettings.BinarySearchSteps[0] = (int)updBinarySearchSteps_0.Value;
            sphereSettings.MaxSamples[0] = (int)updMaxSamples_0.Value;

            sphereSettings.SamplingInterval[1] = (float)updSamplingInterval_1.Value;
            sphereSettings.BinarySearchSteps[1] = (int)updBinarySearchSteps_1.Value;
            sphereSettings.MaxSamples[1] = (int)updMaxSamples_1.Value;

            // Clipping
            sphereSettings.ClippingAxes = Clipping.GetAxes(cmbClipPlane.SelectedIndex);
            sphereSettings.ClippingOffset = (float)updClipOffset.Value;
            sphereSettings.UseClipping = chkUseClipping.Checked;

            // Surface
            sphereSettings.Bailout = (float)updBailout.Value;
            sphereSettings.BoundaryInterval = (float)updBoundaryInterval.Value;
            sphereSettings.SurfaceSmoothing = (float)updSurfaceSmoothing.Value;
            sphereSettings.SurfaceThickness = (float)updSurfaceThickness.Value;

/*            // Clear the rotation and offset inputs
            updClipRotate.Value = 0;
            updClipOffset.Value = 0;*/
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

            chkCudaMode.Checked = sphereSettings.CudaMode;

            updSamplingInterval_0.Value = (decimal)sphereSettings.SamplingInterval[0];
            updBinarySearchSteps_0.Value = sphereSettings.BinarySearchSteps[0];
            updMaxSamples_0.Value = sphereSettings.MaxSamples[0];

            updSamplingInterval_1.Value = (decimal)sphereSettings.SamplingInterval[1];
            updBinarySearchSteps_1.Value = sphereSettings.BinarySearchSteps[1];
            updMaxSamples_1.Value = sphereSettings.MaxSamples[1];

            // Clipping
            chkUseClipping.Checked = sphereSettings.UseClipping;

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

        private void enableButtons(bool state)
        {
            btnApply.Enabled = state;
            btnApplyColour.Enabled = state;
            btnRaytrace.Enabled = state;
        }

        private void btnApply_Click(object sender, EventArgs e)
        {
            enableButtons(false);
            SaveRendering();
            RefreshImage();
            enableButtons(true);
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

        private void updAxis_ValueChanged(object sender, EventArgs e)
        {
            if (isLoaded)
            {
                translationValues[axisIndex] = (float)updTranslate.Value;
                axisIndex = (int)updAxis.Value - 1;
                updTranslate.Value = (decimal)translationValues[axisIndex];
            }
        }

        private void updTranslate_Leave(object sender, EventArgs e)
        {
            axisIndex = (int)updAxis.Value - 1;
            translationValues[axisIndex] = (float)updTranslate.Value;
        }

        private void cmbNavPlane_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (isLoaded)
            {
                navRotationValues[navPlaneIndex] = (float)updNavRotate.Value;
                navPlaneIndex = cmbNavPlane.SelectedIndex;
                updNavRotate.Value = (decimal)navRotationValues[navPlaneIndex];
            }
        }

        private void updNavRotate_Leave(object sender, EventArgs e)
        {
            navPlaneIndex = (int)cmbNavPlane.SelectedIndex;
            navRotationValues[navPlaneIndex] = (float)updNavRotate.Value;
        }

        private void chkUseClipping_CheckedChanged(object sender, EventArgs e)
        {
            grpClipping.Enabled = chkUseClipping.Checked;
        }

        private void cmbClipPlane_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (isLoaded)
            {
                clipRotationValues[clipPlaneIndex] = (float)updClipRotate.Value;
                clipPlaneIndex = cmbClipPlane.SelectedIndex;
                updClipRotate.Value = (decimal)clipRotationValues[clipPlaneIndex];
            }
        }

        private void updClipRotate_Leave(object sender, EventArgs e)
        {
            clipPlaneIndex = (int)cmbClipPlane.SelectedIndex;
            clipRotationValues[clipPlaneIndex] = (float)updClipRotate.Value;
        }

        private void btnClearNav_Click(object sender, EventArgs e)
        {
            // Clear sphere settings
            for (int axisIndex = 0; axisIndex < 5; axisIndex++)
            { 
                translationValues[axisIndex] = 0.0F;
            }
            for (int planeIndex = 0; planeIndex < 10; planeIndex++)
            {
                navRotationValues[planeIndex] = 0.0F;
            }
            axisIndex = 0;
            navPlaneIndex = 0;

            // Clear navigation controls
            updTranslate.Value = 0;
            updNavRotate.Value = 0;
            cmbNavPlane.SelectedIndex = 0;
            cmbNavCentre.SelectedIndex = (int)RotationCentre.Sphere;
            updAxis.Value = 1;
        }

        private void btnClearClipping_Click(object sender, EventArgs e)
        {
            // Clear sphere settings
            for (int planeIndex = 0; planeIndex < 10; planeIndex++)
            {
                clipRotationValues[planeIndex] = 0.0F;
            }
            sphereSettings.ClippingOffset = 0.0F;

            // Clear clipping controls
            cmbClipPlane.SelectedIndex = 0;
            updClipRotate.Value = 0;
            updClipOffset.Value = 0;
            chkUseClipping.Checked = false;
        }

        private void chkCudaMode_CheckedChanged(object sender, EventArgs e)
        {

        }
    }
}
