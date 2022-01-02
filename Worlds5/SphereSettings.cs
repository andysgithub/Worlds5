using Model;
using System;
using System.Windows.Forms;

namespace Worlds5
{
    public partial class SphereSettings : Form
    {
        #region Delegates

        public delegate void RefreshDelegate();
        public event RefreshDelegate RefreshImage;

        #endregion

        public SphereSettings()
        {
            InitializeComponent();
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            clsSphere sphere = Model.Globals.Sphere;

            // Sphere Viewing window
            sphere.AngularResolution = (double)updResolution.Value;
            sphere.Radius = (double)updSphereRadius.Value;
            sphere.CentreLatitude = (double)updCentreLatitude.Value;
            sphere.CentreLongitude = (double)updCentreLongitude.Value;
            sphere.VerticalView = (double)updViewportHeight.Value;
            sphere.HorizontalView = (double)updViewportWidth.Value;

            // Raytracing
            sphere.SamplingInterval[0] = (double)updSamplingInterval.Value;
            sphere.RayPoints[0] = (int)updRayPoints.Value;
            sphere.MaxSamples[0] = (int)updMaxSamples.Value;
            sphere.BinarySearchSteps[0] = (int)updBinarySearchSteps.Value;
            sphere.BoundaryInterval = (double)updBoundaryInterval.Value;
            sphere.SurfaceThickness = (double)updSurfaceThickness.Value;
            sphere.ActiveIndex = chkShowSurface.Checked ? 0 : 1;

            // Rendering
            SaveRendering();

            RefreshImage();
            this.Close();
        }

        private void SaveRendering()
        {
            clsSphere sphere = Model.Globals.Sphere;

            sphere.ExposureValue[0] = (float)updExposureValue.Value;
            sphere.Saturation[0] = (float)updSaturation.Value;
            sphere.StartDistance[0] = (double)updStartDistance.Value;
            sphere.EndDistance[0] = (double)updEndDistance.Value;
            sphere.SurfaceContrast = (float)updSurfaceContrast.Value;
            sphere.LightingAngle = (float)updLightingAngle.Value;
            sphere.InteriorExposure = (float)updInteriorExposure.Value;
            sphere.InteriorSaturation = (float)updInteriorSaturation.Value;
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            this.Close();
        }

        private void Settings_Load(object sender, EventArgs e)
        {
            clsSphere sphere = Model.Globals.Sphere;

            // Sphere Viewing window
            updResolution.Value = (decimal)sphere.AngularResolution;
            updSphereRadius.Value = (decimal)sphere.Radius;
            updCentreLatitude.Value = (decimal)sphere.CentreLatitude;
            updCentreLongitude.Value = (decimal)sphere.CentreLongitude;
            updViewportHeight.Value = (decimal)sphere.VerticalView;
            updViewportWidth.Value = (decimal)sphere.HorizontalView;

            // Raytracing
            updSamplingInterval.Value = (decimal)sphere.SamplingInterval[0];
            updSurfaceThickness.Value = (decimal)sphere.SurfaceThickness;
            updRayPoints.Value = sphere.RayPoints[0];
            updMaxSamples.Value = sphere.MaxSamples[0];
            updBoundaryInterval.Value = (decimal)sphere.BoundaryInterval;
            updBinarySearchSteps.Value = (decimal)sphere.BinarySearchSteps[0];
            chkShowSurface.Checked = sphere.ActiveIndex == 0;
            chkShowVolume.Checked = sphere.ActiveIndex == 1;

            // Rendering
            updExposureValue.Value = (decimal)sphere.ExposureValue[0];
            updSaturation.Value = (decimal)sphere.Saturation[0];
            updStartDistance.Value = (decimal)sphere.StartDistance[0];
            updEndDistance.Value = (decimal)sphere.EndDistance[0];
            updSurfaceContrast.Value = (decimal)sphere.SurfaceContrast;
            updLightingAngle.Value = (decimal)sphere.LightingAngle;
            updInteriorExposure.Value = (decimal)sphere.InteriorExposure;
            updInteriorSaturation.Value = (decimal)sphere.InteriorSaturation;
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

        private void updImageJpgQuality_HelpRequested(object sender, HelpEventArgs hlpevent)
        {
            showDetails("Image Jpeg Quality", "");
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
            showDetails("Sphere Radius", "The distance from the centre of the sphere to first ray tracing point");
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

        private void updRayPoints_HelpRequested(object sender, HelpEventArgs hlpevent)
        {
            showDetails("Ray Points", "The number of boundary points recorded during ray tracing");
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
            SaveRendering();
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

        private void label13_Click(object sender, EventArgs e)
        {

        }

        private void label12_Click(object sender, EventArgs e)
        {

        }

        private void numericUpDown1_ValueChanged(object sender, EventArgs e)
        {

        }

        private void numericUpDown2_ValueChanged(object sender, EventArgs e)
        {

        }
    }
}
