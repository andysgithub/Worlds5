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

        clsSphere sphere = Model.Globals.Sphere;
        private int regionIndex = 0;
        private bool isLoaded = false;

        public SphereSettings()
        {
            InitializeComponent();
            cmbRegion.SelectedIndex = regionIndex;
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            // Sphere Viewing window
            sphere.AngularResolution = (double)updResolution.Value;
            sphere.Radius = (double)updSphereRadius.Value;
            sphere.CentreLatitude = (double)updCentreLatitude.Value;
            sphere.CentreLongitude = (double)updCentreLongitude.Value;
            sphere.VerticalView = (double)updViewportHeight.Value;
            sphere.HorizontalView = (double)updViewportWidth.Value;

            // Raytracing
            sphere.ActiveIndex = chkShowSurface.Checked ? 0 : 1;

            sphere.SamplingInterval[0] = (double)updSamplingInterval_0.Value;
            sphere.BinarySearchSteps[0] = (int)updBinarySearchSteps_0.Value;
            sphere.RayPoints[0] = (int)updRayPoints_0.Value;
            sphere.MaxSamples[0] = (int)updMaxSamples_0.Value;

            sphere.SamplingInterval[1] = (double)updSamplingInterval_1.Value;
            sphere.BinarySearchSteps[1] = (int)updBinarySearchSteps_1.Value;
            sphere.RayPoints[1] = (int)updRayPoints_1.Value;
            sphere.MaxSamples[1] = (int)updMaxSamples_1.Value;

            // Surface
            sphere.Bailout = (float)updBailout.Value;
            sphere.BoundaryInterval = (double)updBoundaryInterval.Value;
            sphere.SurfaceThickness = (double)updSurfaceThickness.Value;

            // Rendering
            SaveRendering();

            //RefreshImage();
            this.Close();
        }

        private void SaveRendering()
        {
            sphere.ExposureValue[regionIndex] = (float)updExposureValue.Value;
            sphere.Saturation[regionIndex] = (float)updSaturation.Value;
            sphere.StartDistance[regionIndex] = (double)updStartDistance.Value;
            sphere.EndDistance[regionIndex] = (double)updEndDistance.Value;

            sphere.SurfaceContrast = (float)updSurfaceContrast.Value;
            sphere.LightingAngle = (float)updLightingAngle.Value;
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            this.Close();
        }

        private void Settings_Load(object sender, EventArgs e)
        {
            isLoaded = true;

            // Sphere Viewing window
            updResolution.Value = (decimal)sphere.AngularResolution;
            updSphereRadius.Value = (decimal)sphere.Radius;
            updCentreLatitude.Value = (decimal)sphere.CentreLatitude;
            updCentreLongitude.Value = (decimal)sphere.CentreLongitude;
            updViewportHeight.Value = (decimal)sphere.VerticalView;
            updViewportWidth.Value = (decimal)sphere.HorizontalView;

            // Raytracing
            chkShowSurface.Checked = sphere.ActiveIndex == 0;
            chkShowVolume.Checked = sphere.ActiveIndex == 1;

            updSamplingInterval_0.Value = (decimal)sphere.SamplingInterval[0];
            updBinarySearchSteps_0.Value = sphere.BinarySearchSteps[0];
            updRayPoints_0.Value = sphere.RayPoints[0];
            updMaxSamples_0.Value = sphere.MaxSamples[0];

            updSamplingInterval_1.Value = (decimal)sphere.SamplingInterval[1];
            updBinarySearchSteps_1.Value = sphere.BinarySearchSteps[1];
            updRayPoints_1.Value = sphere.RayPoints[1];
            updMaxSamples_1.Value = sphere.MaxSamples[1];
            
            // Surface
            updBoundaryInterval.Value = (decimal)sphere.BoundaryInterval;
            updSurfaceThickness.Value = (decimal)sphere.SurfaceThickness;
            updBailout.Value = (decimal)sphere.Bailout;

            // Rendering
            updExposureValue.Value = (decimal)sphere.ExposureValue[regionIndex];
            updSaturation.Value = (decimal)sphere.Saturation[regionIndex];
            updStartDistance.Value = (decimal)sphere.StartDistance[regionIndex];
            updEndDistance.Value = (decimal)sphere.EndDistance[regionIndex];

            updSurfaceContrast.Value = (decimal)sphere.SurfaceContrast;
            updLightingAngle.Value = (decimal)sphere.LightingAngle;
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

        private void cmbRegion_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (isLoaded)
            {
                sphere.ExposureValue[regionIndex] = (float)updExposureValue.Value;
                sphere.Saturation[regionIndex] = (float)updSaturation.Value;
                sphere.StartDistance[regionIndex] = (double)updStartDistance.Value;
                sphere.EndDistance[regionIndex] = (double)updEndDistance.Value;

                regionIndex = cmbRegion.SelectedIndex;

                updExposureValue.Value = (decimal)sphere.ExposureValue[regionIndex];
                updSaturation.Value = (decimal)sphere.Saturation[regionIndex];
                updStartDistance.Value = (decimal)sphere.StartDistance[regionIndex];
                updEndDistance.Value = (decimal)sphere.EndDistance[regionIndex];

                updSurfaceContrast.Enabled = (regionIndex == 0);
                updLightingAngle.Enabled = (regionIndex == 0);
                updStartDistance.Enabled = (regionIndex < 2);
                updEndDistance.Enabled = (regionIndex < 2);
            }
        }
    }
}
