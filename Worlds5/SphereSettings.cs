using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using Model;

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
            clsSphere sphere = Model.Globals.Sphere;

            sphere.ExposureValue[0] = (float)updExposureValue_0.Value;
            sphere.Saturation[0] = (float)updSaturation_0.Value;
            sphere.StartDistance[0] = (double)updStartDistance_0.Value;
            sphere.EndDistance[0] = (double)updEndDistance_0.Value;

            sphere.ExposureValue[1] = (float)updExposureValue_1.Value;
            sphere.Saturation[1] = (float)updSaturation_1.Value;
            sphere.StartDistance[1] = (double)updStartDistance_1.Value;
            sphere.EndDistance[1] = (double)updEndDistance_1.Value;

            sphere.SurfaceContrast = (float)updSurfaceContrast.Value;
            sphere.LightingAngle = (float)updLightingAngle.Value;
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
            updExposureValue_0.Value = (decimal)sphere.ExposureValue[0];
            updSaturation_0.Value = (decimal)sphere.Saturation[0];
            updStartDistance_0.Value = (decimal)sphere.StartDistance[0];
            updEndDistance_0.Value = (decimal)sphere.EndDistance[0];

            updExposureValue_1.Value = (decimal)sphere.ExposureValue[1];
            updSaturation_1.Value = (decimal)sphere.Saturation[1];
            updStartDistance_1.Value = (decimal)sphere.StartDistance[1];
            updEndDistance_1.Value = (decimal)sphere.EndDistance[1];

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
    }
}
