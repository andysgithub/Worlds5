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
    public partial class Settings : Form
    {
        public Settings()
        {
            InitializeComponent();
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            clsSphere sphere = Model.Globals.Sphere;

            sphere.AngularResolution = (double)updResolution.Value;
            sphere.Radius = (double)updSphereRadius.Value;
            sphere.CentreLatitude = (double)updCentreLatitude.Value;

            // Sphere Viewing window
            sphere.AngularResolution = (double)updResolution.Value;
            sphere.Radius = (double)updSphereRadius.Value;
            sphere.CentreLatitude = (double)updCentreLatitude.Value;
            sphere.CentreLongitude = (double)updCentreLongitude.Value;
            sphere.VerticalView = (double)updViewportHeight.Value;
            sphere.HorizontalView = (double)updViewportWidth.Value;

            // Raytracing
            sphere.SamplingInterval = (double)updSamplingInterval.Value;
            sphere.RayPoints = (int)updRayPoints.Value;
            sphere.MaxSamples = (int)updMaxSamples.Value;
            sphere.BoundaryInterval = (double)updBoundaryInterval.Value;
            sphere.BinarySearchSteps = (int)updBinarySearchSteps.Value;
            sphere.SurfaceThickness = (double)updSurfaceThickness.Value;
            sphere.ShowSurface = chkShowSurface.Checked;
            sphere.ShowExterior = chkShowExterior.Checked;

            // Rendering
            SaveRendering();

            this.Close();
        }

        private void SaveRendering()
        {
            clsSphere sphere = Model.Globals.Sphere;

            Globals.SetUp.ImageJpgQuality = (int)updImageJpgQuality.Value;
            Globals.SetUp.ImageFileFormat = (int)cmbImageFileFormat.SelectedIndex;
            Globals.SetUp.BitmapWidth = (int)updHorizontalView.Value;
            Globals.SetUp.BitmapHeight = (int)updVerticalView.Value;
            sphere.ExposureValue = (float)updExposureValue.Value;
            sphere.Saturation = (float)updSaturation.Value;
            sphere.SurfaceContrast = (float)updSurfaceContrast.Value;
            sphere.LightingAngle = (float)updLightingAngle.Value;
            sphere.StartDistance = (double)updStartDistance.Value;
            sphere.EndDistance = (double)updEndDistance.Value;
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            this.Close();
        }

        private void Settings_Load(object sender, EventArgs e)
        {
            clsSphere sphere = Model.Globals.Sphere;

            updResolution.Value = (decimal)sphere.AngularResolution;
            updSphereRadius.Value = (decimal)sphere.Radius;
            updCentreLatitude.Value = (decimal)sphere.CentreLatitude;

            // Sphere Viewing window
            updResolution.Value = (decimal)sphere.AngularResolution;
            updSphereRadius.Value = (decimal)sphere.Radius;
            updCentreLatitude.Value = (decimal)sphere.CentreLatitude;
            updCentreLongitude.Value = (decimal)sphere.CentreLongitude;
            updViewportHeight.Value = (decimal)sphere.VerticalView;
            updViewportWidth.Value = (decimal)sphere.HorizontalView;

            // Raytracing
            updSamplingInterval.Value = (decimal)sphere.SamplingInterval;
            updSurfaceThickness.Value = (decimal)sphere.SurfaceThickness;
            updRayPoints.Value = sphere.RayPoints;
            updMaxSamples.Value = sphere.MaxSamples;
            updBoundaryInterval.Value = (decimal)sphere.BoundaryInterval;
            updBinarySearchSteps.Value = (decimal)sphere.BinarySearchSteps;
            chkShowSurface.Checked = sphere.ShowSurface;
            chkShowExterior.Checked = sphere.ShowExterior;

            // Rendering
            updImageJpgQuality.Value = Globals.SetUp.ImageJpgQuality;
            cmbImageFileFormat.SelectedIndex = Globals.SetUp.ImageFileFormat;
            updHorizontalView.Value = Globals.SetUp.BitmapWidth;
            updVerticalView.Value = Globals.SetUp.BitmapHeight;
            updExposureValue.Value = (decimal)sphere.ExposureValue;
            updSaturation.Value = (decimal)sphere.Saturation;
            updSurfaceContrast.Value = (decimal)sphere.SurfaceContrast;
            updLightingAngle.Value = (decimal)sphere.LightingAngle;
            updStartDistance.Value = (decimal)sphere.StartDistance;
            updEndDistance.Value = (decimal)sphere.EndDistance;
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

        private void updVerticalView_HelpRequested(object sender, HelpEventArgs hlpevent)
        {
            showDetails("Vertical View", "");
        }

        private void updHorizontalView_HelpRequested(object sender, HelpEventArgs hlpevent)
        {
            showDetails("Horizontal View", "");
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
            //((Main)this.Owner).Redisplay();
        }

        private void chkShowSurface_CheckedChanged(object sender, EventArgs e)
        {
            updSurfaceThickness.Enabled = chkShowSurface.Checked;
            lblSurfaceThickness.Enabled = chkShowSurface.Checked;

            if (!chkShowSurface.Checked)
            {
                chkShowExterior.Checked = true;
            }
        }

        private void chkShowExterior_CheckedChanged(object sender, EventArgs e)
        {
            if (!chkShowExterior.Checked)
            {
                chkShowSurface.Checked = true;
            }
        }
    }
}
