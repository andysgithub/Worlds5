using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Imaging;
using System.Text;
using System.Windows.Forms;

namespace Worlds5
{
    public partial class SequenceSettings : Form
    {
        public SequenceSettings(string defaultName)
        {
            InitializeComponent();
            this.txtBaseName.Text = defaultName;
            this.cmbFormat.SelectedIndex = 0;

            Model.clsSphere sphere = Model.Globals.Sphere;
            this.updRadiusStart.Value = (decimal)sphere.settings.Radius;
            this.updRadiusEnd.Value = (decimal)sphere.settings.Radius;
        }

        private void btnStart_Click(object sender, EventArgs e)
        {
            this.Hide();
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            this.Close();
        }

        public double[] CentreCoords
        {
            get
            {
                double[] coords = new double[5];
                coords[0] = Double.Parse(txtCoord1.Text);
                coords[1] = Double.Parse(txtCoord2.Text);
                coords[2] = Double.Parse(txtCoord3.Text);
                coords[3] = Double.Parse(txtCoord4.Text);
                coords[4] = Double.Parse(txtCoord5.Text);
                return coords;
            }
        }

        public string BaseName
        {
            get { return txtBaseName.Text; }
        }

        public double[,] Angles
        {
            get
            {
                // Factor to convert total degrees into radians per frame
                double factor = Globals.DEG_TO_RAD / Stages;
                double[,] angles = new double[5, 6];

                angles[1, 2] = Double.Parse(txtDegrees12.Text) * factor;
                angles[1, 3] = Double.Parse(txtDegrees13.Text) * factor;
                angles[1, 4] = Double.Parse(txtDegrees14.Text) * factor;
                angles[1, 5] = Double.Parse(txtDegrees15.Text) * factor;
                angles[2, 3] = Double.Parse(txtDegrees23.Text) * factor;
                angles[2, 4] = Double.Parse(txtDegrees24.Text) * factor;
                angles[2, 5] = Double.Parse(txtDegrees25.Text) * factor;
                angles[3, 4] = Double.Parse(txtDegrees34.Text) * factor;
                angles[3, 5] = Double.Parse(txtDegrees35.Text) * factor;
                angles[4, 5] = Double.Parse(txtDegrees45.Text) * factor;

                return angles;
            }
        }

        public double[] SphereRadius
        {
            get
            {
                double[] radius = new double[2];
                radius[0] = (double)updRadiusStart.Value;
                radius[1] = (double)updRadiusEnd.Value;
                return radius;
            }
        }

        public int Stages
        {
            get { return int.Parse(txtStages.Text); }
        }

        public ImageFormat Format
        {
            get
            {
                switch (cmbFormat.SelectedIndex)
                {
                    case 1:
                        return ImageFormat.Tiff;
                    case 2:
                        return ImageFormat.Bmp;
                    case 3:
                        return ImageFormat.Png;
                    default:
                        return ImageFormat.Jpeg;
                }
            }
        }

        private void textBox5_TextChanged(object sender, EventArgs e)
        {

        }
    }
}
