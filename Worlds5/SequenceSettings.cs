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
            this.updRadiusStart.Value = (decimal)sphere.settings.SphereRadius;
            this.updRadiusEnd.Value = (decimal)sphere.settings.SphereRadius;
        }

        private void btnStart_Click(object sender, EventArgs e)
        {
            this.Hide();
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            this.Close();
        }

        public float[] CentreCoords
        {
            get
            {
                float[] coords = new float[5];
                coords[0] = Single.Parse(txtCoord1.Text);
                coords[1] = Single.Parse(txtCoord2.Text);
                coords[2] = Single.Parse(txtCoord3.Text);
                coords[3] = Single.Parse(txtCoord4.Text);
                coords[4] = Single.Parse(txtCoord5.Text);
                return coords;
            }
        }

        public string BaseName
        {
            get { return txtBaseName.Text; }
        }

        public float[,] Angles
        {
            get
            {
                // Factor to convert total degrees into radians per frame
                float factor = Globals.DEG_TO_RAD / Stages;
                float[,] angles = new float[5, 6];

                angles[1, 2] = Single.Parse(txtDegrees12.Text) * factor;
                angles[1, 3] = Single.Parse(txtDegrees13.Text) * factor;
                angles[1, 4] = Single.Parse(txtDegrees14.Text) * factor;
                angles[1, 5] = Single.Parse(txtDegrees15.Text) * factor;
                angles[2, 3] = Single.Parse(txtDegrees23.Text) * factor;
                angles[2, 4] = Single.Parse(txtDegrees24.Text) * factor;
                angles[2, 5] = Single.Parse(txtDegrees25.Text) * factor;
                angles[3, 4] = Single.Parse(txtDegrees34.Text) * factor;
                angles[3, 5] = Single.Parse(txtDegrees35.Text) * factor;
                angles[4, 5] = Single.Parse(txtDegrees45.Text) * factor;

                return angles;
            }
        }

        public float[] SphereRadius
        {
            get
            {
                float[] sphereRadius = new float[2];
                sphereRadius[0] = (float)updRadiusStart.Value;
                sphereRadius[1] = (float)updRadiusEnd.Value;
                return sphereRadius;
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
