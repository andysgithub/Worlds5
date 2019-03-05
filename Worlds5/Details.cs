using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace Worlds5
{
    public partial class Details : Form
    {
        public Details(string Title, string Description)
        {
            InitializeComponent();

            lblTitle.Text = Title;
            lblDescription.Text = Description;
        }

        private void btnClose_Click(object sender, EventArgs e)
        {
            this.Close();
        }
    }
}
