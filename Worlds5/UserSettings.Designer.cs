namespace Worlds5
{
    partial class UserSettings
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.chkToolbar = new System.Windows.Forms.CheckBox();
            this.chkLabels = new System.Windows.Forms.CheckBox();
            this.chkStatusBar = new System.Windows.Forms.CheckBox();
            this.grpMainWindow = new System.Windows.Forms.GroupBox();
            this.chkTooltips = new System.Windows.Forms.CheckBox();
            this.txtNavigationPath = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.btnNavigationPath = new System.Windows.Forms.Button();
            this.grpFilePaths = new System.Windows.Forms.GroupBox();
            this.btnOK = new System.Windows.Forms.Button();
            this.btnCancel = new System.Windows.Forms.Button();
            this.btnSequencePath = new System.Windows.Forms.Button();
            this.label2 = new System.Windows.Forms.Label();
            this.txtSequencePath = new System.Windows.Forms.TextBox();
            this.grpMainWindow.SuspendLayout();
            this.grpFilePaths.SuspendLayout();
            this.SuspendLayout();
            // 
            // chkToolbar
            // 
            this.chkToolbar.AutoSize = true;
            this.chkToolbar.Location = new System.Drawing.Point(19, 30);
            this.chkToolbar.Name = "chkToolbar";
            this.chkToolbar.Size = new System.Drawing.Size(62, 17);
            this.chkToolbar.TabIndex = 0;
            this.chkToolbar.Text = "Toolbar";
            this.chkToolbar.UseVisualStyleBackColor = true;
            // 
            // chkLabels
            // 
            this.chkLabels.AutoSize = true;
            this.chkLabels.Location = new System.Drawing.Point(19, 53);
            this.chkLabels.Name = "chkLabels";
            this.chkLabels.Size = new System.Drawing.Size(57, 17);
            this.chkLabels.TabIndex = 1;
            this.chkLabels.Text = "Labels";
            this.chkLabels.UseVisualStyleBackColor = true;
            // 
            // chkStatusBar
            // 
            this.chkStatusBar.AutoSize = true;
            this.chkStatusBar.Location = new System.Drawing.Point(106, 53);
            this.chkStatusBar.Name = "chkStatusBar";
            this.chkStatusBar.Size = new System.Drawing.Size(74, 17);
            this.chkStatusBar.TabIndex = 3;
            this.chkStatusBar.Text = "Status bar";
            this.chkStatusBar.UseVisualStyleBackColor = true;
            // 
            // grpMainWindow
            // 
            this.grpMainWindow.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.grpMainWindow.Controls.Add(this.chkStatusBar);
            this.grpMainWindow.Controls.Add(this.chkTooltips);
            this.grpMainWindow.Controls.Add(this.chkLabels);
            this.grpMainWindow.Controls.Add(this.chkToolbar);
            this.grpMainWindow.Location = new System.Drawing.Point(14, 169);
            this.grpMainWindow.Name = "grpMainWindow";
            this.grpMainWindow.Size = new System.Drawing.Size(196, 87);
            this.grpMainWindow.TabIndex = 5;
            this.grpMainWindow.TabStop = false;
            this.grpMainWindow.Text = "Main Window";
            // 
            // chkTooltips
            // 
            this.chkTooltips.AutoSize = true;
            this.chkTooltips.Location = new System.Drawing.Point(106, 30);
            this.chkTooltips.Name = "chkTooltips";
            this.chkTooltips.Size = new System.Drawing.Size(63, 17);
            this.chkTooltips.TabIndex = 2;
            this.chkTooltips.Text = "Tooltips";
            this.chkTooltips.UseVisualStyleBackColor = true;
            // 
            // txtNavigationPath
            // 
            this.txtNavigationPath.Location = new System.Drawing.Point(14, 47);
            this.txtNavigationPath.Name = "txtNavigationPath";
            this.txtNavigationPath.Size = new System.Drawing.Size(238, 20);
            this.txtNavigationPath.TabIndex = 0;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(12, 29);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(82, 13);
            this.label1.TabIndex = 1;
            this.label1.Text = "Nagivation path";
            // 
            // btnNavigationPath
            // 
            this.btnNavigationPath.Location = new System.Drawing.Point(258, 47);
            this.btnNavigationPath.Name = "btnNavigationPath";
            this.btnNavigationPath.Size = new System.Drawing.Size(64, 20);
            this.btnNavigationPath.TabIndex = 2;
            this.btnNavigationPath.Text = "Browse...";
            this.btnNavigationPath.UseVisualStyleBackColor = true;
            this.btnNavigationPath.Click += new System.EventHandler(this.btnNavigationPath_Click);
            // 
            // grpFilePaths
            // 
            this.grpFilePaths.Controls.Add(this.btnSequencePath);
            this.grpFilePaths.Controls.Add(this.label2);
            this.grpFilePaths.Controls.Add(this.txtSequencePath);
            this.grpFilePaths.Controls.Add(this.btnNavigationPath);
            this.grpFilePaths.Controls.Add(this.label1);
            this.grpFilePaths.Controls.Add(this.txtNavigationPath);
            this.grpFilePaths.Location = new System.Drawing.Point(12, 12);
            this.grpFilePaths.Name = "grpFilePaths";
            this.grpFilePaths.Size = new System.Drawing.Size(334, 140);
            this.grpFilePaths.TabIndex = 4;
            this.grpFilePaths.TabStop = false;
            this.grpFilePaths.Text = "File Paths";
            // 
            // btnOK
            // 
            this.btnOK.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.btnOK.Location = new System.Drawing.Point(274, 199);
            this.btnOK.Name = "btnOK";
            this.btnOK.Size = new System.Drawing.Size(75, 23);
            this.btnOK.TabIndex = 2;
            this.btnOK.Text = "OK";
            this.btnOK.UseVisualStyleBackColor = true;
            this.btnOK.Click += new System.EventHandler(this.btnOK_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.btnCancel.Location = new System.Drawing.Point(274, 233);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(75, 23);
            this.btnCancel.TabIndex = 3;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.UseVisualStyleBackColor = true;
            this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
            // 
            // btnSequencePath
            // 
            this.btnSequencePath.Location = new System.Drawing.Point(258, 97);
            this.btnSequencePath.Name = "btnSequencePath";
            this.btnSequencePath.Size = new System.Drawing.Size(64, 20);
            this.btnSequencePath.TabIndex = 5;
            this.btnSequencePath.Text = "Browse...";
            this.btnSequencePath.UseVisualStyleBackColor = true;
            this.btnSequencePath.Click += new System.EventHandler(this.btnSequencePath_Click);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(12, 79);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(80, 13);
            this.label2.TabIndex = 4;
            this.label2.Text = "Sequence path";
            // 
            // txtSequencePath
            // 
            this.txtSequencePath.Location = new System.Drawing.Point(14, 97);
            this.txtSequencePath.Name = "txtSequencePath";
            this.txtSequencePath.Size = new System.Drawing.Size(238, 20);
            this.txtSequencePath.TabIndex = 3;
            // 
            // UserSettings
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(361, 274);
            this.Controls.Add(this.grpMainWindow);
            this.Controls.Add(this.grpFilePaths);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnOK);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "UserSettings";
            this.Text = "UserSettings";
            this.Load += new System.EventHandler(this.UserSettings_Load);
            this.grpMainWindow.ResumeLayout(false);
            this.grpMainWindow.PerformLayout();
            this.grpFilePaths.ResumeLayout(false);
            this.grpFilePaths.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion
        private System.Windows.Forms.CheckBox chkToolbar;
        private System.Windows.Forms.CheckBox chkLabels;
        private System.Windows.Forms.CheckBox chkStatusBar;
        private System.Windows.Forms.GroupBox grpMainWindow;
        private System.Windows.Forms.CheckBox chkTooltips;
        private System.Windows.Forms.TextBox txtNavigationPath;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Button btnNavigationPath;
        private System.Windows.Forms.GroupBox grpFilePaths;
        private System.Windows.Forms.Button btnOK;
        private System.Windows.Forms.Button btnCancel;
        private System.Windows.Forms.Button btnSequencePath;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox txtSequencePath;
    }
}