namespace Worlds5
{
    partial class Main
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
            this.components = new System.ComponentModel.Container();
            this.menuStrip1 = new System.Windows.Forms.MenuStrip();
            this.mnuFile = new System.Windows.Forms.ToolStripMenuItem();
            this.mnuLoadSphere = new System.Windows.Forms.ToolStripMenuItem();
            this.mnuSaveSphere = new System.Windows.Forms.ToolStripMenuItem();
            this.mnuSaveImage = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItem1 = new System.Windows.Forms.ToolStripSeparator();
            this.mnuClose = new System.Windows.Forms.ToolStripMenuItem();
            this.settingsToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.mnuSettings = new System.Windows.Forms.ToolStripMenuItem();
            this.mnuRotation = new System.Windows.Forms.ToolStripMenuItem();
            this.viewToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.mnuRefresh = new System.Windows.Forms.ToolStripMenuItem();
            this.mnuReset = new System.Windows.Forms.ToolStripMenuItem();
            this.sequenceToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.mnuVideoConversion = new System.Windows.Forms.ToolStripMenuItem();
            this.staStatus = new System.Windows.Forms.StatusStrip();
            this.lblPatchStatus = new System.Windows.Forms.ToolStripStatusLabel();
            this.lblTileStatus = new System.Windows.Forms.ToolStripStatusLabel();
            this.picImage = new System.Windows.Forms.PictureBox();
            this.dlgOpenFile = new System.Windows.Forms.OpenFileDialog();
            this.pnlImage = new System.Windows.Forms.Panel();
            this.tmrRedraw = new System.Windows.Forms.Timer(this.components);
            this.menuStrip1.SuspendLayout();
            this.staStatus.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.picImage)).BeginInit();
            this.pnlImage.SuspendLayout();
            this.SuspendLayout();
            // 
            // menuStrip1
            // 
            this.menuStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.mnuFile,
            this.settingsToolStripMenuItem,
            this.viewToolStripMenuItem,
            this.sequenceToolStripMenuItem});
            this.menuStrip1.Location = new System.Drawing.Point(0, 0);
            this.menuStrip1.Name = "menuStrip1";
            this.menuStrip1.Size = new System.Drawing.Size(723, 24);
            this.menuStrip1.TabIndex = 0;
            this.menuStrip1.Text = "menuStrip1";
            // 
            // mnuFile
            // 
            this.mnuFile.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.mnuLoadSphere,
            this.mnuSaveSphere,
            this.mnuSaveImage,
            this.toolStripMenuItem1,
            this.mnuClose});
            this.mnuFile.Name = "mnuFile";
            this.mnuFile.Size = new System.Drawing.Size(37, 20);
            this.mnuFile.Text = "File";
            // 
            // mnuLoadSphere
            // 
            this.mnuLoadSphere.Name = "mnuLoadSphere";
            this.mnuLoadSphere.Size = new System.Drawing.Size(180, 22);
            this.mnuLoadSphere.Text = "Load Sphere...";
            this.mnuLoadSphere.Click += new System.EventHandler(this.mnuLoadSphere_Click);
            // 
            // mnuSaveSphere
            // 
            this.mnuSaveSphere.Name = "mnuSaveSphere";
            this.mnuSaveSphere.Size = new System.Drawing.Size(180, 22);
            this.mnuSaveSphere.Text = "Save Sphere...";
            this.mnuSaveSphere.Click += new System.EventHandler(this.mnuSaveSphere_Click);
            // 
            // mnuSaveImage
            // 
            this.mnuSaveImage.Name = "mnuSaveImage";
            this.mnuSaveImage.Size = new System.Drawing.Size(180, 22);
            this.mnuSaveImage.Text = "Save Image...";
            this.mnuSaveImage.Click += new System.EventHandler(this.mnuSaveImage_Click);
            // 
            // toolStripMenuItem1
            // 
            this.toolStripMenuItem1.Name = "toolStripMenuItem1";
            this.toolStripMenuItem1.Size = new System.Drawing.Size(177, 6);
            // 
            // mnuClose
            // 
            this.mnuClose.Name = "mnuClose";
            this.mnuClose.Size = new System.Drawing.Size(180, 22);
            this.mnuClose.Text = "Close";
            this.mnuClose.Click += new System.EventHandler(this.mnuClose_Click);
            // 
            // settingsToolStripMenuItem
            // 
            this.settingsToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.mnuSettings,
            this.mnuRotation});
            this.settingsToolStripMenuItem.Name = "settingsToolStripMenuItem";
            this.settingsToolStripMenuItem.Size = new System.Drawing.Size(47, 20);
            this.settingsToolStripMenuItem.Text = "Tools";
            this.settingsToolStripMenuItem.DropDownOpening += new System.EventHandler(this.settingsToolStripMenuItem_DropDownOpening);
            // 
            // mnuSettings
            // 
            this.mnuSettings.Name = "mnuSettings";
            this.mnuSettings.Size = new System.Drawing.Size(128, 22);
            this.mnuSettings.Text = "Settings...";
            this.mnuSettings.Click += new System.EventHandler(this.mnuSettings_Click);
            // 
            // mnuRotation
            // 
            this.mnuRotation.Name = "mnuRotation";
            this.mnuRotation.Size = new System.Drawing.Size(128, 22);
            this.mnuRotation.Text = "Rotation...";
            this.mnuRotation.Click += new System.EventHandler(this.mnuRotation_Click);
            // 
            // viewToolStripMenuItem
            // 
            this.viewToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.mnuRefresh,
            this.mnuReset});
            this.viewToolStripMenuItem.Name = "viewToolStripMenuItem";
            this.viewToolStripMenuItem.Size = new System.Drawing.Size(44, 20);
            this.viewToolStripMenuItem.Text = "View";
            // 
            // mnuRefresh
            // 
            this.mnuRefresh.Name = "mnuRefresh";
            this.mnuRefresh.Size = new System.Drawing.Size(113, 22);
            this.mnuRefresh.Text = "Refresh";
            this.mnuRefresh.Click += new System.EventHandler(this.mnuRefresh_Click);
            // 
            // mnuReset
            // 
            this.mnuReset.Name = "mnuReset";
            this.mnuReset.Size = new System.Drawing.Size(113, 22);
            this.mnuReset.Text = "Reset";
            this.mnuReset.Click += new System.EventHandler(this.mnuReset_Click);
            // 
            // sequenceToolStripMenuItem
            // 
            this.sequenceToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.mnuVideoConversion});
            this.sequenceToolStripMenuItem.Name = "sequenceToolStripMenuItem";
            this.sequenceToolStripMenuItem.Size = new System.Drawing.Size(70, 20);
            this.sequenceToolStripMenuItem.Text = "Sequence";
            // 
            // mnuVideoConversion
            // 
            this.mnuVideoConversion.Name = "mnuVideoConversion";
            this.mnuVideoConversion.Size = new System.Drawing.Size(176, 22);
            this.mnuVideoConversion.Text = "Video Conversion...";
            this.mnuVideoConversion.Click += new System.EventHandler(this.mnuVideoConversion_Click);
            // 
            // staStatus
            // 
            this.staStatus.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.lblPatchStatus,
            this.lblTileStatus});
            this.staStatus.Location = new System.Drawing.Point(0, 497);
            this.staStatus.Name = "staStatus";
            this.staStatus.Size = new System.Drawing.Size(723, 22);
            this.staStatus.TabIndex = 1;
            this.staStatus.Text = "statusStrip1";
            // 
            // lblPatchStatus
            // 
            this.lblPatchStatus.Margin = new System.Windows.Forms.Padding(0, 3, 60, 2);
            this.lblPatchStatus.Name = "lblPatchStatus";
            this.lblPatchStatus.Size = new System.Drawing.Size(0, 17);
            // 
            // lblTileStatus
            // 
            this.lblTileStatus.Name = "lblTileStatus";
            this.lblTileStatus.Size = new System.Drawing.Size(0, 17);
            // 
            // picImage
            // 
            this.picImage.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.picImage.BackColor = System.Drawing.Color.Black;
            this.picImage.Location = new System.Drawing.Point(0, 0);
            this.picImage.Name = "picImage";
            this.picImage.Size = new System.Drawing.Size(725, 472);
            this.picImage.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.picImage.TabIndex = 2;
            this.picImage.TabStop = false;
            this.picImage.Resize += new System.EventHandler(this.picImage_Resize);
            // 
            // dlgOpenFile
            // 
            this.dlgOpenFile.FileName = "OpenFile";
            // 
            // pnlImage
            // 
            this.pnlImage.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.pnlImage.BackColor = System.Drawing.Color.Black;
            this.pnlImage.Controls.Add(this.picImage);
            this.pnlImage.Location = new System.Drawing.Point(0, 24);
            this.pnlImage.MinimumSize = new System.Drawing.Size(10, 10);
            this.pnlImage.Name = "pnlImage";
            this.pnlImage.Size = new System.Drawing.Size(723, 470);
            this.pnlImage.TabIndex = 3;
            // 
            // tmrRedraw
            // 
            this.tmrRedraw.Interval = 200;
            this.tmrRedraw.Tick += new System.EventHandler(this.tmrRedraw_Tick);
            // 
            // Main
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(723, 519);
            this.Controls.Add(this.pnlImage);
            this.Controls.Add(this.staStatus);
            this.Controls.Add(this.menuStrip1);
            this.MainMenuStrip = this.menuStrip1;
            this.Name = "Main";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "Worlds 5";
            this.FormClosed += new System.Windows.Forms.FormClosedEventHandler(this.Main_FormClosed);
            this.Load += new System.EventHandler(this.frmMain_Load);
            this.KeyUp += new System.Windows.Forms.KeyEventHandler(this.Main_KeyUp);
            this.menuStrip1.ResumeLayout(false);
            this.menuStrip1.PerformLayout();
            this.staStatus.ResumeLayout(false);
            this.staStatus.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.picImage)).EndInit();
            this.pnlImage.ResumeLayout(false);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.MenuStrip menuStrip1;
        private System.Windows.Forms.ToolStripMenuItem mnuFile;
        private System.Windows.Forms.ToolStripMenuItem mnuLoadSphere;
        private System.Windows.Forms.ToolStripMenuItem mnuClose;
        private System.Windows.Forms.StatusStrip staStatus;
        private System.Windows.Forms.PictureBox picImage;
        private System.Windows.Forms.OpenFileDialog dlgOpenFile;
        private System.Windows.Forms.ToolStripStatusLabel lblPatchStatus;
        private System.Windows.Forms.ToolStripMenuItem viewToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem mnuRefresh;
        private System.Windows.Forms.Panel pnlImage;
        private System.Windows.Forms.ToolStripMenuItem mnuReset;
        private System.Windows.Forms.ToolStripStatusLabel lblTileStatus;
        private System.Windows.Forms.Timer tmrRedraw;
        private System.Windows.Forms.ToolStripSeparator toolStripMenuItem1;
        private System.Windows.Forms.ToolStripMenuItem settingsToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem mnuSettings;
        private System.Windows.Forms.ToolStripMenuItem mnuSaveImage;
        private System.Windows.Forms.ToolStripMenuItem mnuRotation;
        private System.Windows.Forms.ToolStripMenuItem sequenceToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem mnuVideoConversion;
        private System.Windows.Forms.ToolStripMenuItem mnuSaveSphere;
    }
}

