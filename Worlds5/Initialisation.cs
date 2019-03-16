using System;
using System.Collections;
using System.Configuration;
using System.Data;
using System.Drawing;
using System.Diagnostics;
using System.Windows.Forms;
using Microsoft.Win32;
using Model;

namespace Worlds5 
{
	sealed public class Initialisation 
	{ 
		//  Initialise settings from property settings
		public static void LoadSettings(ref int iWidth, ref int iHeight, ref int iLeft, ref int iTop, ref string sState) 
		{
            clsSphere sphere = Model.Globals.Sphere;

			try
			{
                Globals.SetUp.NavPath = DecodeTag(Properties.Settings.Default.NavPath);
                Globals.SetUp.SeqPath = DecodeTag(Properties.Settings.Default.SeqPath);
                Globals.SetUp.CachePath = DecodeTag(Properties.Settings.Default.CachePath);

                // Customisation
                Globals.SetUp.Toolbar = Properties.Settings.Default.Toolbar;
                Globals.SetUp.Labels = Properties.Settings.Default.Labels; 
				Globals.SetUp.AddressBar = false;
                Globals.SetUp.ToolTips = Properties.Settings.Default.ToolTips;
                Globals.SetUp.StatusBar = Properties.Settings.Default.StatusBar;
                Globals.SetUp.HomeAddress = Properties.Settings.Default.HomeAddress; 
				Globals.SetUp.BrowseLimit = Properties.Settings.Default.BrowseLimit; 

                // Animation settings
				Globals.SetUp.DefaultFrames = Properties.Settings.Default.DefaultFrames; 
				Globals.SetUp.FramesPerSec = Properties.Settings.Default.FramesPerSec;
                Globals.SetUp.AutoRepeat = Properties.Settings.Default.AutoRepeat;

                // Sphere Viewing window
                sphere.AngularResolution = Properties.Settings.Default.AngularResolution;
                sphere.Radius = Properties.Settings.Default.SphereRadius;
                sphere.CentreLatitude = Properties.Settings.Default.CentreLatitude;
                sphere.CentreLongitude = Properties.Settings.Default.CentreLongitude;
                sphere.VerticalView = Properties.Settings.Default.VerticalView;
                sphere.HorizontalView = Properties.Settings.Default.HorizontalView;

                // Raytracing
                sphere.SamplingInterval = Properties.Settings.Default.SamplingInterval;
                sphere.SurfaceThickness = 0.004;// Properties.Settings.Default.SurfaceThickness;
                sphere.RayPoints = Properties.Settings.Default.RayPoints;
                sphere.MaxSamples = Properties.Settings.Default.MaxSamples;
                sphere.BoundaryInterval = Properties.Settings.Default.BoundaryInterval;
                sphere.BinarySearchSteps = Properties.Settings.Default.BinarySearchSteps;
                sphere.ShowSurface = Properties.Settings.Default.ShowSurface;
                sphere.ShowExterior = Properties.Settings.Default.ShowExterior;

                // Rendering
                Globals.SetUp.BitmapWidth = Properties.Settings.Default.BitmapWidth;
                Globals.SetUp.BitmapHeight = Properties.Settings.Default.BitmapHeight;
                sphere.ExposureValue = Properties.Settings.Default.ExposureValue;
                sphere.Saturation = Properties.Settings.Default.Saturation;
                sphere.StartDistance = Properties.Settings.Default.StartDistance;
                sphere.EndDistance = Properties.Settings.Default.EndDistance;
                sphere.SurfaceContrast = Properties.Settings.Default.SurfaceContrast;
                sphere.LightingAngle = Properties.Settings.Default.LightingAngle;

                sState = Properties.Settings.Default.MainState;
                iWidth = Properties.Settings.Default.MainWidth;
                iHeight = Properties.Settings.Default.MainHeight;
                iLeft = Properties.Settings.Default.MainLeft;
                if (Screen.AllScreens.GetUpperBound(0) == 0 && iLeft > Screen.PrimaryScreen.Bounds.Width)
                {
                    iLeft = Screen.PrimaryScreen.Bounds.Width - iWidth;
                }
				iTop = Properties.Settings.Default.MainTop; 
			}
			catch  
			{ 
			} 
		}

        private static string DecodeTag(string setting)
        {
            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string decodedSetting = setting;

            if (setting.Contains("%APPDATA%"))
            {
                decodedSetting = decodedSetting.Replace("%APPDATA%", appDataPath);
            }
            if (setting.Contains("%APPSTART%"))
            {
                decodedSetting = decodedSetting.Replace("%APPSTART%", Application.StartupPath);
            }

            return decodedSetting;
        }
        
		//  Save settings to property settings
		public static void SaveSettings(int iWidth, int iHeight, int iLeft, int iTop, FormWindowState fwsState) 
		{
            clsSphere sphere = Model.Globals.Sphere;

            try
			{
                Properties.Settings.Default.NavPath = Globals.SetUp.NavPath;
                Properties.Settings.Default.SeqPath = Globals.SetUp.SeqPath;
                Properties.Settings.Default.CachePath = Globals.SetUp.CachePath;

                Properties.Settings.Default.BitmapWidth = Globals.SetUp.BitmapWidth;
                Properties.Settings.Default.BitmapHeight = Globals.SetUp.BitmapHeight;
                Properties.Settings.Default.Toolbar = Globals.SetUp.Toolbar;
                Properties.Settings.Default.Labels = Globals.SetUp.Labels;
                Properties.Settings.Default.ToolTips = Globals.SetUp.ToolTips;
                Properties.Settings.Default.StatusBar = Globals.SetUp.StatusBar;
                Properties.Settings.Default.HomeAddress = Globals.SetUp.HomeAddress;
                Properties.Settings.Default.BrowseLimit = Globals.SetUp.BrowseLimit;

                // Animation settings
                Properties.Settings.Default.DefaultFrames = Globals.SetUp.DefaultFrames;
                Properties.Settings.Default.FramesPerSec = Globals.SetUp.FramesPerSec;
                Properties.Settings.Default.AutoRepeat = Globals.SetUp.AutoRepeat;

                // Sphere Viewing window
                Properties.Settings.Default.AngularResolution = sphere.AngularResolution;
                Properties.Settings.Default.SphereRadius = sphere.Radius;
                Properties.Settings.Default.CentreLatitude = sphere.CentreLatitude;
                Properties.Settings.Default.CentreLongitude = sphere.CentreLongitude;
                Properties.Settings.Default.VerticalView = sphere.VerticalView;
                Properties.Settings.Default.HorizontalView = sphere.HorizontalView;

                // Raytracing
                Properties.Settings.Default.SamplingInterval = sphere.SamplingInterval;
                Properties.Settings.Default.SurfaceThickness = sphere.SurfaceThickness;
                Properties.Settings.Default.RayPoints = sphere.RayPoints;
                Properties.Settings.Default.MaxSamples = sphere.MaxSamples;
                Properties.Settings.Default.BoundaryInterval = sphere.BoundaryInterval;
                Properties.Settings.Default.BinarySearchSteps = sphere.BinarySearchSteps;
                Properties.Settings.Default.ShowSurface = sphere.ShowSurface;
                Properties.Settings.Default.ShowExterior = sphere.ShowExterior;

                // Rendering
                Properties.Settings.Default.BitmapWidth = Globals.SetUp.BitmapWidth;
                Properties.Settings.Default.BitmapHeight = Globals.SetUp.BitmapHeight;
                Properties.Settings.Default.ExposureValue = sphere.ExposureValue;
                Properties.Settings.Default.Saturation = sphere.Saturation;
                Properties.Settings.Default.SurfaceContrast = sphere.SurfaceContrast;
                Properties.Settings.Default.StartDistance = sphere.StartDistance;
                Properties.Settings.Default.EndDistance = sphere.EndDistance;
                Properties.Settings.Default.LightingAngle = sphere.LightingAngle;

				if (fwsState != FormWindowState.Minimized) 
				{
                    Properties.Settings.Default.MainState = fwsState.ToString();
					if (fwsState != FormWindowState.Maximized) 
					{ 
                        Properties.Settings.Default.MainWidth = iWidth;
                        Properties.Settings.Default.MainHeight = iHeight;
                        Properties.Settings.Default.MainLeft = iLeft;
                        Properties.Settings.Default.MainTop = iTop;
					} 
				} 
			}
			catch
			{
			}

            Properties.Settings.Default.Save();
		} 
	} 
}
