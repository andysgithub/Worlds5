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
            // TODO: Load the json file

            clsSphere sphere = Model.Globals.Sphere;

            try
			{
                // User settings
                Globals.SetUp.NavPath = DecodeTag(SettingsData.User.NavPath);
                Globals.SetUp.SeqPath = DecodeTag(SettingsData.User.SeqPath);
                Globals.SetUp.Toolbar = SettingsData.User.Toolbar;
                Globals.SetUp.Labels = SettingsData.User.Labels; 
                Globals.SetUp.ToolTips = SettingsData.User.ToolTips;
                Globals.SetUp.StatusBar = SettingsData.User.StatusBar; 

                // Animation settings
				Globals.SetUp.FramesPerSec = SettingsData.Animation.FramesPerSec;
                Globals.SetUp.AutoRepeat = SettingsData.Animation.AutoRepeat;

                // Sphere Viewing window
                sphere.AngularResolution = SettingsData.Viewing.ViewportResolution;
                sphere.Radius = SettingsData.Viewing.SphereRadius;
                sphere.CentreLatitude = SettingsData.Viewing.CentreLatitude;
                sphere.CentreLongitude = SettingsData.Viewing.CentreLongitude;
                sphere.VerticalView = SettingsData.Viewing.VerticalView;
                sphere.HorizontalView = SettingsData.Viewing.HorizontalView;

                // Raytracing
                sphere.SamplingInterval = SettingsData.Raytracing.SamplingInterval;
                sphere.SurfaceThickness = SettingsData.Raytracing.SurfaceThickness;
                sphere.RayPoints = SettingsData.Raytracing.RayPoints;
                sphere.MaxSamples = SettingsData.Raytracing.MaxSamples;
                sphere.BoundaryInterval = SettingsData.Raytracing.BoundaryInterval;
                sphere.BinarySearchSteps = SettingsData.Raytracing.BinarySearchSteps;
                sphere.ActiveIndex = SettingsData.Raytracing.ActiveIndex;

                // Rendering
                sphere.ExposureValue = SettingsData.Rendering.ExposureValue;
                sphere.Saturation = SettingsData.Rendering.Saturation;
                sphere.StartDistance = SettingsData.Rendering.StartDistance;
                sphere.EndDistance = SettingsData.Rendering.EndDistance;
                sphere.SurfaceContrast = SettingsData.Rendering.SurfaceContrast;
                sphere.LightingAngle = SettingsData.Rendering.LightingAngle;
                Globals.SetUp.BitmapWidth = SettingsData.Rendering.BitmapWidth;
                Globals.SetUp.BitmapHeight = SettingsData.Rendering.BitmapHeight;

                sState = SettingsData.MainWindow.MainState;
                iWidth = SettingsData.MainWindow.MainWidth;
                iHeight = SettingsData.MainWindow.MainHeight;
                iLeft = SettingsData.MainWindow.MainLeft;
                if (Screen.AllScreens.GetUpperBound(0) == 0 && iLeft > Screen.PrimaryScreen.Bounds.Width)
                {
                    iLeft = Screen.PrimaryScreen.Bounds.Width - iWidth;
                }
				iTop = SettingsData.MainWindow.MainTop; 
			}
			catch  
			{ 
			} 
		}

        private static string DecodeTag(string setting)
        {
            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string decodedSetting = setting;

            decodedSetting = decodedSetting.Replace("%APPDATA%", appDataPath);
            decodedSetting = decodedSetting.Replace("%APPSTART%", Application.StartupPath);
            return decodedSetting;
        }
        
		//  Save settings to property settings
		public static void SaveSettings(int iWidth, int iHeight, int iLeft, int iTop, FormWindowState fwsState) 
		{
            clsSphere sphere = Model.Globals.Sphere;

            try
			{
                SettingsData.User.NavPath = Globals.SetUp.NavPath;
                SettingsData.User.SeqPath = Globals.SetUp.SeqPath;

                SettingsData.User.Toolbar = Globals.SetUp.Toolbar;
                SettingsData.User.Labels = Globals.SetUp.Labels;
                SettingsData.User.ToolTips = Globals.SetUp.ToolTips;
                SettingsData.User.StatusBar = Globals.SetUp.StatusBar;

                // Animation settings
                SettingsData.Animation.FramesPerSec = Globals.SetUp.FramesPerSec;
                SettingsData.Animation.AutoRepeat = Globals.SetUp.AutoRepeat;

                // Sphere Viewing window
                SettingsData.Viewing.ViewportResolution = sphere.AngularResolution;
                SettingsData.Viewing.SphereRadius = sphere.Radius;
                SettingsData.Viewing.CentreLatitude = sphere.CentreLatitude;
                SettingsData.Viewing.CentreLongitude = sphere.CentreLongitude;
                SettingsData.Viewing.VerticalView = sphere.VerticalView;
                SettingsData.Viewing.HorizontalView = sphere.HorizontalView;

                // Raytracing
                SettingsData.Raytracing.SamplingInterval = sphere.SamplingInterval;
                SettingsData.Raytracing.SurfaceThickness = sphere.SurfaceThickness;
                SettingsData.Raytracing.RayPoints = sphere.RayPoints;
                SettingsData.Raytracing.MaxSamples = sphere.MaxSamples;
                SettingsData.Raytracing.BoundaryInterval = sphere.BoundaryInterval;
                SettingsData.Raytracing.BinarySearchSteps = sphere.BinarySearchSteps;
                SettingsData.Raytracing.ActiveIndex = sphere.ActiveIndex;

                // Rendering
                SettingsData.Rendering.ExposureValue = sphere.ExposureValue;
                SettingsData.Rendering.Saturation = sphere.Saturation;
                SettingsData.Rendering.SurfaceContrast = sphere.SurfaceContrast;
                SettingsData.Rendering.StartDistance = sphere.StartDistance;
                SettingsData.Rendering.EndDistance = sphere.EndDistance;
                SettingsData.Rendering.LightingAngle = sphere.LightingAngle;
                SettingsData.Rendering.BitmapWidth = Globals.SetUp.BitmapWidth;
                SettingsData.Rendering.BitmapHeight = Globals.SetUp.BitmapHeight;

                if (fwsState != FormWindowState.Minimized) 
				{
                    SettingsData.MainWindow.MainState = fwsState.ToString();
					if (fwsState != FormWindowState.Maximized) 
					{
                        SettingsData.MainWindow.MainWidth = iWidth;
                        SettingsData.MainWindow.MainHeight = iHeight;
                        SettingsData.MainWindow.MainLeft = iLeft;
                        SettingsData.MainWindow.MainTop = iTop;
					} 
				} 
			}
			catch
			{
			}

            // TODO: Save the json file
		} 
	} 
}
