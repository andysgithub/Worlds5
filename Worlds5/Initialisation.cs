using System;
using System.Collections;
using System.Configuration;
using System.Data;
using System.Drawing;
using System.IO;
using System.Diagnostics;
using System.Windows.Forms;
using Microsoft.Win32;
using Model;
using Newtonsoft.Json;
using System.Collections.Generic;

namespace Worlds5 
{
	sealed public class Initialisation 
	{ 
		//  Initialise settings from property settings
		public static void LoadSettings(ref int iWidth, ref int iHeight, ref int iLeft, ref int iTop, ref string sState) 
		{
            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string settingsPath = Path.Combine(appDataPath, "Worlds5", "settings.json");

            if (!File.Exists(settingsPath))
            {
                // Save the default settings file to the app data path
                string sourceSettings = Path.Combine(Application.StartupPath, "default_settings.json");
                File.Copy(sourceSettings, settingsPath);
            }

            SettingsData settingsData = new SettingsData();

            // Load the json settings file
            using (StreamReader r = new StreamReader(settingsPath))
            {
                string jsonSettings = r.ReadToEnd();
                settingsData = JsonConvert.DeserializeObject<SettingsData>(jsonSettings);
            }

            clsSphere sphere = Model.Globals.Sphere;

            // SettingsData.User user = new settingsData.User();
            // SettingsData.Animation animation = new settingsData.Animation();
            // SettingsData.Viewing viewing = new settingsData.Viewing();
            // SettingsData.Raytracing raytracing = new settingsData.Raytracing();
            // SettingsData.Rendering rendering = new settingsData.Rendering();
            // SettingsData.MainWindow mainWindow = new settingsData.MainWindow();

            SettingsData.RootObject settingsRoot = new settingsData.RootObject();

            SettingsData.User user = settingsRoot.User;
            SettingsData.Animation animation = settingsRoot.Animation;
            SettingsData.Viewing viewing = settingsRoot.Viewing;
            SettingsData.Raytracing raytracing = settingsRoot.Raytracing;
            SettingsData.Rendering rendering = settingsRoot.Rendering;
            SettingsData.MainWindow mainWindow = settingsRoot.MainWindow;

            try
            {
                // User settings
                Globals.SetUp.NavPath = DecodeTag(user.NavPath);
                Globals.SetUp.SeqPath = DecodeTag(user.SeqPath);
                Globals.SetUp.Toolbar = user.Toolbar;
                Globals.SetUp.Labels = user.Labels; 
                Globals.SetUp.ToolTips = user.ToolTips;
                Globals.SetUp.StatusBar = user.StatusBar; 

                // Animation settings
				Globals.SetUp.FramesPerSec = animation.FramesPerSec;
                Globals.SetUp.AutoRepeat = animation.AutoRepeat;

                // Sphere Viewing window
                sphere.AngularResolution = viewing.ViewportResolution;
                sphere.Radius = viewing.SphereRadius;
                sphere.CentreLatitude = viewing.CentreLatitude;
                sphere.CentreLongitude = viewing.CentreLongitude;
                sphere.VerticalView = viewing.VerticalView;
                sphere.HorizontalView = viewing.HorizontalView;

                // Raytracing
                sphere.SamplingInterval = raytracing.SamplingInterval;
                sphere.SurfaceThickness = raytracing.SurfaceThickness;
                sphere.RayPoints = raytracing.RayPoints;
                sphere.MaxSamples = raytracing.MaxSamples;
                sphere.BoundaryInterval = raytracing.BoundaryInterval;
                sphere.BinarySearchSteps = raytracing.BinarySearchSteps;
                sphere.ActiveIndex = raytracing.ActiveIndex;

                // Rendering
                sphere.ExposureValue = rendering.ExposureValue;
                sphere.Saturation = rendering.Saturation;
                sphere.StartDistance = rendering.StartDistance;
                sphere.EndDistance = rendering.EndDistance;
                sphere.SurfaceContrast = rendering.SurfaceContrast;
                sphere.LightingAngle = rendering.LightingAngle;
                Globals.SetUp.BitmapWidth = rendering.BitmapWidth;
                Globals.SetUp.BitmapHeight = rendering.BitmapHeight;

                sState = mainWindow.MainState;
                iWidth = mainWindow.MainWidth;
                iHeight = mainWindow.MainHeight;
                iLeft = mainWindow.MainLeft;
                if (Screen.AllScreens.GetUpperBound(0) == 0 && iLeft > Screen.PrimaryScreen.Bounds.Width)
                {
                    iLeft = Screen.PrimaryScreen.Bounds.Width - iWidth;
                }
				iTop = mainWindow.MainTop; 
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
            SettingsData settingsData = new SettingsData();

            // SettingsData.User user = new settingsData.User();
            // SettingsData.Animation animation = new settingsData.Animation();
            // SettingsData.Viewing viewing = new settingsData.Viewing();
            // SettingsData.Raytracing raytracing = new settingsData.Raytracing();
            // SettingsData.Rendering rendering = new settingsData.Rendering();
            // SettingsData.MainWindow mainWindow = new settingsData.MainWindow();

            SettingsData.RootObject settingsRoot = new settingsData.RootObject();

            SettingsData.User user = settingsRoot.User;
            SettingsData.Animation animation = settingsRoot.Animation;
            SettingsData.Viewing viewing = settingsRoot.Viewing;
            SettingsData.Raytracing raytracing = settingsRoot.Raytracing;
            SettingsData.Rendering rendering = settingsRoot.Rendering;
            SettingsData.MainWindow mainWindow = settingsRoot.MainWindow;

            try
			{
                user.NavPath = Globals.SetUp.NavPath;
                user.SeqPath = Globals.SetUp.SeqPath;

                user.Toolbar = Globals.SetUp.Toolbar;
                user.Labels = Globals.SetUp.Labels;
                user.ToolTips = Globals.SetUp.ToolTips;
                user.StatusBar = Globals.SetUp.StatusBar;

                // Animation settings
                animation.FramesPerSec = Globals.SetUp.FramesPerSec;
                animation.AutoRepeat = Globals.SetUp.AutoRepeat;

                // Sphere Viewing window
                viewing.ViewportResolution = sphere.AngularResolution;
                viewing.SphereRadius = sphere.Radius;
                viewing.CentreLatitude = sphere.CentreLatitude;
                viewing.CentreLongitude = sphere.CentreLongitude;
                viewing.VerticalView = sphere.VerticalView;
                viewing.HorizontalView = sphere.HorizontalView;

                // Raytracing
                raytracing.SamplingInterval = sphere.SamplingInterval;
                raytracing.SurfaceThickness = sphere.SurfaceThickness;
                raytracing.RayPoints = sphere.RayPoints;
                raytracing.MaxSamples = sphere.MaxSamples;
                raytracing.BoundaryInterval = sphere.BoundaryInterval;
                raytracing.BinarySearchSteps = sphere.BinarySearchSteps;
                raytracing.ActiveIndex = sphere.ActiveIndex;

                // Rendering
                rendering.ExposureValue = sphere.ExposureValue;
                rendering.Saturation = sphere.Saturation;
                rendering.SurfaceContrast = sphere.SurfaceContrast;
                rendering.StartDistance = sphere.StartDistance;
                rendering.EndDistance = sphere.EndDistance;
                rendering.LightingAngle = sphere.LightingAngle;
                rendering.BitmapWidth = Globals.SetUp.BitmapWidth;
                rendering.BitmapHeight = Globals.SetUp.BitmapHeight;

                if (fwsState != FormWindowState.Minimized) 
				{
                    mainWindow.MainState = fwsState.ToString();
					if (fwsState != FormWindowState.Maximized) 
					{
                        mainWindow.MainWidth = iWidth;
                        mainWindow.MainHeight = iHeight;
                        mainWindow.MainLeft = iLeft;
                        mainWindow.MainTop = iTop;
					} 
				} 
			}
			catch
			{
			}

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string settingsPath = Path.Combine(appDataPath, "Worlds5", "settings.json");

            // Save the json file
            using (StreamWriter w = new StreamWriter(settingsPath))
            {
                string settingsJson = JsonConvert.SerializeObject<SettingsData>(jsonSettings);
                w.Write(settingsJson);
                
            }
		} 
	} 
}
