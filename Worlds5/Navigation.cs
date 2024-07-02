using System;
using System.IO;
using System.Collections;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Diagnostics;
using System.Windows.Forms;
using Model;
using Newtonsoft.Json;
using System.IO.Compression;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Runtime.Serialization.Formatters.Binary;

namespace Worlds5
{
    public static class Navigation
    {
        public static string LocationName;

        public static bool Navigate(string Address)
        {
            // Check that address is available if attempting to load file
            if (Address != "")
            {
                // Load navigation file from Address if known
                if (File.Exists(Address))
                {
                    // Load navigation parameters
                    if (LoadData(Address))
                    {
                        return true;
                    }
                }
                else
                {
                    MessageBox.Show("The file is not available:\n" + Address, "Navigation Error 2",
                                    MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
                }
            }
            return false;
        }

        //  Read navigation data into RenderImage class
        private static bool LoadData(string spherePath)
        {
            try
            {
                clsSphere sphere = Model.Globals.Sphere;
                SphereData.RootObject sphereRoot = new SphereData.RootObject();

                // Load the json sphere file
                using (StreamReader r = new StreamReader(spherePath))
                {
                    string jsonSettings = r.ReadToEnd();
                    sphereRoot = JsonConvert.DeserializeObject<SphereData.RootObject>(jsonSettings);
                }

                SphereData.Type fileInfo = sphereRoot.Type;
                SphereData.Navigation navigation = sphereRoot.Navigation;
                SphereData.Viewing viewing = sphereRoot.Viewing;
                SphereData.Raytracing raytracing = sphereRoot.Raytracing;
                SphereData.Rendering rendering = sphereRoot.Rendering;
                SphereData.Colour colour = sphereRoot.Colour;

                // Load file-type reference and total dimensions
                string FileType = fileInfo.FileType;

                // Redimension the transformation matrix
                int dimensions = Convert.ToInt16(fileInfo.Dimensions);
                Model.Globals.Dimensions = dimensions;
                byte[] compressedData;

                // Load Navigation settings
                sphere.settings.PositionMatrix = navigation.PositionMatrix;

/*                if (navigation.RayMap != null)
                {
                    // Convert base64 string to compressed byte array
                    compressedData = Convert.FromBase64String(navigation.RayMap);
                    // Decompress byte array
                    byte[] rayData = Helpers.Decompress(compressedData);
                    // Convert byte array to ray map
                    if (rayData.Length > 0)
                    {
                        sphere.RayMap = Helpers.ConvertToRayMap(rayData);
                    }
                }*/

                // Convert base64 string to compressed byte array
                compressedData = Convert.FromBase64String(navigation.ViewportImage);
                // Decompress byte array
                byte[] imageData = Helpers.Decompress(compressedData);
                // Convert byte array to image
                if (imageData.Length > 0)
                {
                    sphere.ViewportImage = (Bitmap)Image.FromStream(new MemoryStream(imageData));
                }

                // Load Rendering settings
                if (FileType == "1.4")
                {
                    // Viewing window
                    sphere.settings.AngularResolution = viewing.AngularResolution;
                    sphere.settings.Radius = viewing.Radius;
                    sphere.settings.CentreLatitude = viewing.CentreLatitude;
                    sphere.settings.CentreLongitude = viewing.CentreLongitude;
                    sphere.settings.VerticalView = viewing.VerticalView;
                    sphere.settings.HorizontalView = viewing.HorizontalView;

                    // Raytracing
                    sphere.settings.SamplingInterval = raytracing.SamplingInterval;
                    sphere.settings.SurfaceSmoothing = raytracing.SurfaceSmoothing;
                    sphere.settings.SurfaceThickness = raytracing.SurfaceThickness;
                    sphere.settings.MaxSamples = raytracing.MaxSamples;
                    sphere.settings.BoundaryInterval = raytracing.BoundaryInterval;
                    sphere.settings.BinarySearchSteps = raytracing.BinarySearchSteps;
                    sphere.settings.Bailout = raytracing.Bailout;
                    sphere.settings.ActiveIndex = raytracing.ActiveIndex;
                    sphere.settings.CudaMode = raytracing.CudaMode;

                    // Rendering
                    sphere.settings.ExposureValue = rendering.ExposureValue;
                    sphere.settings.Saturation = rendering.Saturation;
                    sphere.settings.SurfaceContrast = rendering.SurfaceContrast;
                    sphere.settings.LightingAngle = rendering.LightingAngle;

                    // Colour
                    sphere.settings.ColourCompression = colour.ColourCompression;
                    sphere.settings.ColourOffset = colour.ColourOffset;
                }
                else
                {
                    MessageBox.Show(
                        "File type not recognised:\n" + spherePath + "\n",
                        "Navigation Error",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Exclamation);
                    return false;
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    "Error reading navigation file:\n" + spherePath + "\n" + ex.Message,
                    "Navigation Error",
                    MessageBoxButtons.RetryCancel,
                    MessageBoxIcon.Exclamation);
                return false;
            }
            return true;
        }

        public static bool SaveData(string spherePath)
        {
            clsSphere sphere = Model.Globals.Sphere;

            SphereData.Type fileInfo = new SphereData.Type();
            SphereData.Navigation navigation = new SphereData.Navigation();
            SphereData.Viewing viewing = new SphereData.Viewing();
            SphereData.Raytracing raytracing = new SphereData.Raytracing();
            SphereData.Rendering rendering = new SphereData.Rendering();
            SphereData.Colour colour = new SphereData.Colour();

            try
            {
                fileInfo.FileType = "1.4";

                int dimensions = Model.Globals.Dimensions;
                byte[] compressedData;

                navigation.PositionMatrix = sphere.settings.PositionMatrix;

/*                BinaryFormatter formatter = new BinaryFormatter();
                // Convert ray map to byte array
                MemoryStream mStream = new MemoryStream();
                formatter.Serialize(mStream, sphere.RayMap);
                byte[] buffer = mStream.ToArray();
                mStream.Close();
                // Compress byte array
                compressedData = Helpers.Compress(buffer);
                // Convert compressed data to base64 string
                navigation.RayMap = Convert.ToBase64String(compressedData);*/

                ImageConverter converter = new ImageConverter();
                // Convert image to byte array
                byte[] imageData = (byte[])converter.ConvertTo(sphere.ViewportImage, typeof(byte[]));
                // Compress byte array
                compressedData = Helpers.Compress(imageData);
                // Convert compressed data to base64 string
                navigation.ViewportImage = Convert.ToBase64String(compressedData);

                // Viewing window
                viewing.AngularResolution = sphere.settings.AngularResolution;
                viewing.Radius = sphere.settings.Radius;
                viewing.CentreLatitude = sphere.settings.CentreLatitude;
                viewing.CentreLongitude = sphere.settings.CentreLongitude;
                viewing.VerticalView = sphere.settings.VerticalView;
                viewing.HorizontalView = sphere.settings.HorizontalView;

                // Raytracing
                raytracing.SamplingInterval = sphere.settings.SamplingInterval;
                raytracing.SurfaceSmoothing = sphere.settings.SurfaceSmoothing;
                raytracing.SurfaceThickness = sphere.settings.SurfaceThickness;
                raytracing.MaxSamples = sphere.settings.MaxSamples;
                raytracing.BoundaryInterval = sphere.settings.BoundaryInterval;
                raytracing.BinarySearchSteps = sphere.settings.BinarySearchSteps;
                raytracing.Bailout = sphere.settings.Bailout;
                raytracing.ActiveIndex = sphere.settings.ActiveIndex;
                raytracing.CudaMode = sphere.settings.CudaMode;

                // Rendering
                rendering.ExposureValue = sphere.settings.ExposureValue;
                rendering.Saturation = sphere.settings.Saturation;
                rendering.SurfaceContrast = sphere.settings.SurfaceContrast;
                rendering.LightingAngle = sphere.settings.LightingAngle;

                // Colour
                colour.ColourCompression = sphere.settings.ColourCompression;
                colour.ColourOffset = sphere.settings.ColourOffset;

                SphereData.RootObject sphereRoot = new SphereData.RootObject();
                sphereRoot.Type = fileInfo;
                sphereRoot.Navigation = navigation;
                sphereRoot.Viewing = viewing;
                sphereRoot.Raytracing = raytracing;
                sphereRoot.Rendering = rendering;
                sphereRoot.Colour = colour;

                // Save the json file
                using (StreamWriter w = new StreamWriter(spherePath))
                {
                    string jsonSettings = JsonConvert.SerializeObject(sphereRoot, Formatting.Indented);
                    w.Write(jsonSettings);

                }
            }
            catch (Exception ex)
            {
                int ErrorNo = (int)(
                    MessageBox.Show(
                        "Couldn't write data to the file:\n" + spherePath + "\n\nError reported: " + ex.Message + "\n",
                        "Save Data Error",
                        MessageBoxButtons.RetryCancel,
                        MessageBoxIcon.Exclamation));

                if (ErrorNo == (int)DialogResult.Cancel)
                {
                    return false;
                }
            }
            return true;
        }
    }
}
